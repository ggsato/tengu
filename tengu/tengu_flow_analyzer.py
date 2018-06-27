#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, math, json, sys, traceback, copy, time, os
import cv2
import numpy as np
from operator import attrgetter
from sets import Set
import StringIO
from pandas import DataFrame
import pandas as pd

from tengu_tracker import TenguTracker, Tracklet

class TenguScene(object):

    def __init__(self, flow_blocks, clustering_threshold=10, flow_similarity_threshold=0.6, min_path_count_for_flow=3):
        super(TenguScene, self).__init__()
        self.logger = logging.getLogger(__name__)
        self._flow_blocks = flow_blocks
        self._flow_blocks_size = None
        self._blk_node_map = None
        # transient
        self._updated_flow_nodes = []
        # {sink: {source: path}}
        self._path_map = {}
        # current flows
        self._flows = []
        self._clustering_threshold = clustering_threshold
        self._flow_similarity_threshold = flow_similarity_threshold
        self._min_path_count_for_flow = min_path_count_for_flow

    def initialize(self, frame_shape):
        self._flow_blocks_size = (int(frame_shape[0]/self._flow_blocks[0]), int(frame_shape[1]/self._flow_blocks[1]))

        self._blk_node_map = {}
        for y_blk in xrange(self._flow_blocks[0]):
            self._blk_node_map[y_blk] = {}
            for x_blk in xrange(self._flow_blocks[1]):
                pos_x = int(self._flow_blocks_size[1]*x_blk + self._flow_blocks_size[1]/2)
                pos_y = int(self._flow_blocks_size[0]*y_blk + self._flow_blocks_size[0]/2)
                flow_node = TenguFlowNode(y_blk, x_blk, (pos_x, pos_y))
                self.logger.debug('created at y_blk,x_blk = {}, {} = {}'.format(y_blk, x_blk, (pos_x, pos_y)))
                self._blk_node_map[y_blk][x_blk] = flow_node

    def serialize(self):
        js = {}
        js['flow_blocks_rows'] = self._flow_blocks[0]
        js['flow_blocks_cols'] = self._flow_blocks[1]
        js['flow_blocks_size_row'] = self._flow_blocks_size[0]
        js['flow_blocks_size_col'] = self._flow_blocks_size[1]
        js_blk_node_map = {}
        for y_blk in sorted(self._blk_node_map):
            js_blk_node_map[y_blk] = {}
            for x_blk in sorted(self._blk_node_map[y_blk]):
                flow_node = self._blk_node_map[y_blk][x_blk]
                js_blk_node_map[y_blk][x_blk] = flow_node.serialize(simple=False)
        js['blk_node_map'] = js_blk_node_map

        return js

    @staticmethod
    def deserialize(js):
        self.logger.debug('deserializing {}'.format(js))
        flow_blocks = (js['flow_blocks_rows'], js['flow_blocks_cols'])
        tengu_scene = TenguScene(flow_blocks)
        return tengu_scene

    @property
    def flow_blocks(self):
        return self._flow_blocks

    @property
    def flow_blocks_size(self):
        return self._flow_blocks_size

    def get_flow_node(self, x_blk, y_blk):
        return self._blk_node_map[y_blk][x_blk]

    def update_flow_node(self, flow_node, tracklet):
        """ update through scene to keep updated ones at this time step
        """
        if tracklet is None:
            return
        flow_node.record_tracklet(tracklet)
        if not flow_node in self._updated_flow_nodes:
            self._updated_flow_nodes.append(flow_node)

    def reset_updated_flow_nodes(self):
        self._updated_flow_nodes = []

    @property
    def updated_flow_nodes(self):
        return self._updated_flow_nodes

    def record_path(self, source, sink):
        """ records a path, and triggers a clustering at intervals
        """
        if not self._path_map.has_key(sink):
            self._path_map[sink] = {}

        sink_map = self._path_map[sink]
        if not sink_map.has_key(source):
            self.logger.info('created a new TenguPath from {} to {}'.format(source.blk_position, sink.blk_position))
            sink_map[source] = TenguPath(source, sink)
        else:
            self.logger.info('incrementing count from {} to {}'.format(source.blk_position, sink.blk_position))
            sink_map[source].increment_count()
            self.logger.info('the current count = {}'.format(sink_map[source].count))

    def check_and_cluster_paths(self):
        # check total counts, and cluster if required
        self.logger.info('checking and clustering paths...')
        updated = False
        total_counts = 0
        ordered_paths = []
        total_paths = 0
        for sink in self._path_map:
            sink_map = self._path_map[sink]
            for source in sink_map:
                path = sink_map[source]
                total_paths += 1
                if path.count < self._min_path_count_for_flow:
                    # ignore this time, not reset count, so will be checked once its count is more than enough
                    continue
                total_counts += path.count
                ordered_paths.append(path)
        if total_counts > self._clustering_threshold:
            ordered_paths = sorted(ordered_paths, key=attrgetter('count'), reverse=True)
            self.cluster_paths(ordered_paths)
            updated = True
            self.logger.info('paths are clustered into {} flows'.format(len(self._flows)))
        else:
            self.logger.info('total count is {}, path count is {}, skipping clustering...'.format(total_counts, total_paths))

        return updated

    def cluster_paths(self, ordered_paths):
        """ cluster paths into flows
        """

        if len(self._flows) == 0:
            # initialize
            first_flow = TenguFlow()
            first_flow.add_path(ordered_paths[0])
            self._flows.append(first_flow)
            del ordered_paths[0]

        for path in ordered_paths:
            flow = self.find_similar_flow(path)
            if flow is None:
                # create a new flow
                flow = TenguFlow()
                self._flows.append(flow)
            flow.add_path(path)

    def find_similar_flow(self, path):
        """ find the most similar flow
        """
        most_similar_flow = None
        best_similarity = self._flow_similarity_threshold
        for flow in self._flows:
            # calculate similarity
            similarity = self.similarity(flow, path)
            if similarity > best_similarity:
                most_similar_flow = flow
                best_similarity = similarity

        return most_similar_flow

    def similarity(self, flow, path):
        """ a key algorithm to calculate a similarity between a flow and a path

        The main idea of path for a human is assumed here that:
        1. where did it go?        (sink)
        2. where did it come from? (source)

        So, details on a way from source or to sink is not important.
        This is especially true for a turning car.

        A sink is more important than a source, but which is still required.
        For example, a sink of a turning car and a car going straight of its diagonal road could be the same. 
        Or if objects are moving far away, they are converging to a single sink.

        Also note that a trajectory is usually not complete due to many reasons. 
        """
        similarity = 0

        sink_similarity = 0
        source_similarity = 0

        # path adjacency
        source_adj, sink_adj = flow.path_adjacency(path)

        # sink similarity
        sink_similarity = sink_adj        

        # source similarity
        source_similarity = source_adj

        # equally important
        similarity = (sink_similarity + source_similarity) / 2.

        return similarity

    @property
    def flows(self):
        return self._flows
    

class TenguFlow(object):

    """ TenguFlow is a cluster of TenguPaths
    """

    def __init__(self):
        super(TenguFlow, self).__init__()
        self.logger = logging.getLogger(__name__)
        # this is a collection of paths, sorted by descending order
        self._similar_paths = []

    @property
    def similar_paths(self):
        return self._similar_paths

    def add_path(self, path):
        path.set_flow(self)
        self._similar_paths.append(path)
        if len(self._similar_paths) > 2:
            self._similar_paths = sorted(self._similar_paths, key=attrgetter('past_count'), reverse=True)

    def path_adjacency(self, another_path):
        """ calculate an average adjacency for a given path 
        """
        path_count = len(self._similar_paths)
        if path_count == 0:
            return 0., 0.

        # source, sink
        adjacency = [0., 0.]
        for path in self._similar_paths:
            adjacency[0] += 1. if path.source.adjacent(another_path.source) else 0
            adjacency[1] += 1. if path.sink.adjacent(another_path.sink) else 0

        self.logger.info('path adjacency = {}, path_count = {}'.format(adjacency, path_count))

        return adjacency[0]/path_count, adjacency[1]/path_count

    def serialize(self):
        js = {}
        paths = []
        for path in self._similar_paths:
            paths.append(path.serialize())
        js['paths'] = paths
        return js

class TenguPath(object):

    """ TenguPath represents a pair of TenguFlowNodes(source and sink)
    """

    def __init__(self, source, sink):
        super(TenguPath, self).__init__()
        self._source = source
        self._sink = sink
        self._count = 1
        self._past_count = 0
        self._current_flow = None

    @property
    def source(self):
        return self._source
    
    @property
    def sink(self):
        return self._sink

    @property
    def count(self):
        return self._count

    @property
    def past_count(self):
        return self._past_count

    @property
    def current_flow(self):
        return self._current_flow

    def increment_count(self):
        self._count += 1

    def set_flow(self, flow):
        if self._current_flow != flow:
            self._current_flow = flow
        self._past_count += self._count
        self._count = 0

    def serialize(self):
        js = {}
        src_blk_pos = self._source.blk_position
        js['source'] = {'x_blk': src_blk_pos[0], 'y_blk': src_blk_pos[1]}
        sink_blk_pos = self._sink.blk_position
        js['sink'] = {'x_blk': sink_blk_pos[0], 'y_blk': sink_blk_pos[1]}
        js['past_count'] = self._past_count
        return js

class StatsDataFrame(object):
    """ A wraper object of stats DataFrame
    """

    def __init__(self, df=None):
        super(StatsDataFrame, self).__init__()
        self._df = df

    def __repr__(self):
        return '{}'.format('Empty' if self._df is None else self._df.describe())

    def append(self, stats_dict):
        """ append an array of predicated values
        """
        new_df = DataFrame.from_dict(stats_dict)
        if self._df is None:
            self._df = new_df
        else:
            self._df = self._df.append(new_df)

    @property
    def std_dev_series(self):

        if self._df is None:
            return None

        return self._df.std()

    @property
    def mean_series(self):

        if self._df is None:
            return None

        return self._df.mean()

    def serialize(self):
        if self._df is None:
            return None
        return self._df.to_json(orient='records')

    @staticmethod
    def deserialize(js):
        return StatsDataFrame(df=pd.read_json(js, orient='records'))


class TenguFlowNode(object):
    """ A FlowNode keeps statistical information about its own region
    """

    def __init__(self, y_blk, x_blk, position, stats=None, adj_ratio=1.0):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        # position has to be tuple
        self._position = position
        # stats
        self._stats = StatsDataFrame()
        self._adj_ratio = adj_ratio

    def __repr__(self):
        return 'FlowNode@[{},{}], {}\n{}'.format(self._y_blk, self._x_blk, self._position, self._stats)

    def serialize(self, simple=True):
        js = {}
        js['y_blk'] = self._y_blk
        js['x_blk'] = self._x_blk
        js['position'] = self._position
        if simple:
            js['means'] = self.means
        else:
            js['stats_df'] = self._stats.serialize()
        return js

    @staticmethod
    def deserialize(js):
        return TenguFlowNode(js['y_blk'], js['x_blk'], js['position'], StatsDataFrame.deserialize(js['stats_df']))

    @property
    def blk_position(self):
        return (self._x_blk, self._y_blk)

    @property
    def position(self):
        return self._position

    def adjacent(self, another_flow_node):
        """ returns True if another flow node is within a mean size of w and h
        """
        means = self.means
        mean_w = self._adj_ratio * means[12]
        mean_h = self._adj_ratio * means[13]

        x_diff = abs(self._position[0] - another_flow_node._position[0])
        y_diff = abs(self._position[1] - another_flow_node._position[1])

        return (x_diff < mean_w) and (y_diff < mean_h)

    def distance(self, another_flow):
        return max(abs(self._y_blk - another_flow._y_blk), abs(self._x_blk - another_flow._x_blk))

    def record_tracklet(self, tracklet):
        stats = tracklet.stats
        if stats is None:
            return
        stats_dict = {}
        stats_dict['x_p'] = {tracklet.obj_id: stats[0]}
        stats_dict['x_v'] = {tracklet.obj_id: stats[1]}
        stats_dict['x_a'] = {tracklet.obj_id: stats[2]}
        stats_dict['y_p'] = {tracklet.obj_id: stats[3]}
        stats_dict['y_v'] = {tracklet.obj_id: stats[4]}
        stats_dict['y_a'] = {tracklet.obj_id: stats[5]}
        stats_dict['x_p_v'] = {tracklet.obj_id: stats[6][0]}
        stats_dict['x_v_v'] = {tracklet.obj_id: stats[7][0]}
        stats_dict['x_a_v'] = {tracklet.obj_id: stats[8][0]}
        stats_dict['y_p_v'] = {tracklet.obj_id: stats[6][1]}
        stats_dict['y_v_v'] = {tracklet.obj_id: stats[7][1]}
        stats_dict['y_a_v'] = {tracklet.obj_id: stats[8][1]}
        stats_dict['w'] = {tracklet.obj_id: tracklet.rect[2]}
        stats_dict['h'] = {tracklet.obj_id: tracklet.rect[3]}
        # this is supposed to keep items within a certain time range like the last 10 mins
        stats_dict['frame_ix'] = {tracklet.obj_id: TenguTracker._global_updates}
        self._stats.append(stats_dict)

    @property
    def std_devs(self):

        std_dev_series = self._stats.std_dev_series
        if std_dev_series is None or std_dev_series.isnull()[0]:
            return None

        return [std_dev_series['x_p'], std_dev_series['x_v'], std_dev_series['x_a'], std_dev_series['y_p'], std_dev_series['y_v'], std_dev_series['y_a'], 
                std_dev_series['w'], std_dev_series['h']]

    @property
    def means(self):

        mean_series = self._stats.mean_series
        if mean_series is None or mean_series.isnull()[0]:
            return None

        return [mean_series['x_p'], mean_series['x_v'], mean_series['x_a'], mean_series['y_p'], mean_series['y_v'], mean_series['y_a'], 
                mean_series['x_p_v'], mean_series['x_v_v'], mean_series['x_a_v'], mean_series['y_p_v'], mean_series['y_v_v'], mean_series['y_a_v'],
                mean_series['w'], mean_series['h']]

    @staticmethod
    def max_node_diff(node1, node2):
        return max(abs(node1._y_blk-node2._y_blk), abs(node1._x_blk-node2._x_blk))

class TenguFlowAnalyzer(object):

    """
    """
    def __init__(self, scene_file=None, flow_blocks=(20, 30), save_untrackled_details=False, **kwargs):
        super(TenguFlowAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._initialized = False
        self._last_tracklets = Set([])
        self._scene_file = scene_file
        # the folowings will be initialized
        self._scene = TenguScene(flow_blocks)
        self._frame_shape = None
        # save folder
        self._output_folder = 'output'
        self._save_untracked_details = save_untrackled_details

    @property
    def initialized(self):
        return self._initialized
    

    def initialize(self, frame_shape):
        if self._initialized:
            self.logger.info('already initialized')
            return
        # flow graph
        self._frame_shape = frame_shape     

        # make sure output folder exists
        if not os.path.exists(self._output_folder):
            self.logger.info('creating the output folder at {}'.format(self._output_folder))
            os.makedirs(self._output_folder)

        # scene
        if self._scene_file is not None:
            self.build_scene_from_file()
        
        self._scene.initialize(self._frame_shape)
        
        self._initialized = True
        self.logger.info('flow analyzer initialized')

    def reset(self):
        self.logger.info('resetting flow analyzer...')
        self._initialized = False

    @property
    def scene(self):
        return self._scene
    

    def update_flow_graph(self, tracklets):
        """ update a flow graph by updated tracklets
        """

        start = time.time()

        # mark if left
        self.mark_tracklets_left(tracklets)

        current_tracklets = Set(tracklets)
        self._scene.reset_updated_flow_nodes()

        if len(self._last_tracklets) == 0:
            self.add_new_tracklets(current_tracklets)
            self._last_tracklets = current_tracklets
            return

        # new tracklets
        new_tracklets = current_tracklets - self._last_tracklets
        self.logger.debug('adding {} tracklets out of {}'.format(len(new_tracklets), len(current_tracklets)))
        self.add_new_tracklets(new_tracklets)

        # existing tracklets
        existing_tracklets = current_tracklets & self._last_tracklets
        self.logger.debug('updating {} tracklets out of {}'.format(len(existing_tracklets), len(current_tracklets)))
        self.update_existing_tracklets(existing_tracklets)

        # removed tracklets
        removed_tracklets = self._last_tracklets - current_tracklets
        self.logger.debug('removing {} tracklets out of {}'.format(len(removed_tracklets), len(current_tracklets)))
        recorded = self.finish_removed_tracklets(removed_tracklets)
        flows_updated = False
        if recorded > 0:
            flows_updated = self._scene.check_and_cluster_paths()

        self._last_tracklets = current_tracklets

        self.logger.debug('update flow graph took {} s'.format(time.time() - start))

        return flows_updated

    def flow_node_at(self, x, y):
        y_blk = self.get_y_blk(y)
        x_blk = self.get_x_blk(x)
        return self._scene.get_flow_node(x_blk, y_blk)

    def get_x_blk(self, x):
        x = min(max(0., x), self._frame_shape[1])
        return min(int(x / self._scene.flow_blocks_size[1]), self._scene.flow_blocks[1]-1)

    def get_y_blk(self, y):
        y = min(max(0., y), self._frame_shape[0])
        return min(int(y / self._scene.flow_blocks_size[0]), self._scene.flow_blocks[0]-1)

    def mark_tracklets_left(self, tracklets):
        for tracklet in tracklets:
            # check if it has already left
            if tracklet.has_left:
                continue
            # check if any of rect corners has left
            has_left = tracklet.rect[0] <= 0 or tracklet.rect[1] <= 0 \
                    or (tracklet.rect[0] + tracklet.rect[2] >= self._frame_shape[1]) \
                    or (tracklet.rect[1] + tracklet.rect[3] >= self._frame_shape[0])
            if has_left:
                tracklet.mark_left()
                continue

    def add_new_tracklets(self, new_tracklets):

        start = time.time()
        
        for new_tracklet in new_tracklets:
            flow_node = self.flow_node_at(*new_tracklet.location)
            # save a node in this tracklet
            new_tracklet.add_flow_node_to_path(flow_node)

        self.logger.debug('added new {} tracklet in {} s'.format(len(new_tracklets), time.time() - start))

    def update_existing_tracklets(self, existing_tracklets):

        start = time.time()
        
        for existing_tracklet in existing_tracklets:
            self.logger.debug('checking existing tracklet {} in {} s'.format(existing_tracklet.obj_id, time.time() - start))
            if existing_tracklet.has_left or not existing_tracklet.is_confirmed:
                self.logger.debug('non-qualified tracklet check done in {} s'.format(time.time() - start))
                continue

            start_check = time.time()

            prev_flow_node = existing_tracklet.path[-1]
            flow_node = self.flow_node_at(*existing_tracklet.location)
            if prev_flow_node == flow_node:
                # no change
                self.logger.debug('{} stays on the same flow node, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
                continue

            #if self._tracker.ignore_tracklet(existing_tracklet):
            #    self.logger.debug('ignoring update of {}, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
            #    continue

            # save a node in this tracklet
            existing_tracklet.add_flow_node_to_path(flow_node)

            # update stats of this node
            self._scene.update_flow_node(flow_node, existing_tracklet)

        self.logger.debug('updated existing {} tracklet in {} s'.format(len(existing_tracklets), time.time() - start))

    def finish_removed_tracklets(self, removed_tracklets):

        start = time.time()
        
        recorded = 0
        for removed_tracklet in removed_tracklets:
            self.logger.debug('checking removed tracklet {} in {} s'.format(removed_tracklet.obj_id, time.time() - start))

            check_start = time.time()
            
            if self._save_untracked_details:
                # save this tracklet details
                self.save_tracklet(removed_tracklet)

            if removed_tracklet.speed < 0:
                self.logger.debug('{} has too short path, speed is not available, not counted, check done in {} s'.format(removed_tracklet, time.time() - check_start))
                continue

            # if this tracklet is not moving, just remove
            max_diff = TenguFlowNode.max_node_diff(removed_tracklet.path[0], removed_tracklet.path[-1])
            if max_diff < 2:
                # within adjacent blocks, this is stationally
                self.logger.debug('{} is removed, but not for counting, stationally, done in {} s'.format(removed_tracklet, time.time() - check_start))
                continue

            # still, the travel distance might have been done by prediction, then skip it
            if removed_tracklet.observed_travel_distance < max(*self._scene.flow_blocks_size)*2:
                self.logger.debug('{} has moved, but the travel distance is {} smaller than a block size {}, so being skipped, check done in {} s'.format(removed_tracklet, removed_tracklet.observed_travel_distance, max(*self._scene.flow_blocks_size)*2, time.time() - check_start))
                continue

            #if self._tracker.ignore_tracklet(removed_tracklet):
            #    self.logger.debug('{} is removed, but not for counting, within ignored directions, check done in {} s'.format(removed_tracklet, time.time() - check_start))
            #    continue

            if not self._save_untracked_details:
                # save this tracklet details
                self.save_tracklet(removed_tracklet)

            sink = removed_tracklet.path[-1]
            source = removed_tracklet.path[0]
            self._scene.record_path(source, sink)
            recorded += 1

        self.logger.debug('finished removed {} tracklet in {} s'.format(len(removed_tracklets), time.time() - start))

        return recorded

    def save_tracklet(self, tracklet):
        
        file = os.path.join(self._output_folder, '{}.txt'.format(tracklet.obj_id))
        with open(file, 'w') as f:
            f.write(tracklet.history())

    def save(self, file):
        """
        save
        """
        scene_js = self._scene.serialize()
        # and, save edges
        js = self.serialize()
        js['scene'] = scene_js
        # write
        f = open(file, 'w')
        try:
            js_string = json.dumps(js, sort_keys=True, indent=4, separators=(',', ': '))
            f.write(js_string)
        finally:
            f.close()

    def serialize(self):
        js = {}
        js['frame_shape'] = self._frame_shape
        js['scene'] = self._scene.serialize()
        return js

    def deserialize(self, js):
        frame_shape = (js['frame_shape'][0], js['frame_shape'][1], js['frame_shape'][2])
        if frame_shape != self._frame_shape:
            self.logger.error('frame shape is different {} != {}'.format(js['frame_shape'], self._frame_shape))
            raise
        self._scene = TenguScene.deserialize(js['scene'])
        self.logger.debug('deserialized scene {}'.format(self._scene))

    def load(self):
        """
        load flow_map from folder
        """
        f = open(self._scene_file, 'r')
        try:
            buf = StringIO.StringIO()
            for line in f:
                buf.write(line)
            js_string = buf.getvalue()
            buf.close()
            self.deserialize(json.loads(js_string))
        finally:
            f.close()

    def build_scene_from_file(self):
        self.load()
        self.logger.debug('build from file {}'.format(self._scene_file))