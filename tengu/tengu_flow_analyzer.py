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

    def __init__(self, flow_blocks, clustering_threshold=10, flow_similarity_threshold=0.4):
        super(TenguScene, self).__init__()
        self.logger = logging.getLogger(__name__)
        self._flow_blocks = flow_blocks
        self._flow_blocks_size = None
        self._blk_node_map = None
        self._frame_shape = None
        # transient
        self._updated_flow_nodes = []
        # recorded(not clustered) trackelts, {tracklet: tracklet_image}
        self._tracklets = {}
        self._clustering_threshold = clustering_threshold
        self._flow_similarity_threshold = flow_similarity_threshold
        # current flows
        self._flows = []

    def initialize(self, frame_shape):
        self._frame_shape = frame_shape
        self._flow_blocks_size = (int(self._frame_shape[0]/self._flow_blocks[0]), int(self._frame_shape[1]/self._flow_blocks[1]))

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

    def record_tracklet(self, tracklet):
        """ records a tracklet, and triggers a clustering at intervals
        """
        self.build_tracklet_image(tracklet)

    def cluster_tracklets(self):
        """ cluster paths into flows
        """
        updated = False

        if len(self._tracklets) < self._clustering_threshold:
            return updated

        tracklets_by_flow = {}
        for tracklet in self._tracklets:
            tracklet_image = self._tracklets[tracklet]
            flow = self.find_similar_flow(tracklet, tracklet_image)
            if flow is None:
                # create a new flow
                flow = TenguFlow()
                flow.add_tracklet_and_images([[tracklet, tracklet_image]])
                self._flows.append(flow)
            if not tracklets_by_flow.has_key(flow):
                tracklets_by_flow[flow] = []
            tracklets_by_flow[flow].append([tracklet, tracklet_image])

        # update flows
        valid_flows = []
        for flow in self._flows:
            if not tracklets_by_flow.has_key(flow):
                tracklets_by_flow[flow] = []
            valid = flow.add_tracklet_and_images(tracklets_by_flow[flow])
            if valid:
                valid_flows.append(flow)

        # reset
        self._flows = valid_flows
        self._tracklets = {}

        updated = True

        return updated

    def find_similar_flow(self, tracklet, tracklet_image):
        """ find the most similar flow
        """
        most_similar_flow = None
        best_similarity = self._flow_similarity_threshold
        for flow in self._flows:
            # calculate similarity
            similarity = flow.similarity(tracklet, tracklet_image)
            if similarity > best_similarity:
                most_similar_flow = flow
                best_similarity = similarity

        return most_similar_flow

    def build_tracklet_image(self, tracklet):
        tracklet_image = np.zeros((self._frame_shape[0], self._frame_shape[1], 1), dtype=np.uint8)
        # draw rectangles on its path
        for i, flow_node in enumerate(tracklet.path):
            means = flow_node.means
            if means is None:
                rect = tracklet.milestones[i][1]
                w = int(rect[2])
                h = int(rect[3])
            else:
                w = int(means[12])
                h = int(means[13])
            node_position = flow_node.position
            x = max(0, int(node_position[0] + self._flow_blocks_size[0]/2 - w/2))
            y = max(0, int(node_position[1] + self._flow_blocks_size[1]/2 - h/2))
            print('x, y, w, h = {}, {}, {}, {}'.format(x, y, w, h))
            tracklet_image[y:min(y+h, self._frame_shape[0]), x:min(x+w, self._frame_shape[1])] = TenguFlow.VALUE_TRACKLET

        self._tracklets[tracklet] = tracklet_image
        print('build tracklet image of {}'.format(tracklet))

    @property
    def flows(self):
        return self._flows
    

class TenguFlow(object):

    """ TenguFlow is a representative trajectory of similar trajectories

    A TenguFlow is initialized with N TenguObjects observed in a certain period of time, 
    and each of which contains the whole history of Kalman Filter state variables, and others.

    So, the key is how to cluster them, how to measure similarities between them.
    There are some popular algorithms for this purpose.
    
    For example, Hausdorff Distance measures the maximum Euclidian distance among those between each of them.
    If, a trajectory is always complete, this could be used, but is not in reality.
    Most of trajectories are imcomplete, represent only some parts.
    But, two such different parts have to be measured more similar than one in the next lane.

    Here, Intersection Over Union is calculated with an image between one of a flow and another.
    Such an image of a flow or a trajectory consists of a series of object detection areas with some fixed values.
    Say, if it is 127, union area is summed up to 254, non-overlapped areas stay 127, and 0 otherwise.

    Union Area              = np.sum(image == 254)
    Non Overlapped Area     = np.sum(image == 127)
    Intersection Over Union = Union Area / (Union Area + Non Overlapped Area)

    And this has to be doen in both ways because one is only a part of another,
    in which case IoU gets smaller, but fine here.

    
    The next question is how to represent a flow that consists of multiple trajectories.
    
    At first thought, by averaging Tracklet's paths, build a temporal scene only regarding to those tracklets.
    Then, find the longest paths between each of source and sink, and make it a representative path.
    The above IoU calculation is also based on this representative path.
    
    But this sounds complicated, and time consuming.

    So, an easy implementation is to calculate an average of source nodes, and one of sink nodes.

    """

    VALUE_FLOW = 63
    VALUE_TRACKLET = 192

    def __init__(self, max_tracklets=10, flow_decay=0.9):
        super(TenguFlow, self).__init__()
        self.logger = logging.getLogger(__name__)

        # [x_blk, y_blk]
        self._avg_source = None
        self._avg_sink = None

        self._tracklet_and_images = []
        self._max_tracklets = max_tracklets
        self._flow_decay = flow_decay

        self._flow_image = None
        self._flow_image_binary = None

    def similarity(self, tracklet, tracklet_image):
        
        added = self._flow_image_binary + tracklet_image
        union_count = float(np.sum(added == (TenguFlow.VALUE_FLOW + TenguFlow.VALUE_TRACKLET)))
        non_overlapped_flow = np.sum(added == TenguFlow.VALUE_FLOW)
        non_overlapped_tracklet = np.sum(added == TenguFlow.VALUE_TRACKLET)

        if union_count == 0 and non_overlapped_flow == 0:
            print('zero output, added = {}'.format(added))

        iou_over_flow = union_count / (union_count + non_overlapped_flow)
        iou_over_tracklet = union_count / (union_count + non_overlapped_tracklet)

        return max(iou_over_flow, iou_over_tracklet)

    def add_tracklet_and_images(self, tracklet_and_images):

        if len(tracklet_and_images) > 0:
        
            for tracklet_and_image in tracklet_and_images:
                self._tracklet_and_images.append(tracklet_and_image)

            while len(self._tracklet_and_images) > self._tracklet_and_images:
                del self._tracklet_and_images[0]

            added = None
            added_source = None
            added_sink = None
            for tracklet_and_image in self._tracklet_and_images:

                tracklet = tracklet_and_image[0]
                tracklet_image = tracklet_and_image[1]
                source = tracklet.path[0]
                sink = tracklet.path[-1]

                if added is None:
                    # copy the one of tracklet
                    added = np.copy(tracklet_image).astype(np.float64)
                    added_source = [source.blk_position[0], source.blk_position[1]]
                    added_sink = [sink.blk_position[0], sink.blk_position[1]]
                    # binary
                    self._flow_image_binary = np.zeros_like(tracklet_image, dtype=np.uint16)

                else:

                    added += tracklet_image
                    added_source[0] += source.blk_position[0]
                    added_source[1] += source.blk_position[1]
                    added_sink[0] += sink.blk_position[0]
                    added_sink[1] += sink.blk_position[1] 

            # averaged
            total = len(self._tracklet_and_images)

            self._flow_image = added / total
            self._avg_source = {'x_blk': int(added_source[0] / total), 'y_blk': int(added_source[1] / total)}
            self._avg_sink = {'x_blk': int(added_sink[0] / total), 'y_blk': int(added_sink[1] / total)}

        else:

            # decrease
            self._flow_image *= self._flow_decay

        # binary for the next IoU
        return self.make_binary()

    def make_binary(self):
        valid = True
        # thresholding
        # 127 if more than min_threshold, 0 otherwise
        self._flow_image_binary[self._flow_image >= TenguFlow.VALUE_FLOW] = TenguFlow.VALUE_FLOW
        self._flow_image_binary[self._flow_image < TenguFlow.VALUE_FLOW] = 0

        if np.sum(self._flow_image_binary == TenguFlow.VALUE_FLOW) == 0:
            # this means this flow is not effective anymore
            valid = False

        return valid

    def serialize(self):
        js = {}
        
        js['source'] = self._avg_source
        js['sink'] = self._avg_sink

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
        if means is None:
            return False
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
        # True to save all the tracklets including untracked ones
        # False to save only the tracklets successfully tracked
        # None to save nothing
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

        updated = False

        # mark if left
        self.mark_tracklets_left(tracklets)

        current_tracklets = Set(tracklets)
        self._scene.reset_updated_flow_nodes()

        if len(self._last_tracklets) == 0:
            self.add_new_tracklets(current_tracklets)
            self._last_tracklets = current_tracklets
            return updated

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
        updated = self.finish_removed_tracklets(removed_tracklets)

        self._last_tracklets = current_tracklets

        self.logger.debug('update flow graph took {} s'.format(time.time() - start))

        return updated

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

            # save a node in this tracklet
            existing_tracklet.add_flow_node_to_path(flow_node)

            # update stats of this node
            self._scene.update_flow_node(flow_node, existing_tracklet)

        self.logger.debug('updated existing {} tracklet in {} s'.format(len(existing_tracklets), time.time() - start))

    def finish_removed_tracklets(self, removed_tracklets):

        start = time.time()

        for removed_tracklet in removed_tracklets:
            self.logger.debug('checking removed tracklet {} in {} s'.format(removed_tracklet.obj_id, time.time() - start))

            check_start = time.time()
            
            if self._save_untracked_details is not None and self._save_untracked_details:
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

            if self._save_untracked_details is not None and not self._save_untracked_details:
                # save this tracklet details
                self.save_tracklet(removed_tracklet)

            self._scene.record_tracklet(removed_tracklet)

        updated = self._scene.cluster_tracklets()

        self.logger.debug('finished removed {} tracklet in {} s'.format(len(removed_tracklets), time.time() - start))

        return updated

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