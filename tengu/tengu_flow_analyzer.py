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

    """
    TenguScene is a set of named TenguFlows.
    A named TenguFlow is a set of TengFlows sharing the same name.
    """

    def __init__(self, flow_blocks, direction_based_flows=[], ss_df=None):
        super(TenguScene, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._flow_blocks = flow_blocks
        self._flow_blocks_size = None
        self._blk_node_map = None
        self._direction_based_flows = direction_based_flows
        # data frame
        self._ss_df = SourceSinkDataFrame() if ss_df is None else ss_df
        # transient
        self._updated_flow_nodes = []

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
        js_direction_based_flows = []
        for direction_based_flow in self._direction_based_flows:
            js_direction_based_flows.append(direction_based_flow.serialize())
        js['direction_based_flows'] = js_direction_based_flows

        js['ss_df'] = self._ss_df.serialize()

        return js

    @staticmethod
    def deserialize(js):
        logging.debug('deserializing {}'.format(js))
        # direction based flows
        direction_based_flows = []
        if js.has_key('direction_based_flows'):
            direction_based_flows_js = js['direction_based_flows']
            for direction_based_flow_js in direction_based_flows_js:
                direction_based_flow = DirectionBasedFlow.deserialize(direction_based_flow_js)
                direction_based_flows.append(direction_based_flow)
        flow_blocks = (js['flow_blocks_rows'], js['flow_blocks_cols'])
        ss_df = SourceSinkDataFrame.deserialize(js['ss_df'])
        tengu_scene = TenguScene(flow_blocks, direction_based_flows=direction_based_flows, ss_df=ss_df)
        return tengu_scene

    @property
    def flow_blocks(self):
        return self._flow_blocks

    @property
    def flow_blocks_size(self):
        return self._flow_blocks_size

    def get_flow_node(self, x_blk, y_blk):
        return self._blk_node_map[y_blk][x_blk]

    @property
    def direction_based_flows(self):
        return self._direction_based_flows

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

    def mark_sink(self, tracklet, source, sink):

        src_pos = source.blk_position
        sink_pos = sink.blk_position

        ss_dict = {}
        ss_dict['x_blk_src'] = {tracklet.obj_id: src_pos[0]}
        ss_dict['y_blk_src'] = {tracklet.obj_id: src_pos[1]}
        ss_dict['x_blk_sink'] = {tracklet.obj_id: sink_pos[0]}
        ss_dict['y_blk_sink'] = {tracklet.obj_id: sink_pos[1]}
        ss_dict['frame_ix'] = {tracklet.obj_id: TenguTracker._global_updates}
        self._ss_df.append(ss_dict)

class DirectionBasedFlow(object):

    def __init__(self, group, high_priority=False, direction_range=[], angle_movements=[]):
        super(DirectionBasedFlow, self).__init__()
        self._group = group
        self._high_priority = high_priority
        # (from, to)
        self._direction_range = direction_range
        # (from, to)
        self._angle_movements = angle_movements
        self._tracklets = []

    def __repr__(self):
        return 'group={}, high_priority={}, direction_range={}, angle_movements={}'.format(self._group, self._high_priority, self._direction_range, self._angle_movements)

    def serialize(self):
        js = {}
        js['direction_from'] = self.direction_from
        js['direction_to'] = self.direction_to
        js['angle_movements_from'] = self.angle_movements_from
        js['angle_movements_to'] = self.angle_movements_to
        js['high_priority'] = 1 if self._high_priority else 0
        js['group'] = self._group
        return js

    @staticmethod
    def deserialize(js):
        logging.debug('deserializing {}'.format(js))
        js_group = js['group']
        js_high_priority = js['high_priority']
        if js.has_key('direction_from') and js.has_key('direction_to'):
            direction_range = [js['direction_from'], js['direction_to']]
        else:
            direction_range = []
        if js.has_key('angle_movements_from') and js.has_key('angle_movements_to'):
            angle_movements = [js['angle_movements_from'], js['angle_movements_to']]
        else:
            angle_movements = []
        return DirectionBasedFlow(js_group, True if js_high_priority == 1 else False, direction_range, angle_movements)

    #### properties to resemble a TenguFlow

    @property
    def group(self):
        return self._group

    def remove_tracklet(self, tracklet):
        del self._tracklets[self._tracklets.index(tracklet)]

    # own properties
    @property
    def high_priority(self):
        return self._high_priority

    @property
    def direction_from(self):
        if len(self._direction_range) == 0:
            return None
        return self._direction_range[0]

    @property
    def direction_to(self):
        if len(self._direction_range) == 0:
            return None
        return self._direction_range[1]

    @property
    def angle_movements_from(self):
        if len(self._angle_movements) == 0:
            return None
        return self._angle_movements[0]

    @property
    def angle_movements_to(self):
        if len(self._angle_movements) == 0:
            return None
        return self._angle_movements[1]

    @property
    def tracklets(self):
        return self._tracklets

    def add_tracklet(self, tracklet):
        self._tracklets.append(tracklet)

class SourceSinkDataFrame(object):

    def __init__(self, df=None):
        self._df = df

    def append(self, ss_dict):
        """ append an array of predicated values
        """
        new_df = DataFrame.from_dict(ss_dict)
        if self._df is None:
            self._df = new_df
        else:
            self._df = self._df.append(new_df)

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
        return SourceSinkDataFrame(df=pd.read_json(js, orient='records'))

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

    def __init__(self, y_blk, x_blk, position, stats=None):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        # position has to be tuple
        self._position = position
        # stats
        self._stats = StatsDataFrame()

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

    def adjacent(self, another_flow):
        return abs(self._y_blk - another_flow._y_blk) <= 1 and abs(self._x_blk - another_flow._x_blk) <= 1

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
    def __init__(self, scene_file=None, flow_blocks=(20, 30), **kwargs):
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
        self.finish_removed_tracklets(removed_tracklets)

        self._last_tracklets = current_tracklets

        self.logger.debug('update flow graph took {} s'.format(time.time() - start))

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
        
        for removed_tracklet in removed_tracklets:
            self.logger.debug('checking removed tracklet {} in {} s'.format(removed_tracklet.obj_id, time.time() - start))

            check_start = time.time()
            
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

            sink = removed_tracklet.path[-1]
            source = removed_tracklet.path[0]
            self._scene.mark_sink(removed_tracklet, source, sink)

            # check direction baesd flow
            self.check_direction_based_flow(removed_tracklet)

        self.logger.debug('finished removed {} tracklet in {} s'.format(len(removed_tracklets), time.time() - start))

    def check_direction_based_flow(self, tracklet):
        """ check direction based flow if available

        A direction based flow is only characterized by a specific range of directions.
        If its priority is high, it wins over an already assigned path based flow,
        otherwise, it is evaluated only when no path based flow is assigned.
        """
        start = time.time()
        direction_based_flows = self._scene.direction_based_flows
        for direction_based_flow in direction_based_flows:

            # check d based flow
            if direction_based_flow.direction_from is not None:
                if direction_based_flow.direction_from < tracklet.direction and tracklet.direction <= direction_based_flow.direction_to:
                    pass
                else:
                    continue

            # check angle movements if required
            angle_movements_ok = False
            if direction_based_flow.angle_movements_from is not None:
                if tracklet.angle_movement == Tracklet.angle_movement_not_available:
                    pass
                elif direction_based_flow.angle_movements_from < tracklet.angle_movement and tracklet.angle_movement <= direction_based_flow.angle_movements_to:
                    angle_movements_ok = True
            else:
                angle_movements_ok = True
            if not angle_movements_ok:
                continue

            direction_based_flow.add_tracklet(tracklet)
            tracklet.mark_removed()
            self.logger.debug('found a matching direction based flow {} for {}'.format(direction_based_flow, tracklet))
            break

        self.logger.debug('checked direction based flow in {} s'.format(time.time() - start))

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