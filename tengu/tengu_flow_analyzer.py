#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, math, json, sys, traceback, copy, time
import cv2
import numpy as np
from operator import attrgetter
from sets import Set
import StringIO
from pandas import DataFrame
import pandas as pd

from .tengu_tracker import TenguTracker, Tracklet, TenguCostMatrix

class TenguScene(object):

    """
    TenguScene is a set of named TenguFlows.
    A named TenguFlow is a set of TengFlows sharing the same name.
    """

    def __init__(self, direction_based_flows=[]):
        super(TenguScene, self).__init__()
        self._direction_based_flows = direction_based_flows

    @property
    def direction_based_flows(self):
        return self._direction_based_flows

    def serialize(self):
        js = {}
        return js

    @staticmethod
    def deserialize(js, blk_node_map):
        logging.debug('deserializing {}'.format(js))
        # direction based flows
        direction_based_flows = []
        if js.has_key('direction_based_flows'):
            direction_based_flows_js = js['direction_based_flows']
            for direction_based_flow_js in direction_based_flows_js:
                direction_based_flow = DirectionBasedFlow.deserialize(direction_based_flow_js)
                direction_based_flows.append(direction_based_flow)
        tengu_scene = TenguScene(direction_based_flows=direction_based_flows)
        return tengu_scene

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

class StatsDataFrame(object):
    """ A wraper object of stats DataFrame
    """

    def __init__(self):
        super(StatsDataFrame, self).__init__()
        self._df = None

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


class TenguFlowNode(object):
    """ A FlowNode keeps statistical information about its own region
    """

    def __init__(self, y_blk, x_blk, position):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        # position has to be tuple
        self._position = position
        # stats
        self._stats = StatsDataFrame()

    def __repr__(self):
        return 'FlowNode@[{},{}], {}\n{}'.format(self._y_blk, self._x_blk, self._position, self._stats)

    def serialize(self):
        js = {}
        js['id'] = id(self)
        js['y_blk'] = self._y_blk
        js['x_blk'] = self._x_blk
        js['position'] = self._position
        return js

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
        stats_dict['direction'] = {tracklet.obj_id: stats[6]}
        stats_dict['frame_ix'] = {tracklet.obj_id: TenguTracker._global_updates}
        self._stats.append(stats_dict)

    @property
    def std_devs(self):

        std_dev_series = self._stats.std_dev_series
        if std_dev_series is None or std_dev_series.isnull()[0]:
            return None

        return [std_dev_series['x_p'], std_dev_series['x_v'], std_dev_series['x_a'], std_dev_series['y_p'], std_dev_series['y_v'], std_dev_series['y_a']]

    @property
    def means(self):

        mean_series = self._stats.mean_series
        if mean_series is None or mean_series.isnull()[0]:
            return None

        return [mean_series['x_p'], mean_series['x_v'], mean_series['x_a'], mean_series['y_p'], mean_series['y_v'], mean_series['y_a']]

    @staticmethod
    def max_node_diff(node1, node2):
        return max(abs(node1._y_blk-node2._y_blk), abs(node1._x_blk-node2._x_blk))

class TenguFlowAnalyzer(object):

    """
    TenguFlowAnalyzer collects a set of flows, identify similar types of flows,
    then assign each tracklet to one of such types of flows.
    A flow is characterized by its source and sink.

    TenguFlowAnalyzer holds a directed weighed graph, 
    and which has nodes in the shape of flow_blocks.
    Each flow_block is assigned its dedicated region of frame_shape respectively.

    For example, given a set of flow_blocks F = {fb0, fb1, ..., fbn},
    fbn =
    """

    rebuild_scene_ratio = 10

    def __init__(self, min_length=10, scene_file=None, flow_blocks=(20, 30), **kwargs):
        super(TenguFlowAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._initialized = False
        self._last_tracklets = Set([])
        self._tracker = TenguTracker(self, min_length)
        self._scene_file = scene_file
        self._flow_blocks = flow_blocks
        # the folowings will be initialized
        self._scene = None
        self._frame_shape = None
        self._flow_blocks_size = None
        self._blk_node_map = None

    def update_model(self, frame_shape, detections, class_names):
        self.logger.debug('updating model for the shape {}, detections = {}, class_names = {}'.format(frame_shape, detections, class_names))

        if not self._initialized:
            # this means new src came in
            self.initialize(frame_shape)
            self._initialized = True

        # update tracklets with new detections
        tracklets = self._tracker.resolve_tracklets(detections, class_names)

        self.update_flow_graph(tracklets)

        return tracklets, self._scene

    def initialize(self, frame_shape):
        if self._initialized:
            self.logger.info('already initialized')
            return
        # flow graph
        # gray scale
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

        # scene
        if self._scene_file is not None:
            self.build_scene_from_file()
        else:
            self._scene = TenguScene()
        
        self._initialized = True
        self.logger.info('flow analyzer initialized')

    def update_flow_graph(self, tracklets):
        """
        """
        start = time.time()
        current_tracklets = Set(tracklets)

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

        self.logger.info('update flow graph took {} s'.format(time.time() - start))

    def flow_node_at(self, x, y):
        y_blk = self.get_y_blk(y)
        x_blk = self.get_x_blk(x)
        return self._blk_node_map[y_blk][x_blk]

    def get_x_blk(self, x):
        x = min(max(0., x), self._frame_shape[1])
        return min(int(x / self._flow_blocks_size[1]), self._flow_blocks[1]-1)

    def get_y_blk(self, y):
        y = min(max(0., y), self._frame_shape[0])
        return min(int(y / self._flow_blocks_size[0]), self._flow_blocks[0]-1)

    def add_new_tracklets(self, new_tracklets):

        start = time.time()
        
        for new_tracklet in new_tracklets:
            flow_node = self.flow_node_at(*new_tracklet.location)
            new_tracklet.add_flow_node_to_path(flow_node)

        self.logger.info('added new {} tracklet in {} s'.format(len(new_tracklets), time.time() - start))

    def update_existing_tracklets(self, existing_tracklets):

        start = time.time()
        
        for existing_tracklet in existing_tracklets:
            self.logger.info('checking existing tracklet {} in {} s'.format(existing_tracklet.obj_id, time.time() - start))
            if existing_tracklet.has_left or not existing_tracklet.is_confirmed:
                self.logger.info('non-qualified tracklet check done in {} s'.format(time.time() - start))
                continue

            start_check = time.time()

            prev_flow_node = existing_tracklet.path[-1]
            flow_node = self.flow_node_at(*existing_tracklet.location)
            if prev_flow_node == flow_node:
                # no change
                self.logger.info('{} stays on the same flow node, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
                continue
            # update edge
            if prev_flow_node is None:
                self.logger.error('no last flow exists on {}, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
                raise

            # if flow_node is not adjacent of prev, skip it
            if not flow_node.adjacent(prev_flow_node):
                self.logger.info('skipping update of {}, not adjacent move, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
                continue

            if self._tracker.ignore_tracklet(existing_tracklet):
                self.logger.info('ignoring update of {}, check done in {} s'.format(existing_tracklet.obj_id, time.time() - start_check))
                continue

            # add flow
            existing_tracklet.add_flow_node_to_path(flow_node)

        self.logger.info('updated existing {} tracklet in {} s'.format(len(existing_tracklets), time.time() - start))

    def finish_removed_tracklets(self, removed_tracklets):

        start = time.time()
        
        for removed_tracklet in removed_tracklets:
            self.logger.info('checking removed tracklet {} in {} s'.format(removed_tracklet.obj_id, time.time() - start))

            check_start = time.time()
            if removed_tracklet.speed < 0:
                self.logger.info('{} has too short path, speed is not available, not counted, check done in {} s'.format(removed_tracklet, time.time() - check_start))
                continue

            # if this tracklet is not moving, just remove
            max_diff = TenguFlowNode.max_node_diff(removed_tracklet.path[0], removed_tracklet.path[-1])
            if max_diff < 2:
                # within adjacent blocks, this is stationally
                self.logger.info('{} is removed, but not for counting, stationally, done in {} s'.format(removed_tracklet, time.time() - check_start))
                continue

            # still, the travel distance might have been done by prediction, then skip it
            if removed_tracklet.observed_travel_distance < max(*self._flow_blocks_size)*2:
                self.logger.info('{} has moved, but the travel distance is {} smaller than a block size {}, so being skipped, check done in {} s'.format(removed_tracklet, removed_tracklet.observed_travel_distance, max(*self._flow_blocks_size)*2, time.time() - check_start))
                continue

            if self._tracker.ignore_tracklet(removed_tracklet):
                self.logger.info('{} is removed, but not for counting, within ignored directions, check done in {} s'.format(removed_tracklet, time.time() - check_start))
                continue

            sink_node = removed_tracklet.path[-1]
            source_node = removed_tracklet.path[0]
            if sink_node == source_node:
                self.logger.info('same source {} and sink {}, skipped, check done in {} s'.format(source_node, sink_node, time.time() - check_start))
                return

            # check direction baesd flow
            self.check_direction_based_flow(removed_tracklet)

        self.logger.info('finished removed {} tracklet in {} s'.format(len(removed_tracklets), time.time() - start))

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
            self.logger.info('found a matching direction based flow {} for {}'.format(direction_based_flow, tracklet))
            break

        self.logger.info('checked direction based flow in {} s'.format(time.time() - start))

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
        #
        cv2.imwrite('{}.jpg'.format(file[:file.rindex('.')]), self._last_graph_img)

    def serialize(self):
        js = {}
        js['frame_shape'] = self._frame_shape
        js['flow_blocks'] = self._flow_blocks
        return js

    def deserialize(self, js):
        frame_shape = (js['frame_shape'][0], js['frame_shape'][1], js['frame_shape'][2])
        flow_blocks = (js['flow_blocks'][0], js['flow_blocks'][1])
        if frame_shape != self._frame_shape or flow_blocks != self._flow_blocks:
            self.logger.error('frame shape is different {} != {}'.format(js['frame_shape'], self._frame_shape))
            raise
        self._scene = TenguScene.deserialize(js['scene'], self._blk_node_map)
        self.logger.info('deserialized scene {}'.format(self._scene))

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
        self.logger.info('build from file {}'.format(self._scene_file))