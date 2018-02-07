#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, math, json, sys, traceback, copy
import cv2
import numpy as np
import networkx as nx
from operator import attrgetter
from sets import Set
import StringIO
import seaborn as sns
# dijkstra
from heapq import heappush, heappop
from itertools import count

from .tengu_tracker import TenguTracker, Tracklet, TenguCostMatrix
from .common import draw_str

class TenguNode(object):

    # location, orientation, acceleration, speed
    _min_length = 25
    # if threshold is 0.5, the lowest similarity Smin:
    # Smin = 100
    _min_distance = 50
    # Smin = 10 degrees
    _min_angle = math.pi / 180 * 10
    # Smin = 5 per 10 frames
    _min_speed = float(_min_distance) / 10
    # Smin = 1
    _min_acceleration = float(_min_speed) / 5

    def __init__(self, tr, *argv):
        super(TenguNode, self).__init__(*argv)
        self.logger = logging.getLogger(__name__)
        self.tr = tr
        self._last_detection = None
        self._last_detected_at = TenguTracker._global_updates
        self._last_updated_at = TenguTracker._global_updates
        self._movement = None
        self._angle = None
        self._property_updated_at = -1
        self._left = False

        self._owner = None

    def __repr__(self):
        return 'node at {} detected at {} updated at {}'.format(self.tr[-1], self._last_detected_at, self._last_updated_at)

    def update_last_detected(self, detection):
        self._last_detection = detection
        self._last_detected_at = TenguTracker._global_updates

    @property
    def last_detected_at(self):
        return self._last_detected_at

    def update_tr(self, x, y):
        self.tr.append((x, y))
        self._last_updated_at = TenguTracker._global_updates

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def position(self):
        if self._rect is None:
            return None

        return [int(self._rect[0]+self._rect[2]/2), int(self._rect[1]+self._rect[3]/2)]

    @property
    def movement(self):
        return self._movement

    @property
    def angle(self):
        return self._angle

    def inside_rect(self, rect):
        x, y = self.tr[-1]
        return (x > rect[0] and x < (rect[0]+rect[2])) and (y > rect[1] and y < (rect[1]+rect[3]))

    def similarity(self, another):
        """
        similarity is measured as a weighed average of the followings
        1. location (x, y)
        this is useful to detect a node that gets stuck and stays somewhere
        location_similarity = ??
        2. orientation
        this is useful when an object moves to a different direction passes in front, and steals some nodes
        orientation_similarity = ??
        3. accelaration
        this is useful when an object moves to a similar direction passes in front, and steals some nodes
        accelaration_similarity = ??
        4. speed
        this is useful when an object moves to a similar direction at faster but constant speed passes in front, and steals some nodes
        speed_similarity = ??
        """

        if self == another:
            self.logger.debug('similarity is 1.0, the same node')
            return [1.0, 1.0, 1.0, 1.0]

        pos0 = self.tr[-1]
        pos1 = another.tr[-1]
        distance = Tracklet.compute_distance(pos0, pos1)
        if self._last_detection is not None and (TenguTracker._global_updates-self._last_detected_at)<Tracklet._min_confidence:
            # use detection
            half_rect = min(self._last_detection[2:])/2
            location_similarity = half_rect/max(half_rect, distance)
        else:
            location_similarity = TenguNode._min_distance/max(TenguNode._min_distance, distance)
        self.logger.debug('location_similarity between {} and {} is {}, distance={}'.format(pos0, pos1, location_similarity, distance))

        if len(self.tr) < TenguNode._min_length or len(another.tr) < TenguNode._min_length:
            # not reliable
            orientation_similarity = 0
            speed_similarity = 0
            acceleration_similarity = 0
        else:
            pos01 = self.tr[-1 * TenguNode._min_length]
            pos11 = another.tr[-1 * TenguNode._min_length]
            distance0 = Tracklet.compute_distance(pos0, pos01)
            distance1 = Tracklet.compute_distance(pos1, pos11)
            if distance0 > 5:
                self.logger.debug('distance0 between {} and {} is {}'.format(pos0, pos01, distance0))
                self.logger.debug('tr = {}'.format(self.tr))

            if distance0 < TenguNode._min_speed:
                # stationally
                angle0 = None
            else:
                angle0 = TenguNode.get_angle(self.tr[-1 * TenguNode._min_length], self.tr[-1])
            if distance1 < TenguNode._min_speed:
                # stationally
                angle1 = None
            else:
                angle1 = TenguNode.get_angle(another.tr[-1 * TenguNode._min_length], another.tr[-1])

            diff_angle = None
            if angle0 is None and angle1 is None:
                # both are stationally
                orientation_similarity = 1.0
            elif angle0 is None or angle1 is None:
                orientation_similarity = 0.0
            else:
                diff_angle = max(TenguNode._min_angle, math.fabs(angle0 - angle1))
                if diff_angle > math.pi:
                    diff_angle = 2*math.pi - diff_angle
                orientation_similarity = TenguNode._min_angle/diff_angle
            self.logger.debug('orientation_similarity between {} and {} is {}, diff={}'.format(angle0, angle1, orientation_similarity, diff_angle))

            diff_speed = max(TenguNode._min_speed, math.fabs(distance0 - distance1))
            speed_similarity = TenguNode._min_speed/diff_speed
            self.logger.debug('speed similarity between {} and {} is {}, diff={}'.format(distance0, distance1, speed_similarity, diff_speed))

            if len(self.tr) < TenguNode._min_length*2 or len(another.tr) < TenguNode._min_length*2:
                acceleration_similarity = 1.0
                self.logger.debug('skipping acceleration similarity calculation')
            else:
                distance00 = Tracklet.compute_distance(self.tr[0], self.tr[TenguNode._min_length])
                distance10 = Tracklet.compute_distance(another.tr[0], another.tr[TenguNode._min_length])
                acceleration0 = distance0 - distance00
                acceleration1 = distance1 - distance10
                diff_acceleration = max(TenguNode._min_acceleration, math.fabs(acceleration1 - acceleration0))
                acceleration_similarity = TenguNode._min_acceleration / diff_acceleration
                self.logger.debug('acceleration similarity between {} and {} is {}, diff={}'.format(acceleration0, acceleration1, acceleration_similarity, diff_acceleration))

            # update
            self._movement = [int((pos0[0]-pos01[0])/TenguNode._min_length), int((pos0[1]-pos01[1])/TenguNode._min_length)]
            self._angle = angle0
            self._property_updated_at = TenguTracker._global_updates

        # debug
        disable_similarity = False
        if disable_similarity:
            location_similarity = 1.0
            orientation_similarity = 1.0
            speed_similarity = 1.0
            acceleration_similarity = 1.0

        similarity = [location_similarity, orientation_similarity, speed_similarity, acceleration_similarity]

        self.logger.debug('similarity = {}'.format(similarity))

        return similarity

    @staticmethod
    def get_angle(p_from, p_to):
        diff_x = p_to[0] - p_from[0]
        diff_y = p_to[1] - p_from[1]
        # angle = (-pi, pi)
        angle = math.atan2(diff_y, diff_x)
        return angle

    def last_move(self):
        if len(self.tr) < TenguNode._min_length:
            return None

        prev = self.tr[-1]
        prev2 = self.tr[-1 * TenguNode._min_length]
        move_x = prev[0]-prev2[0]
        move_y = prev[1]-prev2[1]
        return [int(move_x/TenguNode._min_length), int(move_y/TenguNode._min_length)] 

    @property
    def has_left(self):
        return self._left

    def mark_left(self):
        """ this indicates this node has left the image
        """
        self._left = True

    @property
    def owner(self):
        return self._owner

    def set_owner(self, tracklet):
        if self._owner is not None:
            self.logger.error('{} is already owned by {}'.format(self, self._owner))
            return
        self._owner = tracklet

class KLTAnalyzer(object):

    _max_nodes = 1000
    
    def __init__(self, no_detector=False, draw_flows=False, lk_params=None, feature_params=None, count_lines=None, update_interval=1, **kwargs):
        super(KLTAnalyzer, self).__init__(**kwargs)

        self.logger= logging.getLogger(__name__)
        self._no_detector = no_detector
        self.draw_flows = draw_flows
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.count_lines = count_lines

        self.frame_idx = 0
        self.update_interval = update_interval
        self.nodes = [] 
        self.max_track_length = 100
        self.prev_gray = None
        self._last_removed_nodes = []
        self.debug = None

        # used for mask
        self.last_detections = None
        self.last_frame = None

    @property
    def last_removed_nodes(self):
        removed_nodes = self._last_removed_nodes
        self._last_removed_nodes = []
        return removed_nodes

    def analyze_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.draw_flows:
            self.debug = frame_gray.copy()
        # calculate optical flow
        if len(self.nodes) > 0:
            self.nodes = self.calculate_flow(self.prev_gray, frame_gray)
            self.logger.debug('{} nodes are currently tracked'.format(len(self.nodes)))
        # update tracking points
        if self.frame_idx % self.update_interval == 0:
            mask = self.find_corners_to_track(frame_gray)
            if self.draw_flows:
                cv2.imshow('KLT Debug - Mask', mask)
        # set prev
        self.last_frame = frame
        self.prev_gray = frame_gray
        self.frame_idx += 1

        if self.draw_flows:
            cv2.imshow('KLT Debug - Flows', self.debug)

    def calculate_flow(self, img0, img1):
        self.logger.debug('calculating flow')
        p0 = np.float32([node.tr[-1] for node in self.nodes]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_nodes = []
        for node, (x, y), good_flag in zip(self.nodes, p1.reshape(-1, 2), good):
            if not good_flag:
                self._last_removed_nodes.append(node)
                continue

            if img1.shape[0] < int(y) or img1.shape[1] < int(x) or int(y) <= 0 or int(x) <= 0:
                # this has left the area
                node.mark_left()

            if len(node.tr) > self.max_track_length:
                del node.tr[0]

            node.update_tr(x, y)
            new_nodes.append(node)

            if self.draw_flows:
                cv2.circle(self.debug, (x, y), 3, (255, 255, 255), -1)

        if self.draw_flows:
            cv2.polylines(self.debug, [np.int32(node.tr) for node in new_nodes], False, (192, 192, 192))

        return new_nodes

    def find_corners_to_track(self, frame_gray):
        self.logger.debug('finding corners')

        if self._no_detector:
            # every pixel is tracked by default
            mask = np.ones_like(frame_gray) * 255
        else:
            # every pixel is *not* tracked by default
            mask = np.zeros_like(frame_gray)
            if self.last_detections is not None:
                for detection in self.last_detections:
                    cv2.rectangle(mask, (int(detection[0]), int(detection[1])), (int(detection[0]+detection[2]), int(detection[1]+detection[3])), 255, -1)
        # don't pick up existing pixels
        for x, y in [np.int32(node.tr[-1]) for node in self.nodes]:
            cv2.circle(mask, (x, y), 20, 0, -1)
        # find good points
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
        if p is None:
            self.logger.debug('No good features')
        else:
            for x, y in np.float32(p).reshape(-1, 2):
                new_node = TenguNode([(x, y)])
                self.nodes.append(new_node)
                if len(self.nodes) > KLTAnalyzer._max_nodes:
                    self._last_removed_nodes.append(self.nodes[0])
                    del self.nodes[0]

        return mask

class TenguScene(object):

    """
    TenguScene is a set of named TenguFlows.
    A named TenguFlow is a set of TengFlows sharing the same name.
    """

    def __init__(self):
        super(TenguScene, self).__init__()
        self._flow_map = {}

    def set_flows(self, flows):
        self._flow_map = {}
        for flow in flows:
            if not self._flow_map.has_key(flow.name):
                self._flow_map[flow.name] = []
            self._flow_map[flow.name].append(flow)

    @property
    def flows(self):
        flows = []
        for name in self.flow_names:
            for flow in self.named_flows(name):
                flows.append(flow)
        return flows

    @property
    def flow_names(self):
        return self._flow_map.keys()

    def named_flows(self, name):
        return self._flow_map[name]

    def serialize(self):
        js = {}
        flows_js = []
        for name in self._flow_map:
            flows = self._flow_map[name]
            for flow in flows:
                flows_js.append(flow.serialize())
        js['flows'] = flows_js
        return js

    @staticmethod
    def deserialize(js, blk_node_map):
        logging.debug('deserializing {}'.format(js))
        flows_js = js['flows']
        flows = []
        for flow_js in flows_js:
            flows.append(TenguFlow.deserialize(flow_js, blk_node_map))
        tengu_scene = TenguScene()
        tengu_scene.set_flows(flows)
        return tengu_scene

class TenguFlow(object):

    """
    TenguFlow represents a flow of typical movments by a particular type of objects,
    and whichi is characterized by its source, path, and sink.
    """
    min_source_sink_similarity_dist = 2.0

    def __init__(self, source=None, sink=None, path=[], name='default', group='default'):
        super(TenguFlow, self).__init__()
        self._source = source
        self._sink = sink
        self._path = path
        self._name = name
        self._group = group

        # transient attributes

        # direction
        self._direction = TenguNode.get_angle(self._source.position, self._sink.position)

        # a set of tracklets currently on this flow
        # can be sorted by Tracklet#dist_to_sink
        self._tracklets = Set([])

    def __repr__(self):
        return 'id={}, name={}, group={}'.format(id(self), self._name, self._group)

    def serialize(self):
        js = {}
        js['source'] = self._source.serialize()
        js['sink'] = self._sink.serialize()
        js_path = []
        for node in self._path:
            js_path.append(node.serialize())
        js['path'] = js_path
        js['name'] = self._name
        js['group'] = self._group
        return js

    @staticmethod
    def deserialize(js, blk_node_map):
        logging.debug('deserializing {}'.format(js))
        path = []
        for js_node in js['path']:
            path.append(blk_node_map[js_node['y_blk']][js_node['x_blk']])
        js_source = js['source']
        js_sink = js['sink']
        return TenguFlow(blk_node_map[js_source['y_blk']][js_source['x_blk']], blk_node_map[js_sink['y_blk']][js_sink['x_blk']], path, js['name'], js['group'])

    @property
    def source(self):
        return self._source

    @property
    def sink(self):
        return self._sink

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def group(self):
        return self._group

    @property
    def direction(self):
        return self._direction

    def similarity(self, another_flow):
        # path similarity
        shorter = self if len(self.path) < len(another_flow.path) else another_flow
        longer = self if another_flow == shorter else another_flow
        flow_node_similarities = 0.0
        flow_nodes_in_longer_path = copy.copy(longer.path)
        for flow_node in shorter.path:
            if flow_node in longer.path:
                flow_node_similarities += 1.0
            else:
                nearest = None
                nearest_node = None
                for another_node in flow_nodes_in_longer_path:
                    distance = flow_node.distance(another_node)
                    if nearest is None:
                        nearest = distance
                        nearest_node = another_node
                    elif distance < nearest:
                        nearest = distance
                        nearest_node = another_node
                # if diff is only 1, similarity = 1.0, otherwise, 0 ~ 1.0
                flow_node_similarities += 1.0 / nearest
                del flow_nodes_in_longer_path[flow_nodes_in_longer_path.index(nearest_node)]

        path_similarity = flow_node_similarities / len(shorter.path)

        # source and sink similarity
        source_dist = TenguFlow.max_node_diff(self.source, another_flow.source)
        source_similarity = TenguFlow.min_source_sink_similarity_dist / max(TenguFlow.min_source_sink_similarity_dist, source_dist)
        sink_dist = TenguFlow.max_node_diff(self.sink, another_flow.sink)
        sink_similarity = TenguFlow.min_source_sink_similarity_dist / max(TenguFlow.min_source_sink_similarity_dist, sink_dist)
        source_sink_similarity = (source_similarity + sink_similarity) / 2

        # direction similarity
        direction_similarity = 0.0
        # a tracklet may not have its directoin, then ignore this
        if another_flow.direction is None:
            direction_similarity = 1.0
        else:
            diff_angle = math.fabs(self._direction - another_flow.direction)
            # make the diff between 0 and pi
            if diff_angle > math.pi:
                diff_angle = 2*math.pi - diff_angle
            # calculate
            if diff_angle > math.pi/2:
                # no similarity, no change
                pass
            elif diff_angle == 0:
                direction_similarity = 1.0
            else:
                # 1.0 if diff is within +-pi/18(10 degrees), otherwise, between 0.1(1/9) ~ 1.0
                direction_similarity = min(1.0, math.pi/18 / diff_angle)

        logging.debug('flow similarity of {} to {}: path={}, source={}, sink={}, direction={}'.format(self, another_flow, path_similarity, source_similarity, sink_similarity, direction_similarity))

        return (path_similarity + source_sink_similarity + direction_similarity) / 3

    @staticmethod
    def max_node_diff(node1, node2):
        return max(abs(node1._y_blk-node2._y_blk), abs(node1._x_blk-node2._x_blk))

    def put_tracklet(self, tracklet, dist_to_sink, similarity, shortest_path_for_debug=None):
        if tracklet._current_flow is not None and tracklet._current_flow != self:
            # move to a different flow
            tracklet._current_flow._tracklets.remove(tracklet)
        if not tracklet in self._tracklets:
            self._tracklets.add(tracklet)
        # set flow
        tracklet.set_flow(self, dist_to_sink, similarity, shortest_path_for_debug=shortest_path_for_debug)

    def tracklets_by_dist(self):
        """ return tracklets ordered by distance to sink in an ascending order
        """
        return sorted(self._tracklets, key=attrgetter('dist_to_sink'))

    def remove_tracklet(self, tracklet):
        if tracklet in self._tracklets:
            self._tracklets.remove(tracklet)

class TenguFlowNode(object):

    def __init__(self, y_blk, x_blk, position):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        # position has to be tuple
        self._position = position
        self._source_count = 0
        self._sink_count = 0
        # pair of source, count
        self._sources = {}
        # pair of sink, count
        self._sinks = {}

    def __repr__(self):
        return json.dumps(self.serialize())

    def serialize(self):
        js = {}
        js['id'] = id(self)
        js['y_blk'] = self._y_blk
        js['x_blk'] = self._x_blk
        js['position'] = self._position
        return js

    @property
    def source_count(self):
        return self._source_count

    def mark_source(self):
        self._source_count += 1

    @property
    def sink_count(self):
        return self._sink_count

    def mark_sink(self, source):
        self._sink_count += 1

        if not self._sources.has_key(source):
            self._sources[source] = 0
        self._sources[source] += 1

        if not source._sinks.has_key(self):
            source._sinks[self] = 0
        source._sinks[self] += 1

    @property
    def position(self):
        return self._position

    def adjacent(self, another_flow):
        return abs(self._y_blk - another_flow._y_blk) <= 1 and abs(self._x_blk - another_flow._x_blk) <= 1

    def distance(self, another_flow):
        return max(abs(self._y_blk - another_flow._y_blk), abs(self._x_blk - another_flow._x_blk))

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

    def __init__(self, detector, tracker, scene_file=None, flow_blocks=(20, 30), show_graph=True, majority_in_percent=5, initial_weight=200, min_sink_count_for_flow=10, allow_non_adjacent_edge=False, identical_flow_similarity=0.5, **kwargs):
        super(TenguFlowAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._initialized = False
        self._last_tracklets = Set([])
        self._klt_analyzer = KLTAnalyzer(no_detector=(detector is None), **kwargs)
        self._detector = detector
        self._tracker = tracker
        if self._tracker is not None:
            self._tracker.set_flow_analyzer(self)
        self._scene_file = scene_file
        self._flow_blocks = flow_blocks
        self._show_graph = show_graph
        self._majority_in_percent = majority_in_percent
        self._initial_weight = initial_weight
        self._weight_func = lambda u, v, d, prev_prev_node, s=self: s.calculate_weight(u, v, d, prev_prev_node)
        self._min_sink_count_for_flow = min_sink_count_for_flow
        self._allow_non_adjacent_edge = allow_non_adjacent_edge
        self._identical_flow_similarity = identical_flow_similarity
        # the folowings will be initialized
        self._scene = None
        self._frame_shape = None
        self._flow_graph = None
        self._flow_blocks_size = None
        self._blk_node_map = None

    def analyze_flow(self, frame, frame_no):

        if not self._initialized:
            # this means new src came in
            self.initialize(frame.shape)
            self._initialized = True

        detections = []
        class_names = []
        tracklets = []

        # A) optical flow
        self._klt_analyzer.analyze_frame(frame)
        
        if self._detector is not None:
            # B) detections
            detections, class_names = self._detector.detect(frame)

            if self._tracker is not None:
                # C) update existing tracklets with new detections
                tracklets = self._tracker.resolve_tracklets(detections, class_names)
                self.update_flow_graph(tracklets)

                if self._scene_file is None:
                    # actively build scene
                    if frame_no % (self._initial_weight * TenguFlowAnalyzer.rebuild_scene_ratio) == 0:
                        self.build_scene()

                img = self.draw_graph()
                self._last_graph_img = img

                if self._show_graph:
                    # show
                    cv2.imshow('TenguFlowAnalyzer Graph', img)
        else:
            # cleanup removed_nodes
            removed_nodes = self._klt_analyzer.last_removed_nodes
            self.logger.debug('cleaned up {} nodes'.format(len(removed_nodes)))

        if self._show_graph or self._klt_analyzer.draw_flows:
            ch = 0xFF & cv2.waitKey(1)

        return detections, class_names, tracklets, self._scene

    def print_graph(self):
        self.logger.debug('flow_graph is not None? {}'.format(self._flow_graph is not None))
        for flow_node in self._flow_graph:
            self.logger.debug('{}'.format(flow_node))

    def initialize(self, frame_shape):
        # flow graph
        # gray scale
        self._frame_shape = frame_shape
        self._flow_graph = nx.DiGraph()
        self._flow_blocks_size = (int(self._frame_shape[0]/self._flow_blocks[0]), int(self._frame_shape[1]/self._flow_blocks[1]))
        self._blk_node_map = {}

        for y_blk in xrange(self._flow_blocks[0]):
            self._blk_node_map[y_blk] = {}
            for x_blk in xrange(self._flow_blocks[1]):
                pos_x = int(self._flow_blocks_size[1]*x_blk + self._flow_blocks_size[1]/2)
                pos_y = int(self._flow_blocks_size[0]*y_blk + self._flow_blocks_size[0]/2)
                flow_node = TenguFlowNode(y_blk, x_blk, (pos_x, pos_y))
                self.logger.debug('created at y_blk,x_blk = {}, {} = {}'.format(y_blk, x_blk, (pos_x, pos_y)))
                self._flow_graph.add_node(flow_node)
                self._blk_node_map[y_blk][x_blk] = flow_node

        # scene
        if self._scene_file is not None:
            self.build_scene_from_file()
        else:
            self._scene = TenguScene()
        #self.print_graph()

    def update_flow_graph(self, tracklets):
        """
        """
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
        
        for new_tracklet in new_tracklets:
            flow_node = self.flow_node_at(*new_tracklet.center)
            flow_node.mark_source()
            new_tracklet.add_flow_node_to_path(flow_node)
            self.logger.debug('source at {}'.format(flow_node))

    def update_existing_tracklets(self, existing_tracklets):
        
        for existing_tracklet in existing_tracklets:
            prev_flow_node = existing_tracklet.path[-1]
            flow_node = self.flow_node_at(*existing_tracklet.center)
            if prev_flow_node == flow_node:
                # no change
                self.logger.debug('{} stays on the same flow node'.format(existing_tracklet))
                continue
            # update edge
            if prev_flow_node is None:
                self.logger.error('no last flow exists on {}'.format(existing_tracklet))
                raise
            # if flow_node is not adjacent of prev, skip it
            # but this could happen by a quickly moving tracklet
            # so check this only when building scene
            if self._scene_file is None and not self._allow_non_adjacent_edge and not flow_node.adjacent(prev_flow_node):
                self.logger.debug('{} is not adjacent of {}'.format(flow_node, prev_flow_node))
                continue
            
            # add flow
            existing_tracklet.add_flow_node_to_path(flow_node)
            # min_set = [source, prev_flow_node, flow_node]
            if len(existing_tracklet.path) < 5:
                # prev prev is not available
                self.logger.debug('path length is too short for modified shortest path finding for {}'.format(existing_tracklet))
                continue

            if self._tracker.ignore_tracklet(existing_tracklet):
                continue

            if self._scene_file is None:
                # update edge
                if not self._flow_graph.has_edge(prev_flow_node, flow_node):
                    self.logger.debug('making an edge from {} to {}'.format(prev_flow_node, flow_node))
                    self._flow_graph.add_edge(prev_flow_node, flow_node, weight={})
                edge = self._flow_graph[prev_flow_node][flow_node]
                prev_prev_flow_node = existing_tracklet.path[-2]
                if not edge['weight'].has_key(prev_prev_flow_node):
                    edge['weight'][prev_prev_flow_node] = 0
                edge['weight'][prev_prev_flow_node] += 1
                self.logger.debug('updating weight at {}'.format(edge))


            if not existing_tracklet.has_left:
                # check if this tracklet is on a flow
                # total cost of a shortest path tends to point to a closer sink
                # so, check the cost of the next edge on each shortest path,
                # which is correctly weighted by prev prev node
                most_similar_flow = None
                best_similarity = 0
                for flow in self._scene.flows:
                    similarity = flow.similarity(existing_tracklet)
                    self.logger.info('similarity of {} to {} is {:03.2f}'.format(flow, existing_tracklet, similarity))
                    if similarity > best_similarity:
                        most_similar_flow = flow
                        best_similarity = similarity
                self.logger.info('the most similar flow of {} is {} at {}'.format(existing_tracklet, most_similar_flow, best_similarity))
                if most_similar_flow is not None:
                    if existing_tracklet.speed > -1 and existing_tracklet.path[-1].adjacent(most_similar_flow.sink):
                        # ok, mark this as passed
                        existing_tracklet.mark_flow_passed(most_similar_flow)
                        self.logger.debug('{} passed {}'.format(existing_tracklet, flow))
                        continue
                    # TODO: filter by a threshold by lowest cost
                    path, dist_to_sink = self.find_shortest_path_and_cost(flow_node, most_similar_flow.sink)
                    if path is None:
                        # no such path exist
                        self.logger.debug('but no such path from {} to {}'.format(flow_node, most_similar_flow.sink))
                        continue
                    most_similar_flow.put_tracklet(existing_tracklet, dist_to_sink, best_similarity, shortest_path_for_debug=path)
                elif existing_tracklet._current_flow is not None:
                    # this may have left the prev flow, and yet not identified by a new
                    # set None to reset
                    existing_tracklet._current_flow.remove_tracklet(existing_tracklet)
                    existing_tracklet.set_flow(None, 0, 0, shortest_path_for_debug=None)

    def finish_removed_tracklets(self, removed_tracklets):
        
        for removed_tracklet in removed_tracklets:

            if removed_tracklet.speed < 0:
                self.logger.debug('{} has too short path, speed is not available, not counted'.format(removed_tracklet))
                if removed_tracklet._current_flow is not None:
                    removed_tracklet._current_flow.remove_tracklet(removed_tracklet)
                continue

            # if this tracklet is not moving, just remove
            max_diff = TenguFlow.max_node_diff(removed_tracklet.path[0], removed_tracklet.path[-1])
            if max_diff < 2:
                # within adjacent blocks, this is stationally
                self.logger.debug('{} is removed, but not for counting, stationally'.format(removed_tracklet))
                if removed_tracklet._current_flow is not None:
                    removed_tracklet._current_flow.remove_tracklet(removed_tracklet)
                continue

            if self._tracker.ignore_tracklet(removed_tracklet):
                self.logger.debug('{} is removed, but not for counting, within ignored directions'.format(removed_tracklet))
                if removed_tracklet._current_flow is not None:
                    removed_tracklet._current_flow.remove_tracklet(removed_tracklet)
                continue

            flow_node = self.flow_node_at(*removed_tracklet.center)
            source_node = removed_tracklet.path[0]
            if flow_node == source_node:
                self.logger.debug('same source {} and sink {}, skipped'.format(source_node, flow_node))
                if removed_tracklet._current_flow is not None:
                    removed_tracklet._current_flow.remove_tracklet(removed_tracklet)
                return
            flow_node.mark_sink(source_node)
            self.logger.debug('sink at {}'.format(flow_node))

            # flow operations for counting
            if removed_tracklet._current_flow is None:
                if removed_tracklet.passed_flow is not None:
                    # this has already passed a flow
                    removed_tracklet.passed_flow.put_tracklet(removed_tracklet, 0, 1, shortest_path_for_debug=None)
                    removed_tracklet.mark_removed()
                    self.logger.debug('{} has passed {}, and removed for counting'.format(removed_tracklet, removed_tracklet.passed_flow))
                else:
                    self.logger.debug('{} will be just removed without counting, no flow assigned...'.format(removed_tracklet))
            else:
                self.logger.debug('{} was removed for counting on {}'.format(removed_tracklet, removed_tracklet._current_flow))
                removed_tracklet.mark_removed()

    def build_scene(self):
        """
        builds a scene from the current flow_graph
        """
        # flows
        majority = int(len(self._flow_graph)/100*self._majority_in_percent)
        flows = []

        # sinks
        try:
            major_sinks = sorted(self._flow_graph, key=attrgetter('sink_count'), reverse=True)
        except:
            self.print_graph()
            raise

        for major_sink in major_sinks[:majority]:
            if major_sink.sink_count < self._min_sink_count_for_flow:
                continue
            source_nodes = major_sink._sources.keys()
            source_nodes = sorted(source_nodes, key=major_sink._sources.__getitem__, reverse=True)
            major_source = source_nodes[0]
            # path
            if major_source == major_sink:
                self.logger.error('major source {} should not be the same as sink node {}'.format(major_source, major_sink))
                return
            path, _ = self.find_shortest_path_and_cost(major_source, major_sink)
            if path is None:
                continue
            # build
            tengu_flow = TenguFlow(major_source, major_sink, path, name='{:02d}'.format(len(flows)))
            # check similarity
            unique = True
            for flow in flows:
                similarity = flow.similarity(tengu_flow)
                self.logger.debug('similarity between {} and {} = {}'.format(flow, tengu_flow, similarity)) 
                if similarity > self._identical_flow_similarity:
                    self.logger.debug('a flow from {} to {} is too similar to {}, being skipped...'.format(major_source, major_sink, flow))
                    unique = False
                    break
            if not unique:
                continue
            flows.append(tengu_flow)

        # sources
        try:
            major_sources = sorted(self._flow_graph, key=attrgetter('source_count'), reverse=True)
        except:
            self.print_graph()
            raise
        for major_source in major_sources[:majority]:
            if major_source.source_count < self._min_sink_count_for_flow:
                continue
            sink_nodes = major_source._sinks.keys()
            # sinks are set after, so check if exists at all
            if len(sink_nodes) == 0:
                continue
            sink_nodes = sorted(sink_nodes, key=major_source._sinks.__getitem__, reverse=True)
            major_sink = sink_nodes[0]
            # path
            if major_sink == major_source:
                self.logger.error('major sink {} should not be the same as source node {}'.format(major_sink, major_source))
                return
            path, _ = self.find_shortest_path_and_cost(major_source, major_sink)
            if path is None:
                continue
            # build
            tengu_flow = TenguFlow(major_source, major_sink, path, name='{:02d}'.format(len(flows)))
            # check similarity
            unique = True
            for flow in flows:
                similarity = flow.similarity(tengu_flow)
                self.logger.debug('similarity between {} and {} = {}'.format(flow, tengu_flow, similarity)) 
                if similarity > self._identical_flow_similarity:
                    self.logger.debug('a flow from {} to {} is too similar to {}, being skipped...'.format(major_source, major_sink, flow))
                    unique = False
                    break
            if not unique:
                continue
            flows.append(tengu_flow)

        # flow is not yet available
        if len(flows) == 0:
            return

        # ste flows
        self._scene.set_flows(flows)

    def find_shortest_path_and_cost(self, flow_node, sink_node):
        """
        find a shortest path from flow_node to sink_node
        a candidate of such a shortest path is first found by Dijstra,
        """
        path = None
        cost = None
        try:
            sources = {flow_node}
            paths = {source: [source] for source in sources}
            dist = TenguFlowAnalyzer.dijkstra_multisource(self._flow_graph, sources, self._weight_func, paths=paths, target=sink_node)
            #path = nx.dijkstra_path(self._flow_graph, flow_node, sink_node, weight=self._weight_func)
            path = paths[sink_node]
            cost = dist[sink_node]
        except KeyError:
            self.logger.debug('no path from {} to {}'.format(flow_node, sink_node))
            return None, None
        # too short
        if len(path) < 5:
            return None, None
        return path, cost

    @staticmethod
    def dijkstra_multisource(G, sources, weight, pred=None, paths=None, cutoff=None, target=None):
        """ A slightly modifiled version of NetworkX's dijkstra that takes previous node into consideration
        """
        G_succ = G.succ

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        for source in sources:
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                break
            for u, e in G_succ[v].items():
                if len(paths[v]) < 2:
                    prev_prev_node = None
                else:
                    prev_prev_node = paths[v][-2]
                #logging.debug('prev_prev_node = {} from path {}'.format(prev_prev_node, paths[v]))
                cost = weight(v, u, e, prev_prev_node=prev_prev_node)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
                    if vu_dist < dist[u]:
                        raise ValueError('Contradictory paths found:',
                                         'negative weights?')
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)

        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.
        return dist

    def calculate_weight(self, u, v, d, prev_prev_node=None):
        cost = None
        if prev_prev_node is None :
            # this is the first edge, put an equal weight for it
            cost = self._initial_weight
        else:
            if d['weight'].has_key(prev_prev_node):
                cost = max(0, self._initial_weight - d['weight'][prev_prev_node])
            else:
                cost = self._initial_weight
        self.logger.debug('calculating weight for {} with prev_prev_node {}, total cost = {}'.format(d, prev_prev_node, cost))
        return cost

    def save(self, file):
        """
        save
        """
        self.build_scene()
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
        edges_js = []
        for edge in self._flow_graph.edges(data=True):
            from_node = edge[0]
            to_node = edge[1]
            weight_map = edge[2]['weight']
            weights = []
            for key_node in weight_map:
                weights.append([key_node._y_blk, key_node._x_blk, weight_map[key_node]])
            edge_js = {'from': [from_node._y_blk, from_node._x_blk], 'to': [to_node._y_blk, to_node._x_blk], 'weights': weights}
            edges_js.append(edge_js)
        js['edges'] = edges_js
        return js

    def deserialize(self, js):
        frame_shape = (js['frame_shape'][0], js['frame_shape'][1], js['frame_shape'][2])
        flow_blocks = (js['flow_blocks'][0], js['flow_blocks'][1])
        #if frame_shape != self._frame_shape or flow_blocks != self._flow_blocks:
        #    raise
        self._scene = TenguScene.deserialize(js['scene'], self._blk_node_map)
        self.logger.info('deserialized scene {}'.format(self._scene))
        edges_js = js['edges']
        for edge_js in edges_js:
            from_js = edge_js['from']
            to_js = edge_js['to']
            weights = edge_js['weights']
            from_node = self._blk_node_map[from_js[0]][from_js[1]]
            to_node = self._blk_node_map[to_js[0]][to_js[1]]
            weight_map = {}
            for weight in weights:
                key_node = self._blk_node_map[weight[0]][weight[1]]
                weight_map[key_node] = weight[2]
            self.logger.debug('adding an edge from {} to {} with {}'.format(from_node, to_node, weight_map))
            self._flow_graph.add_edge(from_node, to_node, weight=weight_map)
        # check shortest path exist
        for flow in self._scene.flows:
            path, _ = self.find_shortest_path_and_cost(flow.source, flow.sink)
            if path is None:
                self.logger.error('no path exists between source {} and sink of {}'.format(flow.source, flow.sink))

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

    def draw_graph(self):
        img = np.ones(self._frame_shape, dtype=np.uint8) * 128
        diameter = min(*self._flow_blocks_size)
        flows = self._scene.flows
        palette = sns.hls_palette(len(flows))
        # choose colors for each flow
        for y_blk in xrange(self._flow_blocks[0]):
            for x_blk in xrange(self._flow_blocks[1]):
                flow_node = self._blk_node_map[y_blk][x_blk]
                if self._scene_file is None:
                    is_source = flow_node.source_count > flow_node.sink_count
                    if is_source:
                        source_count = flow_node.source_count
                        if source_count == 0:
                            continue
                        grayness = min(255, 128+source_count)
                    else:
                        sink_count = flow_node.sink_count
                        if sink_count == 0:
                            continue
                        grayness = max(0, 128-sink_count)
                    color = (grayness, grayness, grayness)
                else:
                    color = (0, 0, 0)
                r = int(diameter/8)

                cv2.circle(img, flow_node.position, r, color, -1)

                # out edges
                out_edges = self._flow_graph.out_edges(flow_node)
                for out_edge in out_edges:
                    cv2.arrowedLine(img, out_edge[0].position , out_edge[1].position, color, thickness=1, tipLength=0.3)

        # flows
        for flow in flows:
            self.logger.debug('drawing flow {} with {} lines'.format(flow, len(flow.path)))

            # color
            p_color = palette[flows.index(flow)]
            color = (int(p_color[0]*192), int(p_color[1]*192), int(p_color[2]*192))

            # polyline
            lines = []
            for n, node in enumerate(flow.path):
                # rect
                x = int(self._flow_blocks_size[1]*node._x_blk)
                y = int(self._flow_blocks_size[0]*node._y_blk)
                cv2.rectangle(img, (x, y), (x+self._flow_blocks_size[1], y+self._flow_blocks_size[0]), color, -1)
                if node == flow.path[-1]:
                    break
                lines.append(node.position)
            cv2.polylines(img, [np.int32(lines)], False, color, thickness=2)

            # arrow
            cv2.arrowedLine(img, flow.path[-2].position , flow.sink.position, color, thickness=2, tipLength=0.5)

            # number
            draw_str(img, flow.path[-2].position, 'F{}'.format(flow.name))

        # tracklets
        for tracklet in self._last_tracklets:
            lines = []
            for node in tracklet.path:
                lines.append(node.position)
            color = (192, 192, 192)
            cv2.polylines(img, [np.int32(lines)], False, color, thickness=1)
            if tracklet._shortest_path_for_debug is not None:
                shortest_lines = []
                for node in tracklet._shortest_path_for_debug:
                    shortest_lines.append(node.position)
                color = (0, 0, 255)
                cv2.polylines(img, [np.int32(shortest_lines)], False, color, thickness=2)

        return img