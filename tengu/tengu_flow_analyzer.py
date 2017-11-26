#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, math, json, sys
import cv2
import numpy as np
import networkx as nx
from operator import attrgetter
from sets import Set
import StringIO
import seaborn as sns

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
    _min_speed = 5
    # Smin = 1
    _min_acceleration = _min_speed / 5

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
        distance = TenguNode.compute_distance(pos0, pos1)
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
            distance0 = TenguNode.compute_distance(pos0, pos01)
            distance1 = TenguNode.compute_distance(pos1, pos11)
            if distance0 > 5:
                self.logger.debug('distance0 between {} and {} is {}'.format(pos0, pos01, distance0))
                self.logger.debug('tr = {}'.format(self.tr))

            if distance0 < TenguNode._min_speed:
                # stationally
                angle0 = None
            else:
                angle0 = TenguNode.get_angle(self.tr)
            if distance1 < TenguNode._min_speed:
                # stationally
                angle1 = None
            else:
                angle1 = TenguNode.get_angle(another.tr)

            diff_angle = None
            if angle0 is None and angle1 is None:
                # both are stationally
                orientation_similarity = 1.0
            elif angle0 is None or angle1 is None:
                orientation_similarity = 0.0
            else:
                diff_angle = max(TenguNode._min_angle, math.fabs(angle0 - angle1))
                if diff_angle > math.pi:
                    diff_angle -= math.pi
                orientation_similarity = TenguNode._min_angle/diff_angle
            self.logger.debug('orientation_similarity between {} and {} is {}, diff={}'.format(angle0, angle1, orientation_similarity, diff_angle))

            diff_speed = max(TenguNode._min_speed, math.fabs(distance0 - distance1))
            speed_similarity = TenguNode._min_speed/diff_speed
            self.logger.debug('speed similarity between {} and {} is {}, diff={}'.format(distance0, distance1, speed_similarity, diff_speed))

            if len(self.tr) < TenguNode._min_length*2 or len(another.tr) < TenguNode._min_length*2:
                acceleration_similarity = 1.0
                self.logger.debug('skipping acceleration similarity calculation')
            else:
                distance00 = TenguNode.compute_distance(self.tr[0], self.tr[TenguNode._min_length])
                distance10 = TenguNode.compute_distance(another.tr[0], another.tr[TenguNode._min_length])
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

        self.logger.debug('similarity = {} [{}, {}, {}, {}]'.format(similarity, location_similarity, orientation_similarity, speed_similarity, acceleration_similarity))

        return similarity

    @staticmethod
    def compute_distance(pos0, pos1):
        return math.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)

    @staticmethod
    def get_angle(tr):
        p_from = tr[-1 * TenguNode._min_length]
        p_to = tr[-1]
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
        return [int(move_x/min_length), int(move_y/min_length)] 

class KLTAnalyzer(object):

    _max_nodes = 1000
    
    def __init__(self, draw_flows=False, lk_params=None, feature_params=None, count_lines=None, **kwargs):
        super(KLTAnalyzer, self).__init__(**kwargs)

        self.logger= logging.getLogger(__name__)
        self.draw_flows = draw_flows
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.count_lines = count_lines

        self.frame_idx = 0
        self.update_interval = 10
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
            ch = 0xFF & cv2.waitKey(1)

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

        # every pixel is not tracked by default
        mask = np.zeros_like(frame_gray)
        use_detections = True
        if use_detections and self.last_detections is not None:
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

    def __init__(self, frame_shape, flow_blocks):
        super(TenguScene, self).__init__()
        self._frame_shape = frame_shape
        self._flow_blocks = flow_blocks
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
        js['frame_shape'] = self._frame_shape
        js['flow_blocks'] = self._flow_blocks
        flows_js = []
        for name in self._flow_map:
            flows = self._flow_map[name]
            for flow in flows:
                flows_js.append(flow.serialize())
        js['flows'] = flows_js
        return js

    @staticmethod
    def deserialize(js):
        logging.debug('deserializing {}'.format(js))
        flows_js = js['flows']
        flows = []
        for flow_js in flows_js:
            flows.append(TenguFlow.deserialize(flow_js))
        frame_shape = (js['frame_shape'][0], js['frame_shape'][1], js['frame_shape'][2])
        flow_blocks = (js['flow_blocks'][0], js['flow_blocks'][1])
        tengu_scene = TenguScene(frame_shape, flow_blocks)
        tengu_scene.set_flows(flows)
        return tengu_scene

    @staticmethod
    def load(file):
        """
        load flow_map from folder
        """
        f = open(file, 'r')
        try:
            buf = StringIO.StringIO()
            for line in f:
                buf.write(line)
            js_string = buf.getvalue()
            buf.close()
            tengu_scene = TenguScene.deserialize(json.loads(js_string))
            return tengu_scene
        finally:
            f.close()

    def save(self, file):
        """
        save flow_map to folder
        """
        f = open(file, 'w')
        try:
            js_string = json.dumps(self.serialize(), sort_keys=True, indent=4, separators=(',', ': '))
            f.write(js_string)
        finally:
            f.close()

class TenguFlow(object):

    """
    TenguFlow represents a flow of typical movments by a particular type of objects,
    and whichi is characterized by its source, path, and sink.
    """

    def __init__(self, source=None, sink=None, path=[], name='default'):
        super(TenguFlow, self).__init__()
        self._source = source
        self._sink = sink
        self._path = path
        self._name = name

    def __repr__(self):
        return json.dumps(self.serialize())

    def serialize(self):
        js = {}
        js['source'] = self._source.serialize()
        js['sink'] = self._sink.serialize()
        js_path = []
        for node in self._path:
            js_path.append(node.serialize())
        js['path'] = js_path
        js['name'] = self._name
        return js

    @staticmethod
    def deserialize(js):
        logging.debug('deserializing {}'.format(js))
        path = []
        for js_node in js['path']:
            path.append(TenguFlowNode.deserialize(js_node))
        return TenguFlow(TenguFlowNode.deserialize(js['source']), TenguFlowNode.deserialize(js['sink']), path, js['name'])

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

    def __repr__(self):
        return json.dumps(self.serialize())

    def serialize(self):
        js = {}
        js['id'] = id(self)
        js['y_blk'] = self._y_blk
        js['x_blk'] = self._x_blk
        js['position'] = self._position
        return js

    @staticmethod
    def deserialize(js):
        return TenguFlowNode(js['y_blk'], js['x_blk'], (js['position'][0], js['position'][1]))

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

    @property
    def position(self):
        return self._position

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

    def __init__(self, detector, tracker, scene_file=None, analyze_scene=True, flow_blocks=(20, 20), show_graph=True, majority_in_percent=5, initial_weight=100, min_sink_count_for_flow=10, **kwargs):
        super(TenguFlowAnalyzer, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._last_tracklets = Set([])
        self._klt_analyzer = KLTAnalyzer(**kwargs)
        self._detector = detector
        self._tracker = tracker
        if self._tracker is not None:
            self._tracker.set_flow_analyzer(self)
        self._scene_file = scene_file
        self._analyze_scene = analyze_scene
        self._flow_blocks = flow_blocks
        self._show_graph = show_graph
        self._majority_in_percent = majority_in_percent
        self._initial_weight = initial_weight
        self._weight_func = lambda u, v, d, s=self: s.calculate_weight(u, v, d)
        self._min_sink_count_for_flow = min_sink_count_for_flow
        # the folowings will be initialized
        self._scene = None
        self._frame_shape = None
        self._flow_graph = None
        self._flow_blocks_size = None
        self._blk_node_map = None

    def analyze_flow(self, frame, frame_no):

        if frame_no == 0:
            # this means new src came in
            self.initialize(frame.shape)

        detections = []
        tracklets = []
        scene = None

        self._klt_analyzer.analyze_frame(frame)
        
        if self._detector is not None:
            detections = self._detector.detect(frame)

            if self._tracker is not None:
                tracklets = self._tracker.resolve_tracklets(detections)
                self.update_flow_graph(tracklets)

                if self._scene_file is None:
                    if self._analyze_scene:
                        # actively build scene
                        if frame_no % self._initial_weight == 0:
                            self.build_scene()
                else:
                    self.update_scene()
                    scene = self._scene

                if self._show_graph:
                    # show
                    img = self.draw_graph()
                    cv2.imshow('TenguFlowAnalyzer Graph', img)
                    ch = 0xFF & cv2.waitKey(1)

        return detections, tracklets, scene

    def print_graph(self):
        self.logger.info('flow_graph is not None? {}'.format(self._flow_graph is not None))
        for flow_node in self._flow_graph:
            self.logger.info('{}'.format(flow_node))

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
                self.logger.info('created at y_blk,x_blk = {}, {} = {}'.format(y_blk, x_blk, (pos_x, pos_y)))
                self._flow_graph.add_node(flow_node)
                self._blk_node_map[y_blk][x_blk] = flow_node

        # scene
        if self._scene_file is not None:
            self.build_scene_from_file()
        else:
            self._scene = TenguScene(self._frame_shape, self._flow_blocks)
        self.print_graph()

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
        x = min(max(0., x), self._frame_shape[1]-1)
        return int(x / self._flow_blocks_size[1])

    def get_y_blk(self, y):
        y = min(max(0., y), self._frame_shape[0]-1)
        return int(y / self._flow_blocks_size[0])

    def add_new_tracklets(self, new_tracklets):
        
        for new_tracklet in new_tracklets:
            flow_node = self.flow_node_at(*new_tracklet.center)
            flow_node.mark_source()
            new_tracklet.flows.append(flow_node)
            self.logger.info('source at {}'.format(flow_node))

    def update_existing_tracklets(self, existing_tracklets):
        
        for existing_tracklet in existing_tracklets:
            if len(existing_tracklet.flows) < 2:
                # prev prev is not available
                continue
            prev_flow_node = existing_tracklet.flows[-1]
            flow_node = self.flow_node_at(*existing_tracklet.center)
            if prev_flow_node == flow_node:
                # no change
                continue
            # update edge
            if prev_flow_node is None:
                self.logger.error('no last flow exists on {}'.format(existing_tracklet))
                raise
            prev_prev_flow_node = existing_tracklet.flows[-2]
            if not self._flow_graph.has_edge(prev_flow_node, flow_node):
                self._flow_graph.add_edge(prev_flow_node, flow_node, weight={})
            edge = self._flow_graph[prev_flow_node][flow_node]
            if not edge['weight'].has_key(prev_prev_flow_node):
                edge['weight'][prev_prev_flow_node] = 0
            edge['weight'][prev_prev_flow_node] += 1
            # add
            existing_tracklet.flows.append(flow_node)
            self.logger.debug('updating weight at {}'.format(edge))
            # check if this tracklet is on a flow
            # total cost of a shortest path tends to point to a closer sink
            # so, check the cost of the next edge on each shortest path,
            # which is correctly weighted by prev prev node
            lowest_cost = None
            lowest_flow = None
            for flow in self._scene.flows:
                path = self.find_shortest_path(flow_node, flow.sink)
                if path is None:
                    # no such path
                    continue
                next_edge = self._flow_graph[flow_node][path[1]]
                next_cost = self.calculate_weight(None, None, next_edge, prev_prev_node=prev_flow_node)
                if next_cost is None:
                    # this means no such path
                    continue
                if lowest_cost is None:
                    lowest_cost = next_cost
                    lowest_flow = flow
                    continue
                if next_cost < lowest_cost:
                    lowest_cost = next_cost
                    lowest_flow = flow
            if lowest_flow is not None:
                # TODO: filter by a threshold by lowest cost
                existing_tracklet._current_flow = lowest_flow

    def finish_removed_tracklets(self, removed_tracklets):
        
        for removed_tracklet in removed_tracklets:
            if len(removed_tracklet.flows) == 0:
                continue
            flow_node = self.flow_node_at(*removed_tracklet.center)
            source_node = removed_tracklet.flows[0]
            if flow_node == source_node:
                self.logger.info('same source {} and sink {}, skipped'.format(source_node, flow_node))
                return
            flow_node.mark_sink(source_node)
            self.logger.info('sink at {}'.format(flow_node))

    def build_scene(self):
        """
        builds a scene from the current flow_graph
        """
        # flows
        try:
            major_sinks = sorted(self._flow_graph, key=attrgetter('sink_count'), reverse=True)
        except:
            self.print_graph()
            raise

        majority = int(len(self._flow_graph)/100*self._majority_in_percent)
        flows = []
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
            path = self.find_shortest_path(major_source, major_sink)
            if path is None:
                continue
            # build
            tengu_flow = TenguFlow(major_source, major_sink, path, name='{:02d}'.format(len(flows)))
            flows.append(tengu_flow)

        # flow is not yet available
        if len(flows) == 0:
            return

        # ste flows
        self._scene.set_flows(flows)

    def find_shortest_path(self, flow_node, sink_node):
        """
        find a shortest path from flow_node to sink_node
        a candidate of such a shortest path is first found by Dijstra,
        """
        path = None
        try:
            path = nx.dijkstra_path(self._flow_graph, flow_node, sink_node, weight=self._weight_func)
        except nx.NetworkXNoPath:
            self.logger.info('no path from {} to {}'.format(flow_node, sink_node))
            return path
        # too short
        if len(path) < 3:
            return None
        return path

    def calculate_weight(self, u, v, d, prev_prev_node=None):
        cost = None
        if prev_prev_node is None:
            cost = max(0, self._initial_weight - sum(d['weight'].values()))
        elif d['weight'].has_key(prev_prev_node):
            cost = max(0, self._initial_weight - d['weight'][prev_prev_node])
        self.logger.debug('calculating weight for {} with prev_prev_node {}, total cost = {}'.format(d, prev_prev_node, cost))
        return cost

    def build_scene_from_file(self):
        scene = TenguScene.load(self._scene_file)
        self.logger.info('build scene {} from file {}'.format(scene, self._scene_file))
        self._scene = scene

    def save_scene(self, scene_file):
        self.build_scene()
        self._scene.save(scene_file)

    def update_scene(self):
        """
        update scene with last_tracklets
        1. build cost matrix
            1-1: create a cost matrix TxF, where T is a set of Tracklets and F is a set of Flows
            cost calculation: sum of weights with regard to a Flow of edges a Tracklet passed through
        2. assign a flow to a tracklet
            2-1: find the best match by hungarian algorithm
        """
        # 1
        flows = self._scene.flows
        tracklets = sorted(self._last_tracklets, key=attrgetter('obj_id'))

    def draw_graph(self):
        img = np.ones(self._frame_shape, dtype=np.uint8) * 128
        diameter = min(*self._flow_blocks_size)
        flows = self._scene.flows
        palette = sns.hls_palette(len(flows))
        # choose colors for each flow
        for y_blk in xrange(self._flow_blocks[0]):
            for x_blk in xrange(self._flow_blocks[1]):
                flow_node = self._blk_node_map[y_blk][x_blk]
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
                r = int(diameter/4)

                cv2.circle(img, flow_node.position, r, color, -1)

        # finally show flows
        for flow in flows:
            lines = []
            for n, node in enumerate(flow.path):
                if node == flow.path[-1]:
                    break
                lines.append(node.position)

            self.logger.debug('drawing {}th flow {} with {} lines'.format(n, flow, len(flow.path)))

            # color
            p_color = palette[flows.index(flow)]
            color = (int(p_color[0]*192), int(p_color[1]*192), int(p_color[2]*192))
            cv2.polylines(img, [np.int32(lines)], False, color, thickness=2)
            # arrow
            color = (int(p_color[0]*255), int(p_color[1]*255), int(p_color[2]*255))
            cv2.arrowedLine(img, flow.path[-2].position , flow.sink.position, color, thickness=2, tipLength=0.5)
            # number
            draw_str(img, flow.path[-2].position, 'F{}'.format(flow.name))

        return img