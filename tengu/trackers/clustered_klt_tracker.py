#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, math, time
from sets import Set
from operator import attrgetter

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment

from ..tengu_flow_analyzer import KLTAnalyzer, TenguNode
from ..tengu_tracker import TenguTracker, Tracklet, TenguCostMatrix

"""
This tracker represents a list of optical flow trackes as a weighed undirected graph.
Each track is a vertex, and between which an edge is created if any two vertices exist in a same detection.
The number of such co-existence is counted as its weight.

Then, such a graph is somehow clustered to represent an object.
One way is Spectral Clustering by generating Laplacian matrix of the graph.
But it is hard to know the number of clusters beforehand, so this is not ideal.

So, a different approach was taken here. Girvan and Newman's community detection.
The algorithm is available as centrality.girvan_newman in networkx(since 2.0).

Once a graph is clustered into sets of vertices(optical flow tracks), 
a cost matrix is calculated for assignments.
"""

class ClusteredKLTTracklet(Tracklet):

    def __init__(self, tracker, keep_lost_tracklet=False):
        super(ClusteredKLTTracklet, self).__init__()
        self.tracker = tracker
        self._keep_lost_tracklet = keep_lost_tracklet
        self._rect = None
        self._hist = None
        self._centers = []
        self._validated_nodes = Set([])

    @property
    def rect(self):
        """
        an internal rect is updated with or without assignment as follows:
        1. with assignment
            rect <= detection
        2. without assignment
            2-0: update its position by valid ndoes
            2-1: no valid nodes anymore, only one assignment is available
                rect <= NO UPDATE
            2-2: 2 detections are available
                rect <= two previous rects
        """
        return self._rect

    def similarity(self, assignment):
        """
        calculate similarity of assignment to self
        1. rect similarity(position, size similrity)
        2. histogram similarity
        """


        if self._rect is None:
            # this means this is called for the first time
            assignment.hist, assignment.img = self.histogram(assignment.detection)
            return 1.0

        # 1. rect similarity
        rect_similarity = TenguTracker.calculate_overlap_ratio(self._rect, assignment.detection)

        # 2. histogram similarity
        hist0 = self._hist
        hist1, assignment.img = self.histogram(assignment.detection)
        assignment.hist = hist1
        hist_similarity = cv2.compareHist(hist0, hist1, cv2.HISTCMP_CORREL)

        disable_similarity = False
        if disable_similarity:
            rect_similarity = 1.0
            hist_similarity = 1.0

        similarity = [rect_similarity, hist_similarity]
        
        self.logger.debug('similarity = {}'.format(similarity))

        return min(similarity)

    def histogram(self, rect):
        frame = self.tracker._tengu_flow_analyer._klt_analyzer.last_frame
        img = frame[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[3]), :]
        hist = cv2.calcHist([img], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        flattened = hist.flatten()
        return flattened, img

    def last_movement(self):
        if len(self._centers) < TenguNode._min_length:
            # no speed calculation possible
            return None

        pos_last = self._centers[-1]
        pos_first = self._centers[0]

        return [(pos_last[0]-pos_first[0])/TenguNode._min_length, (pos_last[1]-pos_first[1])/TenguNode._min_length]

    def update_properties(self, lost=True):
        assignment = self._assignments[-1]
        self._movement = self.last_movement()
        new_confidence = self.similarity(assignment)
        if new_confidence < self._confidence:
            # instead of directly replacing the value, use decay
            self._confidence = self._confidence * Tracklet._estimation_decay
        else:
            self._confidence = new_confidence
        # rect has to be updated after similarity calculation
        self._rect = assignment.detection
        self._hist = assignment.hist
        self._centers.append(NodeCluster.center(self._rect))
        if len(self._centers) > TenguNode._min_length:
            del self._centers[0]
        if not lost or self._keep_lost_tracklet:
            self._last_updated_at = TenguTracker._global_updates

    def update_with_assignment(self, assignment, class_name):
        if len(self._assignments) > 0:
            self.logger.debug('{}@{}: updating with {} from {} at {}'.format(id(self), self.obj_id, assignment, self._assignments[-1], self._last_updated_at))

        # check ownership
        owned = []
        for node in assignment.group:
            if node.owner is not None:
                if node.owner != self:
                    continue
                else:
                    # already owned
                    owned.append(node)
            else:
                node.set_owner(self)
                owned.append(node)
        assignment.group = owned

        if len(self._assignments) > 0 and self.similarity(assignment) < Tracklet._min_confidence:
            # do not accept this
            self.update_without_assignment()
        else:
            # update
            self._assignments.append(assignment)
            self._validated_nodes = Set(assignment.group)
            self.update_properties(lost=False)
            self.recent_updates_by('1')
            if not self._class_map.has_key(class_name):
                self._class_map[class_name] = 0
            self._class_map[class_name] += 1

    def update_without_assignment(self):
        """
        no update was available
        so create a possible assignment to update
        """

        self.logger.debug('updating without {}@{}'.format(id(self), self.obj_id))

        if len(self._validated_nodes) > 0:
            empty = NodeCluster(self._validated_nodes, None)
            next_rect = empty.estinamte_next_rect(self._rect)
            if next_rect is None:
                self.recent_updates_by('2-0x')
            else:
                empty.detection = next_rect
                self._assignments.append(empty)
                self.update_properties()
                self.recent_updates_by('2-0')
        else:
            # pattern 2
            self.recent_updates_by('2-1')
        
        self.validate_nodes()

    def validate_nodes(self):
        """
        check and merge valid nodes
        """
        latest_set = Set(self._assignments[-1].group)
        if self._validated_nodes is None:
            self._validated_nodes = latest_set
            return

        # check
        validated = Set([])
        for node in self._validated_nodes:
            # in the first place, remove if outdated
            if node.last_updated_at != TenguTracker._global_updates-1:
                continue 
            if node in validated:
                continue
            # then, find at least one similar node
            for another in self._validated_nodes:
                if node == another:
                    continue
                if another.last_updated_at != TenguTracker._global_updates-1:
                    continue 
                similarity = node.similarity(another)
                if min(similarity) >=  ClusteredKLTTracker._minimum_node_similarity:
                    # found one
                    validated = validated | Set([node, another])
                    break
        self._validated_nodes = validated

class NodeCluster(object):

    _min_rect_length = 10

    def __init__(self, group, detection):
        super(NodeCluster, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.group = group
        self.detection = detection
        self.hist = None

    def __repr__(self):
        return '{} nodes of {}, detection={}'.format(len(self.group), self.group, self.detection)

    def rect_from_group(self):
        # larget enough value
        left = 10000
        top = 10000
        # small enough value
        right = 0
        bottom = 0
        for node in self.group:
            x, y = node.tr[-1]
            if x < left:
                left = x
            if x > right:
                right = x

            if y < top:
                top = y
            if y > bottom:
                bottom = y
        return (int(left), int(top), int(right-left), int(bottom-top))

    def avg_movement(self):
        total_move_x = 0
        total_move_y = 0
        valid = 0
        for node in self.group:
            movement = node.last_move()
            if movement is None:
                continue

            total_move_x += movement[0]
            total_move_y += movement[1]
            valid += 1

        if valid == 0:
            return None

        return [total_move_x/valid, total_move_y/valid]

    def estinamte_next_rect(self, current_rect):
        """ estimate the next rect from the movement of pixels
        """
        avg_movement = self.avg_movement()
        if avg_movement is None or avg_movement[0] < 1 or avg_movement[1] < 1:
            # this is stationally
            self.logger.debug('stationally cluster, estimate the same rect as prev')
            return current_rect
        else:
            self.logger.debug('avg_move is {}, not stationally, estimating next rect...'.format(avg_movement))

        total_moves = []
        for node in self.group:
            if len(node.tr) == 1:
                continue
            prev = node.tr[-1]
            prev2 = node.tr[-2]

            move = [prev[0]-prev2[0], prev[1]-prev2[1]]
            total_moves.append(move)

        if len(total_moves) == 0:
            return None

        total_moves_sum = [0, 0]
        for move in total_moves:
            total_moves_sum[0] += move[0]
            total_moves_sum[1] += move[1]

        next_rect = (current_rect[0] + int(total_moves_sum[0]/len(total_moves_sum)), current_rect[1] + int(total_moves_sum[1]/len(total_moves_sum)), current_rect[2], current_rect[3])
        self.logger.debug('next rect {} was estimated from {}'.format(next_rect, current_rect))
        return next_rect

    @staticmethod
    def center(rect):
        return (rect[0]+int(rect[2]/2), rect[1]+int(rect[3]/2))

class ClusteredKLTTracker(TenguTracker):

    _minimum_community_size = 2
    _minimum_node_similarity = 0.5
    
    def __init__(self, keep_lost_tracklet=False, **kwargs):
        super(ClusteredKLTTracker, self).__init__(**kwargs)
        self._keep_lost_tracklet = keep_lost_tracklet
        self._tengu_flow_analyer = None
        self.detected_node_set = Set([])
        self.detection_node_map = None
        self.debug = None

    def prepare_updates(self, detections):

        if self._tengu_flow_analyer._klt_analyzer.draw_flows:
            # this is a debug mode
            self.debug = self._tengu_flow_analyer._klt_analyzer.prev_gray.copy()

        start = time.time()

        # update node and edge(add, remove)
        self.update_detected_node_set()
        lap1 = time.time()
        self.logger.debug('udpate_detected_node_set took {} s'.format(lap1 - start))

        # update weights
        self.detection_node_map = self.update_detection_node_map(detections)
        lap2 = time.time()
        self.logger.debug('update_detection_node_map took {} s'.format(lap2 - lap1))

        end = time.time()
        self.logger.debug('prepare_updates took {} s'.format(end - start))

        # update
        self._tengu_flow_analyer._klt_analyzer.last_detections = detections

        if self._tengu_flow_analyer._klt_analyzer.draw_flows:
            self.draw_detected_node_set()
            cv2.imshow('Clustered KLT Debug', self.debug)

    def initialize_tracklets(self, detections, class_names):
        
        self.prepare_updates(detections)

        for d, detection in enumerate(detections):
            self._tracklets.append(self.new_tracklet(detection, class_names[d]))

    def new_tracklet(self, assignment, class_name):
        to = ClusteredKLTTracklet(self, keep_lost_tracklet=self._keep_lost_tracklet)
        to.update_with_assignment(NodeCluster(self.detection_node_map[assignment], assignment), class_name)
        return to

    def update_detected_node_set(self):
        
        nodes = self._tengu_flow_analyer._klt_analyzer.nodes
        self.logger.debug('updating detected_node_set with {} nodes, and {} detected_node_set nodes'.format(len(nodes), len(self.detected_node_set)))

        # remove
        removed_nodes = self._tengu_flow_analyer._klt_analyzer.last_removed_nodes
        for removed_node in removed_nodes:
            self.detected_node_set.discard(removed_node)

    def update_detection_node_map(self, detections):
        
        detection_node_map = {}
        for detection in detections:
            detection_node_map[detection] = []
        nodes = self._tengu_flow_analyer._klt_analyzer.nodes
        for node in nodes:
            for detection in detections:
                if node.inside_rect(detection):
                    if not node in self.detected_node_set:
                        self.detected_node_set.add(node)
                    else:
                        node.update_last_detected(detection)
                    detection_node_map[detection].append(node)

        for detection in detection_node_map:
            in_nodes = detection_node_map[detection]
            self.logger.debug('found {} in-nodes in {}'.format(len(in_nodes), detection))

        return detection_node_map

    def assign_new_to_tracklet(self, new_assignment, class_name, tracklet):
        if new_assignment is None:
            tracklet.update_without_assignment()
        else:
            tracklet.update_with_assignment(NodeCluster(self.detection_node_map[new_assignment], new_assignment), class_name)

    def obsolete_trackings(self):

        # obsolete trackings
        super(ClusteredKLTTracker, self).obsolete_trackings()

        # obsolete detected nodes
        obsolete_nodes = Set([])
        for node in self.detected_node_set:
            diff = TenguTracker._global_updates - node.last_detected_at
            if diff > self._obsoletion:
                # node is obsolete
                obsolete_nodes.add(node)
                # REMARKS: THIS IS UGLY, ISOLATE ASAP
                if node in self._tengu_flow_analyer._klt_analyzer._last_removed_nodes:
                    # this node was already removed, nothing to do
                    pass
                else:
                    # remove from sceneanalyzer as well
                    self._tengu_flow_analyer._klt_analyzer._last_removed_nodes.append(node)
                    del self._tengu_flow_analyer._klt_analyzer.nodes[self._tengu_flow_analyer._klt_analyzer.nodes.index(node)]
                self.logger.debug('removed obsolete node due to diff = {}'.format(diff))

        # REMARKS: THIS IS UGLY, TOO
        for node in self._tengu_flow_analyer._klt_analyzer.nodes:
            diff = TenguTracker._global_updates - node.last_detected_at
            if diff > self._obsoletion:
                # remove from sceneanalyzer as well
                self._tengu_flow_analyer._klt_analyzer._last_removed_nodes.append(node)
                del self._tengu_flow_analyer._klt_analyzer.nodes[self._tengu_flow_analyer._klt_analyzer.nodes.index(node)]

        for tracklet in self._tracklets:
            # obsolete if one of its nodes has left
            for node in tracklet._validated_nodes:
                if node.has_left:
                    self.logger.debug('removing {}, node {} has left'.format(tracklet, node))
                    tracklet.mark_left()
                    break

        self.detected_node_set = self.detected_node_set - obsolete_nodes

    def draw_detected_node_set(self):
        for node in self._tengu_flow_analyer._klt_analyzer.nodes:
            cv2.circle(self.debug, node.tr[-1], 5, 128, -1)


        for tracklet in self._tracklets:
            node_cluster = tracklet.last_assignment
            rect = node_cluster.rect_from_group()
            cv2.rectangle(self.debug, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, 3)
            for node in tracklet._validated_nodes:
                cv2.circle(self.debug, node.tr[-1], 5, 256, -1)