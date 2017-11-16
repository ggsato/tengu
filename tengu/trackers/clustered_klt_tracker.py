#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, math, time
from operator import attrgetter

import cv2
import numpy as np

import networkx as nx
from networkx.algorithms import community as nxcom
from scipy.optimize import linear_sum_assignment

from ..tengu_scene_analyzer import KLTSceneAnalyzer, TenguNode
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

    def __init__(self, tracker):
        super(ClusteredKLTTracklet, self).__init__()
        self.tracker = tracker
        self._rect = None
        self._validated_nodes = set()

    @property
    def rect(self):
        """
        an internal rect is updated with or without assignment as follows:
        1. with assignment
            rect <= detection
        2. without assignment
            2-0: find valid nodes, and update its position
            2-1: the last update was done without detection either by 3 or 2
                rect <= avg_movement or NO UPDATE
            2-2: the last update was done with detecion
                2-2-1: only one detection is available
                    rect <= NO UPDATE
                2-2-2: the second last node cluster is available
                    rect <= avg_movement or NO UPDATE
                2-2-3: 2 detections are available
                    rect <= two previous rects
        """
        return self._rect

    def update_with_assignment(self, assignment):
        if len(self._assignments) > 0:
            self.logger.debug('{}@{}: updating with {} from {} at {}'.format(id(self), self.obj_id, assignment, self._assignments[-1], self._last_updated_at))
        
        # update
        self._assignments.append(assignment)
        self._rect = assignment.detection
        self.recent_updates_by('1')

        if self.validate_nodes():
            self._last_updated_at = TenguTracker._global_updates

    def update_without_assignment(self):
        if Tracklet._disable_estimation:
            return

        self.logger.debug('updating without {}@{}'.format(id(self), self.obj_id))

        valid_group = []
        if len(self._assignments) > 0 and self._assignments[-1].group is not None:
            for node in self._assignments[-1].group:
                if node.last_updated_at == TenguTracker._global_updates-1:
                    valid_group.append(node)

        # pattern 2
        if len(valid_group) > 0:
            empty = NodeCluster(valid_group, None)
            self._assignments.append(empty)
            next_rect = empty.estinamte_next_rect(self._rect)
            if next_rect is None:
                self.recent_updates_by('2-0x')
            else:
                self._rect = next_rect
                self.recent_updates_by('2-0')
            #self._last_updated_at = TenguTracker._global_updates
            return
        else:
            if self._assignments[-1].detection is None:
                # pattern 2-1
                # estimate from node cluster
                last_move_x, last_move_y = self._assignments[-1].avg_movement()
                if last_move_x is None:
                    # can't estimate
                    self.recent_updates_by('2-1x')
                    return
                self.recent_updates_by('2-1')
            else:
                # pattern 2-2
                if len(self._assignments) == 1:
                    # 2-2-1
                    # only one detection is available, can't estimate
                    self.recent_updates_by('2-2-1')
                    return
                elif self._assignments[-2].detection is None:
                    # 2-2-2
                    last_move_x, last_move_y = self._assignments[-2].avg_movement()
                    if last_move_x is None:
                        # can't estimate
                        self.recent_updates_by('2-2-2x')
                        return
                    self.recent_updates_by('2-2-2')
                else:
                    # 2-2-3
                    prev = self._assignments[-1].detection
                    prev2 = self._assignments[-2].detection
                    last_move_x, last_move_y = Tracklet.movement_from_rects(prev, prev2)
                    self.recent_updates_by('2-2-3')

        new_x = self._rect[0] + last_move_x * Tracklet._estimation_decay
        new_y = self._rect[1] + last_move_y * Tracklet._estimation_decay
        self._rect = (new_x, new_y, self._rect[2], self._rect[3])
        
        if self.validate_nodes():
            self._last_updated_at = TenguTracker._global_updates

    def validate_nodes(self):
        """
        check and merge valid nodes
        """
        latest_set = set(self._assignments[-1].group)
        if len(self._validated_nodes) == 0:
            self._validated_nodes = latest_set
            return True

        # merge
        self._validated_nodes = self._validated_nodes | latest_set

        # check
        validated = set()
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
                similarity = node.similarity(another)
                if min(similarity) >=  ClusteredKLTTracker._minimum_node_similarity:
                    # found one
                    validated = validated | set([node, another])
                    break
        self._validated_nodes = validated

        return len(self._validated_nodes) > 0

class NodeCluster(object):

    _min_rect_length = 10

    def __init__(self, group, detection):
        super(NodeCluster, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.group = group
        self.detection = detection

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

    def avg_movement(self, min_length=25):
        total_move_x = 0
        total_move_y = 0
        valid = 0
        longest = 0
        avg_node = None
        for node in self.group:
            if len(node.tr) < min_length:
                continue

            if len(node.tr) > longest:
                longest = len(node.tr)
                avg_node = node
        if avg_node is None:
            return None, None

        return avg_node.last_move(min_length)

    def estinamte_next_rect(self, current_rect):
        origin_estimates = []
        for node in self.group:
            if len(node.tr) == 1:
                continue
            prev = node.tr[-1]
            prev2 = node.tr[-2]

            origin_estimate = [prev[0]-int(current_rect[2]/2), prev[1]-int(current_rect[3]/2)]
            origin_estimates.append(origin_estimate)

        if len(origin_estimates) == 0:
            return None

        total_estimates = [0, 0]
        for origin_estimate in origin_estimates:
            total_estimates[0] += origin_estimate[0]
            total_estimates[1] += origin_estimate[1]

        next_rect = (int(total_estimates[0]/len(origin_estimates)), int(total_estimates[1]/len(origin_estimates)), current_rect[2], current_rect[3])
        self.logger.debug('next rect {} was estimated from {}'.format(next_rect, current_rect))
        return next_rect

class ClusteredKLTTracker(TenguTracker):

    _minimum_community_size = 2
    _minimum_node_similarity = 0.5
    
    def __init__(self, klt_scene_analyzer, **kwargs):
        super(ClusteredKLTTracker, self).__init__(**kwargs)
        # TODO: check class?
        self._klt_scene_analyzer = klt_scene_analyzer
        self.graph = nx.Graph()
        self.detection_node_map = None
        self.debug = None

    def prepare_updates(self, detections):

        if self._klt_scene_analyzer.draw_flows:
            # this is a debug mode
            self.debug = self._klt_scene_analyzer.prev_gray.copy()

        start = time.time()

        # update node and edge(add, remove)
        self.update_graph()
        lap1 = time.time()
        self.logger.debug('udpate_graph took {} s'.format(lap1 - start))

        # update weights
        self.detection_node_map = self.update_weights(detections)
        lap2 = time.time()
        self.logger.debug('update_weights took {} s'.format(lap2 - lap1))

        end = time.time()
        self.logger.debug('prepare_updates took {} s'.format(end - start))

        # update
        self._klt_scene_analyzer.last_detections = detections

        if self._klt_scene_analyzer.draw_flows:
            self.draw_graph()
            cv2.imshow('Clustered KLT Debug', self.debug)

    def initialize_tracklets(self, detections):
        
        self.prepare_updates(detections)

        for detection in detections:
            self._tracklets.append(self.new_tracklet(detection))

    def new_tracklet(self, assignment):
        to = ClusteredKLTTracklet(self)
        to.update_with_assignment(NodeCluster(self.detection_node_map[assignment], assignment))
        return to

    def update_graph(self):
        
        nodes = self._klt_scene_analyzer.nodes
        graph_nodes = list(self.graph.nodes())
        self.logger.debug('updating graph with {} nodes, and {} graph nodes, {} edges'.format(len(nodes), len(graph_nodes), len(self.graph.edges())))

        # remove
        removed_nodes = self._klt_scene_analyzer.last_removed_nodes
        removed_graph_nodes = []
        for removed_node in removed_nodes:
            if removed_node in graph_nodes:
                removed_edges = self.graph.edges(removed_node)
                self.logger.debug('removing {} edges'.format(len(removed_edges)))
                self.graph.remove_edges_from(removed_edges)
                removed_graph_nodes.append(removed_node)
        self.logger.debug('removing {} nodes'.format(len(removed_graph_nodes)))
        self.graph.remove_nodes_from(removed_graph_nodes)

    def update_weights(self, detections):
        
        detection_node_map = {}
        for detection in detections:
            detection_node_map[detection] = []
        nodes = self._klt_scene_analyzer.nodes
        graph_nodes = list(self.graph.nodes())
        for node in nodes:
            for detection in detections:
                if node.inside_rect(detection):
                    if not node in graph_nodes:
                        self.graph.add_node(node)
                    else:
                        node.update_last_detected(detection)
                    detection_node_map[detection].append(node)

        for detection in detection_node_map:
            in_nodes = detection_node_map[detection]
            self.update_mutual_edges_weight(in_nodes)
            self.logger.debug('found {} in-nodes in {}'.format(len(in_nodes), detection))

        return detection_node_map

    def update_mutual_edges_weight(self, in_nodes):
        """
        check if a node is similar enough to another.
        if similar, create or update their edge, otherwise, ignore.
        """
        last_node = None
        for node in in_nodes:
            if last_node is None:
                last_node = in_nodes[-1]
            similarity = node.similarity(last_node)
            if min(similarity) < ClusteredKLTTracker._minimum_node_similarity:
                self.logger.debug('skipping making edge, {} is not similar enough to {}, the similarity = {}'.format(node, last_node, similarity))
                last_node = node
                continue
            # choose better edge
            # here, choose only one
            make_one = True
            existing_edges = self.graph.edges(node)
            for existing_edge in existing_edges:
                another_node = existing_edge[1]
                if node == another_node:
                    another_node = existing_edge[0]
                another_similarity = node.similarity(another_node)
                if another_similarity >= similarity:
                    make_one = False
            if not make_one:
                continue
            # make an edge
            if not self.graph.has_edge(node, last_node):
                # create one
                self.logger.debug('creating an edge from {} to {}, similarity={}'.format(node, last_node, similarity))
                edge = (node, last_node, {'weight': ClusteredKLTTracker.weight_from_similarity(similarity), 'last_update': TenguTracker._global_updates})
                self.graph.add_edges_from([edge])
                # # check
                # if not self.graph.has_edge(last_node, node):
                #     self.logger.error('edge is directed!')
                # last_edges = self.graph.edges(last_node)
                # found = False
                # for last_edge in last_edges:
                #     if last_edge[0] == node or last_edge[1] == node:
                #         found = True
                #         break
                # if not found:
                #     self.logger.error('edge is not found in edges!')
            edge = self.graph[node][last_node]
            if edge['last_update'] != TenguTracker._global_updates:
                edge['weight'] = ClusteredKLTTracker.weight_from_similarity(similarity)
                edge['last_update'] = TenguTracker._global_updates
                self.logger.debug('updating an edge, {}, from {} to {}'.format(edge, current_weight, edge['weight']))
            last_node = node

    @staticmethod
    def weight_from_similarity(similarity):
        return int(min(similarity) * 10)

    def assign_new_to_tracklet(self, new_assignment, tracklet):
        if new_assignment is None:
            tracklet.update_without_assignment()
        else:
            tracklet.update_with_assignment(NodeCluster(self.detection_node_map[new_assignment], new_assignment))

    def obsolete_trackings(self):

        # obsolete trackings
        super(ClusteredKLTTracker, self).obsolete_trackings()

        # obsolete nodes
        graph_nodes = list(self.graph.nodes())
        for node in graph_nodes:
            diff = TenguTracker._global_updates - node.last_detected_at
            if diff > self._obsoletion:
                # node is obsolete
                removed_edges = self.graph.edges(node)
                self.graph.remove_edges_from(removed_edges)
                self.graph.remove_node(node)
                # remove from sceneanalyzer as well
                del self._klt_scene_analyzer.nodes[self._klt_scene_analyzer.nodes.index(node)]
                self.logger.debug('removed obsolete node and its edges due to diff = {}'.format(diff))
            else:
                self.cleanup_edge(node)

    def cleanup_edge(self, node):

        edges = self.graph.edges(node)
        for edge in edges:
            another = edge[1]
            if node == another:
                continue
            similarity = node.similarity(another)
            # TODO consider better ways to find out this threshold
            if min(similarity) <  ClusteredKLTTracker._minimum_node_similarity:
                self.graph.remove_edge(node, another)
                self.logger.debug('the similarity({}) is too low, removed and edge from {} to {}'.format(similarity, node, another))

    def draw_graph(self):
        graph_nodes = list(self.graph.nodes())
        for graph_node in graph_nodes:
            cv2.circle(self.debug, graph_node.tr[-1], 5, 128, -1)

        for tracklet in self._tracklets:
            node_cluster = tracklet.last_assignment
            rect = node_cluster.rect_from_group()
            cv2.rectangle(self.debug, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, 3)
            for node in node_cluster.group:
                cv2.circle(self.debug, node.tr[-1], 5, 256, -1)
                if self.graph.has_node(node):
                    edges = list(self.graph.edges(node))
                    for edge in edges:
                        cv2.line(self.debug, edge[0].tr[-1], edge[1].tr[-1], 256, min(10, self.graph[edge[0]][edge[1]]['weight']))