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
from ..tengu_tracker import TenguTracker, TrackedObject, TenguCostMatrix

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

class ClusteredKLTTrackedObject(TrackedObject):

    def __init__(self):
        super(ClusteredKLTTrackedObject, self).__init__()

    @property
    def rect(self):
        self.logger.debug('retruning rect from {}'.format(self))
        if isinstance(self.last_assignment, NodeCluster):

            if self.last_assignment.detection is None:
                return self.last_assignment.rect_from_group()

            return self._assignments[-1].detection

        # otherwise, this is just a rect
        return self._assignments[-1]


class NodeCluster(object):

    _min_rect_length = 10

    def __init__(self, group):
        super(NodeCluster, self).__init__()
        self.group = group
        self.detection = None

    def __repr__(self):
        return '{} nodes, detection={}'.format(len(self.group), self.detection)

    def similarity(self, another_node_cluster):
        similarity = 0.
        for node in self.group:
            if node in another_node_cluster.group:
                similarity += 1
        if similarity > 0:
            similarity = similarity / len(self.group)
        return similarity

    def rect_from_group(self):
        sum_x = 0
        sum_y = 0
        for node in self.group:
            x, y = node.tr[-1]
            sum_x += x
            sum_y += y
        avg_x = max(NodeCluster._min_rect_length, int(sum_x / len(self.group)))
        avg_y = max(NodeCluster._min_rect_length, int(sum_y / len(self.group)))
        # x, y, w, h
        offset = int(NodeCluster._min_rect_length/2)
        rect = [avg_x - offset, avg_y - offset, NodeCluster._min_rect_length, NodeCluster._min_rect_length]
        return rect

class ClusteredKLTTracker(TenguTracker):

    _minimum_community_size = 2
    _minimum_node_similarity = 0.5
    
    def __init__(self, klt_scene_analyzer, **kwargs):
        super(ClusteredKLTTracker, self).__init__(**kwargs)
        # TODO: check class?
        self._klt_scene_analyzer = klt_scene_analyzer
        self.graph = nx.Graph()
        self.current_node_clusters = None
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
        self.update_weights(detections)
        lap2 = time.time()
        self.logger.debug('update_weights took {} s'.format(lap2 - lap1))

        # detect communities
        self.current_node_clusters = self.find_node_clusters()
        lap3 = time.time()
        self.logger.debug('find_node_clusters took {} s'.format(lap3 - lap2))

        # assign detections to node_clusters
        self.assign_detections_to_node_clusters(detections, self.current_node_clusters)
        lap4 = time.time()
        self.logger.debug('assign_detections_to_node_clusters took {} s'.format(lap4 - lap3))

        end = time.time()
        self.logger.debug('prepare_updates took {} s'.format(end - start))

        # update
        self._klt_scene_analyzer.last_detections = detections

        if self._klt_scene_analyzer.draw_flows:
            self.draw_graph()
            cv2.imshow('Clustered KLT Debug', self.debug)

    def initialize_tracked_objects(self, detections):
        
        self.prepare_updates(detections)

        for node_cluster in self.current_node_clusters:
            to = self.new_tracked_object(node_cluster)
            self._tracked_objects.append(to)

        self.logger.debug('initialized {} klt tracked objects'.format(len(self.current_node_clusters)))

    def new_tracked_object(self, assignment):
        to = ClusteredKLTTrackedObject()
        self.logger.debug('created klt tracked object {} of id {}'.format(to, to.obj_id))
        to.update_with_assignment(assignment)
        return to

    def calculate_cost_matrix(self, detections):        

        # create a cost matrix
        cost_matrix = self.create_empty_cost_matrix(len(self.current_node_clusters))
        for t, tracked_object in enumerate(self._tracked_objects):
            for c, node_cluster in enumerate(self.current_node_clusters):
                cost = self.calculate_cost(tracked_object, node_cluster)
                cost_matrix[t][c] = cost

        tengu_cost_matrix = TenguCostMatrix(self.current_node_clusters, cost_matrix)
        overlap_cost_matrix = super(ClusteredKLTTracker, self).calculate_cost_matrix(detections)[0]

        return [tengu_cost_matrix, overlap_cost_matrix]

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
        
        in_nodes_dict = {}
        for detection in detections:
            in_nodes_dict[detection] = []
        nodes = self._klt_scene_analyzer.nodes
        graph_nodes = list(self.graph.nodes())
        for node in nodes:
            for detection in detections:
                if node.inside_rect(detection):
                    if not node in graph_nodes:
                        self.graph.add_node(node)
                    else:
                        node.update_last_detected()
                    in_nodes_dict[detection].append(node)

        for detection in in_nodes_dict:
            in_nodes = in_nodes_dict[detection]
            self.update_mutual_edges_weight(in_nodes)
            self.logger.debug('found {} in-nodes in {}'.format(len(in_nodes), detection))

    def update_mutual_edges_weight(self, in_nodes):
        """
        check if a node is similar enough to another.
        if similar, create or update their edge, otherwise, ignore.
        """
        updated_edges = []
        last_node = None
        for node in in_nodes:
            if last_node is None:
                last_node = in_nodes[-1]
            similarity = node.similarity(last_node)
            if similarity < ClusteredKLTTracker._minimum_node_similarity:
                self.logger.debug('skipping making edge, {} is not similar enough to {}, the similarity = {}'.format(node, last_node, similarity))
                continue
            # make an edge
            if not self.graph.has_edge(node, last_node):
                # create one
                self.logger.debug('creating an edge from {} to {}'.format(node, last_node))
                edge = (node, last_node, {'weight': 1})
                self.graph.add_edges_from([edge])
                updated_edges.append(edge)
            edge = self.graph[node][last_node]
            if not edge in updated_edges:
                current_weight = edge['weight']
                edge['weight'] = current_weight + 1
                self.logger.debug('updating an edge, {}, from {} to {}'.format(edge, current_weight, edge['weight']))
                updated_edges.append(edge)
        self.logger.debug('updated {} edges'.format(len(updated_edges)))

    def find_node_clusters(self):

        #communities = nxcom.girvan_newman(self.graph)
        #community = next(communities)
        # TODO: this simply finds connected sets of nodes, so find a way to disconnect
        community = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        node_clusters = []
        for group in community:
            if len(group) > ClusteredKLTTracker._minimum_community_size:
                self.find_and_cut_obsolete_edges(group)
                self.logger.debug('found a group of size {}'.format(len(group)))
                node_clusters.append(NodeCluster(group))
        self.logger.debug('community groups: {}, large enough groups: {}'.format(len(community), len(node_clusters)))

        return node_clusters

    def find_and_cut_obsolete_edges(self, group):

        for node in group:
            edges = self.graph.edges(node)
            for edge in edges:
                another = edge[1]
                similarity = node.similarity(another)
                # TODO consider better ways to find out this threshold
                if similarity <  ClusteredKLTTracker._minimum_node_similarity:
                    self.graph.remove_edge(node, another)
                    self.logger.debug('the similarity({}) is too low, removed and edge from {} to {}'.format(similarity, node, another))
                else:
                    cv2.line(self.debug, node.tr[-1], another.tr[-1], 192, min(10, self.graph[node][another]['weight']))

    def assign_detections_to_node_clusters(self, detections, node_clusters):
        """
        create a cost matrix Cmn, m is the number of node_clusters, n is of detections
        then, solve by hungarian algorithm
        """
        # create cost matrix
        shape = (len(node_clusters), len(detections))
        cost_matrix = np.zeros(shape, dtype=np.float32)
        if len(shape) == 1:
            # the dimesion should be forced
            cost_matrix = np.expand_dims(cost_matrix, axis=1)
        for c, node_cluster in enumerate(node_clusters):
            for d, detection in enumerate(detections):
                covered = 0
                for graph_node in node_cluster.group:
                    if graph_node.inside_rect(detection):
                        covered += 1
                coverage = 0.
                if covered > 0:
                    coverage = covered / len(node_cluster.group)
                cost_matrix[c][d] = -1 * math.log(max(coverage, TenguTracker._min_value))
        # solve
        tengu_cost_matrix = TenguCostMatrix(detections, cost_matrix)
        TenguTracker.optimize_and_assign([tengu_cost_matrix])
        for ix, row in enumerate(tengu_cost_matrix.ind[0]):
            node_clusters[row].detection = detections[tengu_cost_matrix.ind[1][ix]]

        # NOTE that some node clusters may not be assigned detections

    def calculate_cost(self, tracked_object, node_cluster):
        if isinstance(tracked_object.last_assignment, NodeCluster):
            similarity = tracked_object.last_assignment.similarity(node_cluster)
            return -1 * math.log(max(similarity, TenguTracker._min_value))

        return super(ClusteredKLTTracker, self).calculate_cost_by_overlap_ratio(tracked_object.last_assignment, node_cluster.rect_from_group())

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

    def draw_graph(self):
        graph_nodes = list(self.graph.nodes())
        for graph_node in graph_nodes:
             cv2.circle(self.debug, graph_node.tr[-1], 5, 255, -1)
             edges = list(self.graph.edges(graph_node))
             for edge in edges:
                cv2.line(self.debug, edge[0].tr[-1], edge[1].tr[-1], 192, min(10, self.graph[edge[0]][edge[1]]['weight']))