#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from operator import attrgetter

import networkx as nx
from networkx.algorithms import community as nxcom

from ..tengu_scene_analyzer import KLTSceneAnalyzer, TenguNode
from ..tengu_tracker import TenguTracker, TrackedObject

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

	def __init__(self, rect, **kwargs):
		super(ClusteredKLTTrackedObject, self).__init__(rect, **kwargs)


	def update_tracking(self, rect, *args):
		return super(ClusteredKLTTrackedObject, self).update_tracking(rect, *args)

class NodeCluster(object):

	def __init__(self, group):
		super(NodeCluster, self).__init__()
		self.group = group

class ClusteredKLTTracker(TenguTracker):

	_minimum_community_size = 2
	
	def __init__(self, klt_scene_analyzer, **kwargs):
		super(ClusteredKLTTracker, self).__init__(**kwargs)
		# TODO: check class?
		self._klt_scene_analyzer = klt_scene_analyzer
		self.graph = nx.Graph()

	def calculate_cost_matrix(self, detections):
		
		# update node and edge(add, remove)
		self.update_graph()

		# update weights
		self.update_weights(detections)

		# detect communities
		node_clusters = self.find_node_clusters()

		# create a cost matrix


		return super(ClusteredKLTTracker, self).calculate_cost_matrix(detections)

	def update_graph(self):
		
		nodes = self._klt_scene_analyzer.nodes
		graph_nodes = list(self.graph.nodes())
		self.logger.info('updating graph with {} nodes, and {} graph nodes, {} edges'.format(len(nodes), len(graph_nodes), len(self.graph.edges())))

		# remove
		removed_nodes = self._klt_scene_analyzer.last_removed_nodes
		removed_graph_nodes = []
		for removed_node in removed_nodes:
			if removed_node in graph_nodes:
				removed_edges = self.graph.edges(removed_node)
				self.logger.info('removing {} edges'.format(len(removed_edges)))
				self.graph.remove_edges_from(removed_edges)
				removed_graph_nodes.append(removed_node)
		self.logger.info('removing {} nodes'.format(len(removed_graph_nodes)))
		self.graph.remove_nodes_from(removed_graph_nodes)

	def update_weights(self, detections):
		
		nodes = self._klt_scene_analyzer.nodes
		graph_nodes = list(self.graph.nodes())
		for node in nodes:
			for detection in detections:
				in_nodes = []
				if node.inside_rect(detection):
					if not node in graph_nodes:
						self.graph.add_node(node)
					in_nodes.append(node)
				if len(in_nodes) > 0:
					self.update_mutual_edges_weight(in_nodes)
					#self.logger.info('found {} in-nodes'.format(len(in_nodes)))

	def update_mutual_edges_weight(self, in_nodes):

		updated_edges = []
		for node in in_nodes:
			# make sure all the mutual edges exist
			for another_node in in_nodes:
				if another_node == node:
					continue
				if not self.graph.has_edge(node, another_node):
					# create one
					#self.logger.info('creating an edge from {} to {}'.format(node, another_node))
					edge = (node, another_node, {'weight': 1})
					self.graph.add_edges_from([edge])
					updated_edges.append(edge)
					continue
				edge = self.graph[node][another_node]
				if not edge in updated_edges:
					current_weight = edge['weight']
					edge['weight'] = current_weight + 1
					#self.logger.info('updating an edge, {}, from {} to {}'.format(edge, current_weight, edge['weight']))
					updated_edges.append(edge)
		#self.logger.info('updated {} edges'.format(len(updated_edges)))

	def find_node_clusters(self):

		communities = nxcom.girvan_newman(self.graph)
		community = next(communities)
		node_cluster = NodeCluster(community)
		self.logger.info('adding a node cluster: {} having {} nodes'.format(node_cluster, len(community)))
		node_clusters = []
		for group in community:
			if len(group) > ClusteredKLTTracker._minimum_community_size:
				self.logger.info('found a group of size {}'.format(len(group)))
				node_clusters.append(NodeCluster(group))
		return node_clusters