#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from operator import attrgetter

import networkx as nx
from networkx.algorithms import community

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

class ClusteredKLTTracker(TenguTracker):
	
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

		return super(ClusteredKLTTracker, self).calculate_cost_matrix(detections)

	def update_graph(self):
		
		nodes = self._klt_scene_analyzer.nodes
		graph_nodes = list(self.graph.nodes())
		self.logger.info('updating graph with {} nodes, and {} graph nodes'.format(len(nodes), len(graph_nodes)))

		# create
		new_nodes = self._klt_scene_analyzer.last_added_nodes
		self.logger.info('adding {} new nodes'.format(len(new_nodes)))
		self.graph.add_nodes_from(new_nodes)

		# remove
		removed_nodes = self._klt_scene_analyzer.last_removed_nodes
		self.logger.info('removing {} nodes'.format(len(removed_nodes)))
		self.graph.remove_nodes_from(removed_nodes)

	def update_weights(self, detections):
		pass