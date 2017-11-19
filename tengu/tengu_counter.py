#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from sets import Set
import networkx as nx

class TenguCounter(object):

    def __init__(self, reporter=None):
        self.logger = logging.getLogger(__name__)
        self._reporter = reporter
    
    def count(self, tracklets):
        pass

class TenguObsoleteCounter(TenguCounter):

    def __init__(self, **kwargs):
        super(TenguObsoleteCounter, self).__init__(**kwargs)
        self._last_tracklets = Set([])

    def count(self, tracklets):
        current_tracklets = Set(tracklets)

        if len(self._last_tracklets) == 0:
            self._last_tracklets = current_tracklets
            return 0

        obsolete_count = self._last_tracklets - current_tracklets
        
        self._last_tracklets = current_tracklets
        return obsolete_count

"""
TenguFlowCounter collects a set of flows, identify similar types of flows,
then assign each tracklet to one of such types of flows.
A flow is characterized by its source and sink.

TenguFlowCounter holds a directed weighed graph, 
and which has nodes in the shape of flow_blocks.
Each flow_block is assigned its dedicated region of frame_shape respectively.

For example, given a set of flow_blocks F = {fb0, fb1, ..., fbn},
fbn =
"""
class TenguFlow(object):

    def __init__(self, source):
        super(TenguFlow, self).__init__()
        self._source = source
        self._sink = sink

class TenguFlowNode(object):

    def __init__(self, y_blk, x_blk):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        self._source_count = 0
        self._sink_count = 0

    def __repr__(self):
        return 'y_blk={}, x_blk={}, source_count={}, sink_count={}'.format(self._y_blk, self._x_blk, self._source_count, self._sink_count)

    @property
    def source_count(self):
        return self._source_count

    def mark_source(self):
        self._source_count += 1

    @property
    def sink_count(self):
        return self._sink_count

    def mark_sink(self):
        self._sink_count += 1

class TenguFlowCounter(TenguObsoleteCounter):

    def __init__(self, frame_shape=(480, 640), flow_blocks=(20, 20), **kwargs):
        super(TenguFlowCounter, self).__init__(**kwargs)
        self._frame_shape = frame_shape
        self._flow_blocks = flow_blocks
        self._flow_graph = nx.DiGraph()
        self._blk_node_map = {}

    def count(self, tracklets):
        """
        """
        if tracklets is None:
            # this means finish report
            if self._reporter is not None:
                self._reporter.report()
            return

        if len(self._flow_graph) == 0:
            self.initialize_flow_graph()

        current_tracklets = Set(tracklets)

        # new tracklets
        new_tracklets = current_tracklets - self._last_tracklets
        self.add_new_tracklets(new_tracklets)

        # existing tracklets
        existing_tracklets = current_tracklets & self._last_tracklets
        self.update_existing_tracklets(existing_tracklets)

        # removed tracklets
        removed_tracklets = self._last_tracklets - current_tracklets
        self.finish_removed_tracklets(removed_tracklets)

        # update reporter
        counts = super(TenguFlowCounter, self).count(tracklets)
        self._reporter.update_counts(counts)

        return counts

    def initialize_flow_graph(self):

        for y_blk in xrange(self.get_y_blk(self._frame_shape[0])):
            self._blk_node_map[y_blk] = {}
            for x_blk in xrange(self.get_x_blk(self._frame_shape[1])):
                flow_node = TenguFlowNode(y_blk, x_blk)
                self._flow_graph.add_node(flow_node)
                self._blk_node_map[y_blk][x_blk] = flow_node

    def flow_node_at(self, x, y):
        y_blk = self.get_y_blk(y)
        x_blk = self.get_x_blk(x)
        return self._blk_node_map[y_blk][x_blk]

    def get_x_blk(self, x):
        x = min(max(0., x), self._frame_shape[1])
        return int(x / self._flow_blocks[1])

    def get_y_blk(self, y):
        y = min(max(0., y), self._frame_shape[0])
        return int(y / self._flow_blocks[0])

    def add_new_tracklets(self, new_tracklets):
        
        for new_tracklet in new_tracklets:
            flow_node = self.flow_node_at(*new_tracklet.center)
            flow_node.mark_source()
            new_tracklet.flows.append(flow_node)
            self.logger.info('source at {}'.format(flow_node))

    def update_existing_tracklets(self, existing_tracklets):
        
        for existing_tracklet in existing_tracklets:
            prev_flow_node = existing_tracklet.flows[-1]
            flow_node = self.flow_node_at(*existing_tracklet.center)
            if prev_flow_node == flow_node:
                continue
            # update edge
            if not self._flow_graph.has_edge(prev_flow_node, flow_node):
                self._flow_graph.add_edge(prev_flow_node, flow_node, weight=0)
            edge = self._flow_graph[prev_flow_node][flow_node]
            edge['weight'] = edge['weight'] + 1
            # add
            existing_tracklet.flows.append(flow_node)
            self.logger.info('updating weight at {}'.format(edge))


    def finish_removed_tracklets(self, removed_tracklets):
        
        for removed_tracklet in removed_tracklets:
            flow_node = self.flow_node_at(*removed_tracklet.center)
            flow_node.mark_sink()
            self.logger.info('sink at {}'.format(flow_node))