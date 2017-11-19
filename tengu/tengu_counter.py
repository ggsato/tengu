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

    def __init__(self, x_blk, y_blk):
        super(TenguFlowNode, self).__init__()
        self._x_blk = x_blk
        self._y_blk = y_blk

class TenguFlowCounter(TenguObsoleteCounter):

    def __init__(self, frame_shape=(640, 480), flow_blocks=(20, 20), **kwargs):
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

        if len(self._flow_graph):
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
        count = super(TenguFlowCounter, self).count(tracklets)
        self._reporter.update_counts(count)

    def initialize_flow_graph(self):

        for x_blk in xrange(self._flow_blocks[0]):
            self._blk_node_map[x_blk] = {}
            for y_blk in xrange(self._flow_blocks[1]):
                flow_node = TenguFlowNode(x_blk, y_blk)
                self._flow_graph.add_node(flow_node)
                self._blk_node_map[x_blk][y_blk] = flow_node

    def flow_node_at(self, x, y):
        x_blk = self.get_x_blk(x)
        y_blk = self.get_y_blk(y)
        return self._blk_node_map[x_blk][y_blk]

    def get_x_blk(self, x):
        x = min(max(0., x), self._flows.shape[0])
        return int(x / self._flow_blocks[0])

    def get_y_blk(self, y):
        y = min(max(0., y), self._flows.shape[1])
        return int(y / self._flow_blocks[1])

    def add_new_tracklets(self, new_tracklets):
        pass

    def update_existing_tracklets(self, existing_tracklets):
        pass

    def finish_removed_tracklets(self, removed_tracklets):
        pass