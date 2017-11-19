#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from sets import Set
import networkx as nx
import numpy as np
import cv2
from operator import attrgetter

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

    def __init__(self, y_blk, x_blk, position):
        super(TenguFlowNode, self).__init__()
        self._y_blk = y_blk
        self._x_blk = x_blk
        self._position = position
        self._source_count = 0
        self._sink_count = 0
        # pair of source, count
        self._sources = {}

    def __repr__(self):
        return 'y_blk={}, x_blk={}, position={}, source_count={}, sink_count={}'.format(self._y_blk, self._x_blk, self._position, self._source_count, self._sink_count)

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

class TenguFlowCounter(TenguObsoleteCounter):

    def __init__(self, frame_shape=(480, 640), flow_blocks=(20, 20), show_graph=True, **kwargs):
        super(TenguFlowCounter, self).__init__(**kwargs)
        self._frame_shape = frame_shape
        self._flow_blocks = flow_blocks
        self._flow_blocks_size = (int(self._frame_shape[0]/self._flow_blocks[0]), int(self._frame_shape[1]/self._flow_blocks[1]))
        self._flow_graph = nx.DiGraph()
        self._blk_node_map = {}
        self._show_graph = show_graph

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

        if self._show_graph:
            # show
            img = self.draw_graph()
            cv2.imshow('TenguFlowCounter Graph', img)
            ch = 0xFF & cv2.waitKey(1)

        # update reporter
        counts = super(TenguFlowCounter, self).count(tracklets)
        self._reporter.update_counts(counts)

        return counts

    def initialize_flow_graph(self):

        for y_blk in xrange(self._flow_blocks[0]):
            self._blk_node_map[y_blk] = {}
            for x_blk in xrange(self._flow_blocks[1]):
                pos_x = int(self._flow_blocks_size[1]*x_blk + self._flow_blocks_size[1]/2)
                pos_y = int(self._flow_blocks_size[0]*y_blk + self._flow_blocks_size[0]/2)
                flow_node = TenguFlowNode(y_blk, x_blk, (pos_x, pos_y))
                self.logger.info('created at y_blk,x_blk = {}, {} = {}'.format(y_blk, x_blk, (pos_x, pos_y)))
                self._flow_graph.add_node(flow_node)
                self._blk_node_map[y_blk][x_blk] = flow_node

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
            flow_node.mark_sink(removed_tracklet.flows[0])
            self.logger.info('sink at {}'.format(flow_node))

    def draw_graph(self):
        img = np.ones(self._frame_shape, dtype=np.uint8) * 128
        diameter = min(*self._flow_blocks_size)
        out_edges_list = []
        for y_blk in xrange(self._flow_blocks[0]):
            for x_blk in xrange(self._flow_blocks[1]):
                flow_node = self._blk_node_map[y_blk][x_blk]
                is_source = flow_node.source_count > flow_node.sink_count
                if is_source:
                    source_count = flow_node.source_count
                    if source_count == 0:
                        continue
                    color = min(255, 128+source_count)
                else:
                    sink_count = flow_node.sink_count
                    if sink_count == 0:
                        continue
                    color = max(0, 128-sink_count)
                cv2.circle(img, flow_node.position, int(diameter/2), color, -1)

                # out edges
                out_edges = self._flow_graph.out_edges(flow_node)
                out_edges_list.append(out_edges)
        for out_edges in out_edges_list:
            for out_edge in out_edges:
                color = min(255, 128+self._flow_graph[out_edge[0]][out_edge[1]]['weight'])
                cv2.arrowedLine(img, out_edge[0].position , out_edge[1].position, color, thickness=2, tipLength=0.5)
        # finally show top N src=>sink
        major_sinks = sorted(self._flow_graph, key=attrgetter('sink_count'), reverse=True)
        majority = int(len(self._flow_graph)/100*2)
        for major_sink in major_sinks[:majority]:
            if major_sink.sink_count < 10:
                continue
            source_nodes = major_sink._sources.keys()
            source_nodes = sorted(source_nodes, key=major_sink._sources.__getitem__, reverse=True)
            major_source = source_nodes[0]
            cv2.arrowedLine(img, major_source.position , major_sink.position, 255, thickness=5, tipLength=0.1)
        return img