#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import cv2
import numpy as np

class TenguSceneAnalyzer(object):

    def __init__(self, roi=None, scale=1.0):
        self.logger = logging.getLogger(__name__)
        # x1, y1, x2, y2
        self.roi = roi
        self.scale = scale
    
    def analyze_scene(self, frame):
        cropped = None
    	
        if self.scale == 1.0:
            cropped = frame
        else:
            cropped = cv2.resize(cropped, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        if self.roi is not None:
            # crop
            cropped = cropped[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        
        return cropped

class TenguNode(object):

    def __init__(self, tr, *argv):
        super(TenguNode, self).__init__(*argv)
        self.tr = tr

    def inside_rect(self, rect):
        x, y = self.tr[-1]
        return (x > rect[0] and x < (rect[0]+rect[2])) and (y > rect[1] and y < (rect[1]+rect[3]))

class KLTSceneAnalyzer(TenguSceneAnalyzer):

    _max_nodes = 1000
    
    def __init__(self, draw_flows=False, lk_params=None, feature_params=None, track_window=None, count_lines=None, **kwargs):
        super(KLTSceneAnalyzer, self).__init__(roi=[track_window[0], track_window[1], track_window[0]+track_window[2], track_window[1]+track_window[3]], **kwargs)

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

    @property
    def last_removed_nodes(self):
        removed_nodes = self._last_removed_nodes
        self._last_removed_nodes = []
        return removed_nodes

    def analyze_scene(self, frame):
        if self.roi is None:
            h = 480
            w = 640
            offset_y = max(0, int((frame.shape[0] - h) / 2))
            offset_x = max(0, int((frame.shape[1] - w) / 2))
            self.roi = [offset_x, offset_y, offset_x+w, offset_y+h]
        scene = super(KLTSceneAnalyzer, self).analyze_scene(frame)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        if self.draw_flows:
            self.debug = scene_gray.copy()
            self.draw_count_lines(self.debug, self.roi[0], self.roi[1])
        # calculate optical flow
        if len(self.nodes) > 0:
            self.nodes = self.calculate_flow(self.prev_gray, scene_gray)
            self.logger.debug('{} nodes are currently tracked'.format(len(self.nodes)))
        # update tracking points
        if self.frame_idx % self.update_interval == 0:
            mask = self.find_corners_to_track(scene_gray, self.roi[0], self.roi[1])
            if self.draw_flows:
                cv2.imshow('KLT Debug - Mask', mask)
        # set prev
        self.prev_gray = scene_gray
        self.frame_idx += 1

        if self.draw_flows:
            cv2.imshow('KLT Debug - Flows', self.debug)
            ch = 0xFF & cv2.waitKey(1)

        return scene

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

            node.tr.append((x, y))
            new_nodes.append(node)

            if self.draw_flows:
                cv2.circle(self.debug, (x, y), 3, (255, 255, 255), -1)

        if self.draw_flows:
            cv2.polylines(self.debug, [np.int32(node.tr) for node in new_nodes], False, (192, 192, 192))

        return new_nodes

    def find_corners_to_track(self, scene_gray, from_x, from_y):
        self.logger.debug('finding corners')
        # cleanup points outside of lanes and counters
        cleanup_outside_of_lanes = True
        if cleanup_outside_of_lanes:
            self.cleanup_outside_lane_nodes(scene_gray, from_x, from_y)

        # reset mask
        # every pixel is not tracked by default
        mask = np.zeros_like(scene_gray)
        # more masks
        use_counter = True
        if use_counter:
            for lane in self.count_lines:
                lines = len(self.count_lines[lane])
                for i, line in enumerate(self.count_lines[lane]):
                    if i == lines - 1:
                        # don't increase nodes at the last, counter line
                        continue
                    cv2.line(mask, (line[0][0] - from_x, line[0][1] - from_y), (line[0][2] - from_x, line[0][3] - from_y), 255, 3)
        # don't pick up existing pixels
        for x, y in [np.int32(node.tr[-1]) for node in self.nodes]:
            cv2.circle(mask, (x, y), 10, 0, -1)
        # find good points
        p = cv2.goodFeaturesToTrack(scene_gray, mask = mask, **self.feature_params)
        if p is None:
            print('No good features')
        else:
            for x, y in np.float32(p).reshape(-1, 2):
                new_node = TenguNode([(x, y)])
                self.nodes.append(new_node)
                if len(self.nodes) > KLTSceneAnalyzer._max_nodes:
                    self._last_removed_nodes.append(self.nodes[0])
                    del self.nodes[0]

        return mask

    def cleanup_outside_lane_nodes(self, scene_gray, from_x, from_y):
        mask = np.zeros_like(scene_gray)
        # at first, fill out lanes with 255 to clarify inside or outside
        for lane in self.count_lines:
            lane_points_1 = []
            lane_points_2 = []
            for line in self.count_lines[lane]:
                lane_points_1.append((line[0][0] - from_x, line[0][1] - from_y))
                lane_points_2.append((line[0][2] - from_x, line[0][3] - from_y))
                # wider the line
                cv2.line(mask, (line[0][0] - from_x, line[0][1] - from_y), (line[0][2] - from_x, line[0][3] - from_y), 255, 10)
            # draw logical view area
            lane_points_2_size = len(lane_points_2)
            for i in range(lane_points_2_size):
                lane_point_2 = lane_points_2[lane_points_2_size - 1 - i]
                lane_points_1.append(lane_point_2)
            # draw logical view area
            cv2.fillPoly(mask, [np.int32(lane_points_1)], 255)
        # then, iterate over current nodes, and remove if outside
        node_size = len(self.nodes)
        for n in range(node_size):
            node_index = node_size - 1 - n
            latest_point = self.nodes[node_index].tr[-1]
            y = int(latest_point[1])
            x = int(latest_point[0])
            if mask.shape[0] <= y or mask.shape[1] <= x or mask[y][x] == 0:
                #print('removing tracks leading from ' + str(latest_point) + ' at ' + str(node_index) )    
                del self.nodes[node_index]

    def draw_count_lines(self, debug, from_x, from_y):
        if len(self.count_lines) == 0:
            return

        for l, lane in enumerate(self.count_lines):
            for i, line in enumerate(self.count_lines[lane]):
                color = (0, 255, 255)
                if line[6]:
                    color = (0, 0, 255)
                cv2.line(debug, (line[0][0] - from_x, line[0][1] - from_y), (line[0][2] - from_x, line[0][3] - from_y), color, 3)
                # draw a perpendicular line to it
                cv2.arrowedLine(debug, (line[1][0]-from_x, line[1][1]-from_y), (line[1][2]-from_x, line[1][3]-from_y), color)
