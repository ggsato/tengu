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
        cropped = frame
    	if self.roi is not None:
            # crop
            cropped = frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        if self.scale == 1.0:
            return cropped
        
        return cv2.resize(cropped, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

class TenguNode(object):

    def __init__(self, tr, *argv):
        super(TenguNode, self).__init__(*argv)
        self.tr = tr

class KLTSceneAnalyzer(TenguSceneAnalyzer):
    
    def __init__(self, draw_flows=False, lk_params=None, feature_params=None, **kwargs):
        super(KLTSceneAnalyzer, self).__init__(**kwargs)
        self.draw_flows = draw_flows
        self.frame_idx = 0
        self.update_interval = 10
        self.nodes = [] 
        self.max_track_length = 100
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.prev_gray = None
        self._last_added_nodes = []
        self._last_removed_nodes = []

    @property
    def last_added_nodes(self):
        added_nodes = self._last_added_nodes
        self._last_added_nodes = []
        return added_nodes

    @property
    def last_removed_nodes(self):
        removed_nodes = self._last_removed_nodes
        self._last_removed_nodes = []
        return removed_nodes

    def analyze_scene(self, frame):
        scene = super(KLTSceneAnalyzer, self).analyze_scene(frame)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        if len(self.nodes) > 0:
            self.nodes = self.calculate_flow(self.prev_gray, scene_gray)
            self.logger.debug('{} nodes are currently tracked'.format(len(self.nodes)))
        # update tracking points
        if self.frame_idx % self.update_interval == 0:
            self.find_corners_to_track(scene_gray)
        # set prev
        self.prev_gray = scene_gray
        self.frame_idx += 1

        if self.draw_flows:
            return scene_gray

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
                cv2.circle(img1, (x, y), 3, (255, 255, 255), -1)

        if self.draw_flows:
            cv2.polylines(img1, [np.int32(node.tr) for node in new_nodes], False, (192, 192, 192))

        return new_nodes

    def find_corners_to_track(self, scene_gray):
        self.logger.debug('finding corners')
        # create mask
        mask = np.ones_like(scene_gray) * 255
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
                self._last_added_nodes.append(new_node)