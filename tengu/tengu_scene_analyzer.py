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

class KLTSceneAnalyzer(TenguSceneAnalyzer):
    
    def __init__(self, lk_params=None, feature_params=None, **kwargs):
        super(KLTSceneAnalyzer, self).__init__(**kwargs)
        self.frame_idx = 0
        self.update_interval = 10
        self.tracks = [] 
        self.max_track_length = 100
        self.lk_params = lk_params
        self.feature_params = feature_params
        self.prev_gray = None

    def analyze_scene(self, frame):
        scene = super(KLTSceneAnalyzer, self).analyze_scene(frame)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        if len(self.tracks) > 0:
            self.tracks = self.calculate_flow(self.prev_gray, scene_gray)
            self.logger.debug('{} tracks are currently tracked'.format(len(self.tracks)))
        # update tracking points
        if self.frame_idx % self.update_interval == 0:
            self.find_corners_to_track(scene_gray)
        # set prev
        self.prev_gray = scene_gray
        self.frame_idx += 1
        return scene

    def calculate_flow(self, img0, img1):
        self.logger.debug('calculating flow')
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue

            # if len(tr) > self.max_track_length / 10:
            #     full_flow_length = SceneFlow.compute_flow_length(tr)
            #     if full_flow_length < self.scene_flow.flow_block_size:
            #         # ignore this track
            #         # the corresponding block will stay marked as stationally
            #         print('ignoring a track at ' + str(tr[-1]))
            #         continue
            if len(tr) > self.max_track_length:
                del tr[0]

            tr.append((x, y))
            new_tracks.append(tr)

        return new_tracks

    def find_corners_to_track(self, scene_gray):
        self.logger.debug('finding corners')
        # create mask
        mask = np.ones_like(scene_gray) * 255
        # don't pick up existing pixels
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 10, 0, -1)
        # find good points
        p = cv2.goodFeaturesToTrack(scene_gray, mask = mask, **self.feature_params)
        if p is None:
            print('No good features')
        else:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])