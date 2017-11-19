#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import time
import math
import Queue
import logging

from flow import SceneFlow, FlowBlock
from weakref import WeakValueDictionary

from tengu_observer import *

class Tengu(object):

    def __init__(self, every_x_frame=1, rotation=0):
        self.logger= logging.getLogger(__name__)

        self._observers = WeakValueDictionary()
        self._src = None
        self._current_frame = -1
        self._every_x_frame = every_x_frame
        self._rotation = rotation

        self._stopped = False

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        if src == None:
            self.logger.debug('src should not be None')
            return

        if src == self._src:
            self.logger.debug('{} is already set')
            return

        if self._current_frame >= 0:
            self.logger.warning('stop running before changing src')
            return
        
        self.logger.debug('src changed from {} to {}'.format(self._src, src))
        self._src = src
        self._stopped = False

        # notifiy observers
        self._notify_src_changed()

    @property
    def frame_no(self):
        return self._current_frame

    def add_observer(self, observer):
        if observer == None:
            return
        observer_id = id(observer)
        if self._observers.has_key(observer_id):
            return
        self._observers[observer_id] = observer

    def remove_observer(self, observer):
        if observer == None:
            return
        observer_id = id(observer)
        if self._observers.has_key(observer_id):
            del self._observers[observer_id]

    def _notify_src_changed(self):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguSrcChangeObserver):
                observer.src_changed(self.src)

    def _notify_scene_changed(self, scene):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguSceneChangeObserver):
                observer.scene_changed(scene)

    def _notify_tracklets_updated(self, tracklets):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguTrackletsUpdateObserver):
                observer.tracklets_updated(tracklets)

    def _notify_frame_changed(self, frame):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguFrameChangeObserver):
                observer.frame_changed(frame, self._current_frame)

    def _notify_objects_detected(self, detections):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguObjectsDetectionObserver):
                observer.objects_detected(detections)

    def _notify_objects_counted(self, count):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguObjectsCountObserver):
                observer.objects_counted(count)

    def _notify_analysis_finished(self):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguAnalysisObserver):
                observer.analysis_finished()

    def run(self, tengu_scene_analyzer=None, tengu_detector=None, tengu_tracker=None, tengu_counter=None, queue=None, max_queue_wait=10):
        """
        the caller should register by add_observer before calling run if it needs updates during analysis
        this should return quicky for the caller to do its own tasks especially showing progress graphically
        """
        self.logger.debug('running with scene_analyzer:{}, detector:{}, tracker:{}, counter:{}'.format(tengu_scene_analyzer, tengu_detector, tengu_tracker, tengu_counter))
        
        try:
            cam = cv2.VideoCapture(int(self._src))
        except:
            cam = cv2.VideoCapture(self._src)        
        if cam is None or not cam.isOpened():
            self.logger.debug(self._src + ' is not available')
            return

        while not self._stopped:
            ret, frame = cam.read()
            #self.logger.info('frame shape = {}'.format(frame.shape))
            if not ret:
                self.logger.debug('no frame is avaiable')
                break

            # increment
            self._current_frame += 1

            # rotate
            if self._rotation != 0:
                rows, cols, channels = frame.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), self._rotation, 1)
                frame = cv2.warpAffine(frame, M, (cols, rows))

            # block for a client if necessary to synchronize
            if queue is not None:
                # wait until queue becomes ready
                try:
                    queue.put(frame, max_queue_wait)
                except Queue.Full:
                    self.logger.error('queue is full, quitting...')
                    break

            # use copy for gui use, which is done asynchronously, meaning may corrupt buffer during camera updates
            copy = frame.copy()

            # notify
            self._notify_frame_changed(copy)

            # analyze scene
            if tengu_scene_analyzer is not None:
                scene = tengu_scene_analyzer.analyze_scene(frame)
                self._notify_scene_changed(scene)
            else:
                scene = frame

            # skip if necessary
            if self._every_x_frame > 1 and self._current_frame % self._every_x_frame != 0:
                self._current_frame += 1
                self.logger.debug('skipping frame at {}'.format(self._current_frame))
                continue

            # detect
            if tengu_detector is not None:
                detections = tengu_detector.detect(scene)
                self._notify_objects_detected(detections)

                # tracking-by-detection
                if tengu_tracker is not None:
                    tracklets = tengu_tracker.resolve_trackings(detections)
                    self._notify_tracklets_updated(tracklets)

                    # count trackings
                    if tengu_counter is not None:
                        self.logger.debug('calling counter')
                        counts = tengu_counter.count(tracklets)
                        self._notify_objects_counted(counts)

                    else:
                        self.logger.debug('skip calling counter')

            # only for debugging
            #time.sleep(1)

        self.logger.info('exitted run loop, exitting...')
        if tengu_counter is not None:
            tengu_counter.count(None)
        self._notify_analysis_finished()
        self._stopped = True

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.debug('stopping...')
        self._stopped = True

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
