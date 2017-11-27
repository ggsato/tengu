#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import time
import math
import Queue
import logging

from weakref import WeakValueDictionary

from tengu_observer import *

class Tengu(object):

    def __init__(self):
        self.logger= logging.getLogger(__name__)

        self._observers = WeakValueDictionary()
        self._current_frame = -1
        self._stopped = False

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

    def run(self, src=None, roi=None, scale=None, every_x_frame=1, rotation=0, tengu_flow_analyzer=None, tengu_counter=None, queue=None, max_queue_wait=10):
        """
        the caller should register by add_observer before calling run if it needs updates during analysis
        this should return quicky for the caller to do its own tasks especially showing progress graphically
        """
        self.logger.debug('running with flow_analyzer:{}, counter:{}'.format(tengu_flow_analyzer, tengu_counter))
        
        if src is None:
            self.logger.error('src has to be set')
            return

        try:
            cam = cv2.VideoCapture(int(src))
        except:
            cam = cv2.VideoCapture(src)        
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

            # debug
            if self._current_frame > 10000:
                break

            # rotate
            if rotation != 0:
                rows, cols, channels = frame.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
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

            # preprocess
            cropped = self.preprocess(frame, roi, scale)
            self._notify_frame_changed(cropped)

            # skip if necessary
            if every_x_frame > 1 and self._current_frame % every_x_frame != 0:
                self._current_frame += 1
                self.logger.debug('skipping frame at {}'.format(self._current_frame))
                continue

            # detect
            if tengu_flow_analyzer is not None:
                detections, tracklets, scene = tengu_flow_analyzer.analyze_flow(cropped, self._current_frame)
                self._notify_objects_detected(detections)
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

    def preprocess(self, frame, roi, scale):

        cropped = None
        
        if scale == 1.0:
            cropped = frame
        else:
            cropped = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if roi is not None:
            # crop
            cropped = cropped[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        return cropped

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
