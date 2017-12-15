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

    def __init__(self, equalize=False):
        self.logger= logging.getLogger(__name__)
        self._equalize = equalize

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

    def _notify_frame_changed(self, frame):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguFrameChangeObserver):
                observer.frame_changed(frame, self._current_frame)

    def _notify_frame_preprocessed(self, preprocessed):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguFrameChangeObserver):
                observer.frame_preprocessed(preprocessed)

    def _notify_tracklets_updated(self, tracklets):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguTrackletsUpdateObserver):
                observer.tracklets_updated(tracklets)

    def _notify_objects_detected(self, detections, class_names):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguObjectsDetectionObserver):
                observer.objects_detected(detections, class_names)

    def _notify_analysis_finished(self):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguAnalysisObserver):
                observer.analysis_finished()

    def run(self, src=None, roi=None, scale=None, every_x_frame=1, rotation=0, tengu_flow_analyzer=None, tengu_scene_analyzer=None, queue=None, max_queue_wait=10, skip_to=-1):
        """
        the caller should register by add_observer before calling run if it needs updates during analysis
        this should return quicky for the caller to do its own tasks especially showing progress graphically
        """
        self.logger.info('running with flow_analyzer:{}, scene_analyzer:{}, skipping to {}'.format(tengu_flow_analyzer, tengu_scene_analyzer, skip_to))
        
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
            #if self._current_frame > 1000:
            #    break

            # rotate
            if rotation != 0:
                rows, cols, channels = frame.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
                frame = cv2.warpAffine(frame, M, (cols, rows))

            # use copy for gui use, which is done asynchronously, meaning may corrupt buffer during camera updates
            copy = frame.copy()
            self._notify_frame_changed(frame)

            # preprocess
            cropped = self.preprocess(frame, roi, scale)
            self._notify_frame_preprocessed(cropped.copy())

            # block for a client if necessary to synchronize
            if queue is not None:
                done = False
                while not done and not self._stopped:
                    # wait until queue becomes ready
                    try:
                        queue.put_nowait(cropped.copy())
                        done = True
                    except Queue.Full:
                        self.logger.debug('queue is full, quitting...')

                if self._stopped:
                    break

            # skip if necessary
            if (every_x_frame > 1 and self._current_frame % every_x_frame != 0) or (skip_to > 0 and self._current_frame < skip_to):
                self._current_frame += 1
                self.logger.debug('skipping frame at {}'.format(self._current_frame))
                continue

            # detect
            if tengu_flow_analyzer is not None:
                detections, class_names, tracklets, scene = tengu_flow_analyzer.analyze_flow(cropped, self._current_frame)
                self._notify_objects_detected(detections, class_names)
                self._notify_tracklets_updated(tracklets)

                # count trackings
                if tengu_scene_analyzer is not None:
                    self.logger.debug('calling scene analyzer')
                    tengu_scene_analyzer.analyze_scene(cropped, scene)

                else:
                    self.logger.debug('skip calling scene analyzer')

            # only for debugging
            #time.sleep(1)

        self.logger.info('exitted run loop, exitting...')
        if tengu_scene_analyzer is not None:
            tengu_scene_analyzer.finish_analysis()
        self._notify_analysis_finished()
        self._stopped = True

    def preprocess(self, frame, roi, scale):

        cropped = None
        
        if scale == 1.0:
            resized = frame
        else:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if roi is not None:
            # crop
            cropped = resized[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        else:
            cropped = resized

        if self._equalize:
            # at first, apply bilateral filter
            blurred = cv2.bilateralFilter(cropped, 8, 75, 75)

            # then, equalize
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            ycb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            y,c,b = cv2.split(ycb)
            y_eq = clahe.apply(y)
            ycb_eq = cv2.merge((y_eq, c, b))
            equalized = cv2.cvtColor(ycb_eq, cv2.COLOR_YCrCb2BGR)
            cropped = equalized
        
        return cropped

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.info('stopping...')
        self._stopped = True

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
