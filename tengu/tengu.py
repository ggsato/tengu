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

    def __init__(self):
        self.logger= logging.getLogger(__name__)

        self._observers = WeakValueDictionary()
        self._src = None
        self._current_frame = -1

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

    def _notify_tracked_objects_updated(self, tracked_objects):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            if isinstance(observer, TenguTrackedObjectsUpdateObserver):
                observer.tracked_objects_updated(tracked_objects)

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

    def run(self, tengu_scene_analyzer=None, tengu_detector=None, tengu_tracker=None, tengu_counter=None, tengu_count_reporter=None, queue=None, max_queue_wait=10):
        """
        the caller should register by add_observer before calling run if it needs updates during analysis
        this should return quicky for the caller to do its own tasks especially showing progress graphically
        """
        self.logger.debug('running with scene_analyzer:{}, detector:{}, tracker:{}, counter:{}, count_reporter:{}'.format(tengu_scene_analyzer, tengu_detector, tengu_tracker, tengu_counter, tengu_count_reporter))
        
        try:
            cam = cv2.VideoCapture(int(self._src))
        except:
            cam = cv2.VideoCapture(self._src)        
        if cam is None or not cam.isOpened():
            self.logger.debug(self._src + ' is not available')
            return

        while not self._stopped:
            ret, frame = cam.read()
            if not ret:
                self.logger.debug('no frame is avaiable')
                break

            # TEST
            if self._current_frame > 10000:
                break
            self._current_frame += 1
            # block for a client if necessary to synchronize
            if queue is not None:
                # wait until queue becomes ready
                try:
                    queue.put(frame, max_queue_wait)
                except Queue.Full:
                    self.logger.error('queue is full, quitting...')
                    break
            # notify
            self._notify_frame_changed(frame)

            # detect
            if tengu_detector is not None:
                detections = tengu_detector.detect(frame)
                self._notify_objects_detected(detections)

                # tracking-by-detection
                if tengu_tracker is not None:
                    tracked_objects = tengu_tracker.resolve_trackings(detections)
                    self._notify_tracked_objects_updated(tracked_objects)

                    # count trackings
                    if tengu_counter is not None:
                        self.logger.debug('calling counter')
                        counts = tengu_counter.count(tracked_objects)
                        self._notify_objects_counted(counts)

                        # update for report
                        if tengu_count_reporter is not None:
                            self.logger.debug('calling count reporter {}'.format(tengu_count_reporter))
                            tengu_count_reporter.update_counts(counts)
                        else:
                            self.logger.debug('skip calling count reporter')

                    else:
                        self.logger.debug('skip calling counter')

        self.logger.info('exitted run loop, exitting...')
        if tengu_count_reporter is not None:
            tengu_count_reporter.report()
        self._notify_analysis_finished()
        self._stopped = True

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.debug('stopping...')
        self._stopped = True

class App:

    lk_params = dict( winSize  = (21, 21), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    feature_params = dict( maxCorners = 1000, qualityLevel = 0.1, minDistance = 10, blockSize = 10)

    def __init__(self, video_src, global_scale):
        self.cam = cv2.VideoCapture(video_src)
        self.video_src = video_src
        self.global_scale = global_scale   
        self.tracks = [] 
        self.frame_idx = 0
        self.detect_interval = 10
        # e.g. 25fps * 4 seconds * 10 = 40 seconds
        self.max_track_length = 1000
        self.scene_flow = None


    def calculate_flow(self, img0, img1, frame_gray):
        start_time = time.time()
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **App.lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **App.lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                # this track is not effective anymore
                self.scene_flow.update_flow(tr)
                continue

            if frame_gray.shape[0] <= int(y) or frame_gray.shape[1] <= int(x):
                # this track has left the region
                self.scene_flow.update_flow(tr)
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
            cv2.circle(frame_gray, (x, y), 3, (255, 255, 255), -1)

        cv2.polylines(frame_gray, [np.int32(tr) for tr in new_tracks], False, (192, 192, 192))

        elapsed_time = time.time() - start_time
        #print('calculated and clustered in (' + str(elapsed_time) + ')sec')

        return new_tracks

    def update_tracking_points(self, frame_gray):
        # ignore static flow block areas
        tmp_flows = self.scene_flow.get_flows(False)
        tmp_flows_gray = cv2.cvtColor(tmp_flows, cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(tmp_flows_gray,16,255,cv2.THRESH_BINARY)
        # don't pick up existing pixels
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 10, 0, -1)
        # find good points
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **App.feature_params)
        if p is None:
            print('No good features')
        else:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])

        return mask

    def run(self):
        cv2.namedWindow('Flows')
        prev_gray = None
        frame_lines = None
        show_edges = False
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print('no frame is avaiable')
                break

            resized = cv2.resize(frame,None, fx=self.global_scale, fy=self.global_scale, interpolation=cv2.INTER_AREA)
            frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if show_edges and frame_lines is None:
                frame_lines = np.ones_like(frame_gray) * 255
            
            # prepare flows
            if self.scene_flow is None:
                print('initializing shape as ' + str(resized.shape))
                self.scene_flow = SceneFlow(resized.shape)
                print('initialized shape as ' + str(self.scene_flow.get_flows(False).shape))

            # calculate optical flow
            if len(self.tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                self.tracks = self.calculate_flow(img0, img1, frame_gray)

            # update tracking points
            if self.frame_idx % self.detect_interval == 0:
                mask = self.update_tracking_points(frame_gray)
                if show_edges:
                    # get lines
                    new_frame_lines = np.ones_like(frame_lines) * 255
                    edges = cv2.Canny(frame_gray,10,150,apertureSize = 3)
                    masked_edges = cv2.bitwise_and(edges, mask)
                    # probabilistic
                    lines = cv2.HoughLinesP(masked_edges,1,np.pi/180,50,minLineLength=3,maxLineGap=0)
                    if lines is not None:
                        for line in lines:
                            x1,y1,x2,y2 = line[0]
                            # ignore if this is a similar direction to the flow direction
                            blk_angle = self.scene_flow.get_angle_at((x1, y1))
                            line_angle = SceneFlow.get_angle([(x1, y1), (x2, y2)])
                            diff_angle = math.fabs(blk_angle - line_angle)
                            if diff_angle < (math.pi/6):
                                continue
                            else:
                                # inverse line angle
                                line_angle = SceneFlow.get_angle([(x2, y2), (x1, y1)])
                                diff_angle = math.fabs(blk_angle - line_angle)
                                if diff_angle < (math.pi/6):
                                    continue
                            cv2.line(new_frame_lines,(x1,y1),(x2,y2),0,2)
                    # blend
                    frame_lines = cv2.addWeighted(frame_lines,0.9,new_frame_lines,0.1,0)

            # prepare for the next
            self.frame_idx += 1
            prev_gray = frame_gray

            cv2.imshow('Frame', frame_gray)
            #cv2.imshow('Mask', mask)
            cv2.imshow('Flows', self.scene_flow.get_flows())
            if show_edges:
                cv2.imshow('Lines', frame_lines)

            ch = 0xFF & cv2.waitKey(1)

            if ch == 27:
                break
            elif ch == -1:
                continue
            elif ch == 255:
                # do nothing
                continue

        imfile = self.video_src[self.video_src.rindex('/')+1:self.video_src.index('.')]+'.jpg'
        print('saving ' + imfile)
        cv2.imwrite(imfile, self.scene_flow.get_flows())

        cv2.destroyAllWindows()

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
