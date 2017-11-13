#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, cv2

from ..tengu_observer import TenguFrameChangeObserver
from ..tengu_tracker import TenguTracker, Tracklet

class OpenCVTracklet(Tracklet):

    def __init__(self, tracker):
        super(OpenCVTracklet, self).__init__()
        self._tracker = tracker
        self._initialized = False

    @property
    def tracker(self):
        return _tracker

    def initialize(self, frame):
        self._initialized = self._tracker.init(frame, self.rect)
        return self._initialized

    def update_tracker(self, frame):
        if not self.is_confirmed:
            return

        if not self._initialized:
            self.initialize(frame)
            return

        updated, rect = self._tracker.update(frame)
        if updated:
            super(OpenCVTracklet, self).update_tracking(rect)

class OpenCVTracker(TenguTracker, TenguFrameChangeObserver):

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']

    def __init__(self, tracker_type):
        super(OpenCVTracker, self).__init__()
        self.tracker_type = tracker_type
        self.last_frame = None

    def frame_changed(self, frame, frame_no):
        self.last_frame = frame
    
    def resolve_trackings(self, detections):

        """
        1. initialize a tracker for a new tracklet
        OpenCVTracker is an online tracker that should be given an initial rectangle to track,
        then, it will learn and update to track further.
        2. update a tracker for an existing tracklet
        """
        return super(OpenCVTracker, self).resolve_trackings(detections)

    def prepare_updates(self, detections):
        for tracklet in self._tracklets:
            tracklet.update_tracker(self.last_frame)

    def new_tracklet(self, detection):
        tracker = OpenCVTracker.create_opencv_tracker(self.tracker_type)
        to = OpenCVTracklet(tracker)
        to.update_with_assignment(detection)
        return to

    @staticmethod
    def create_opencv_tracker(tracker_type):
        tracker = None

        if not tracker_type in OpenCVTracker.tracker_types:
            logging.error('{} is not available.')
            return tracker

        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()

        return tracker