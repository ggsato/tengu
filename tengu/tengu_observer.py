#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguFrameChangeObserver(object):
    def frame_changed(self, frame, frame_no):
        pass

    def frame_preprocessed(self, preprocessed):
        pass

class TenguObjectsDetectionObserver(object):
    def objects_detected(self, detections):
        pass

class TenguTrackletsUpdateObserver(object):
    def tracklets_updated(self, tracklets):
        pass

class TenguAnalysisObserver(object):
    def analysis_finished(self):
        pass

class TenguObserver(TenguFrameChangeObserver, TenguObjectsDetectionObserver, TenguTrackletsUpdateObserver, TenguAnalysisObserver):
    """
    This is a base class of all the Tengu observers including Detector, Tracker, Counter, CountReporter, and any other GUI classes.
    """
    pass