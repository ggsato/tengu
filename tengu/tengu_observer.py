#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguSceneChangeObserver(object):
    def scene_changed(self, frame):
        pass

class TenguFrameChangeObserver(object):
    def frame_changed(self, frame, frame_no):
        pass

class TenguObjectsDetectionObserver(object):
    def objects_detected(self, detections):
        pass

class TenguTrackletsUpdateObserver(object):
    def tracklets_updated(self, tracklets):
        pass

class TenguObjectsCountObserver(object):
    def objects_counted(self, count):
        pass

class TenguAnalysisObserver(object):
    def analysis_finished(self):
        pass

class TenguObserver(TenguSceneChangeObserver, TenguFrameChangeObserver, TenguObjectsDetectionObserver, TenguTrackletsUpdateObserver, TenguObjectsCountObserver, TenguAnalysisObserver):
    """
    This is a base class of all the Tengu observers including Detector, Tracker, Counter, CountReporter, and any other GUI classes.
    """
    pass