#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguSrcChangeObserver(object):
    def src_changed(self, src):
        pass

class TenguSceneChangeObserver(object):
    def scene_change(self, **kwargs):
        pass

class TenguFrameChangeObserver(object):
    def frame_changed(self, frame, frame_no):
        pass

class TenguObjectsDetectionObserver(object):
    def objects_detected(self, detections):
        pass

class TenguTrackedObjectsUpdateObserver(object):
    def tracked_objects_updated(self, tracked_objects):
        pass

class TenguObjectsCountObserver(object):
    def objects_counted(self, count):
        pass

class TenguAnalysisObserver(object):
    def analysis_finished(self):
        pass

class TenguObserver(TenguSrcChangeObserver, TenguSceneChangeObserver, TenguFrameChangeObserver, TenguObjectsDetectionObserver, TenguTrackedObjectsUpdateObserver, TenguObjectsCountObserver, TenguAnalysisObserver):
    """
    This is a base class of all the Tengu observers including Detector, Tracker, Counter, CountReporter, and any other GUI classes.
    """
    pass