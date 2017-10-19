#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguObserver(object):
    """
    This is a base class of all the Tengu observers including Detector, Tracker, Counter, CountReporter, and any other GUI classes.
    """
    def src_changed(self, src):
        pass

    def scene_change(self, **kwargs):
        pass

    def frame_changed(self, frame, frame_no):
        pass

    def objects_detected(self, detections):
        pass

    def trackings_updateded(self, tracked_objects):
        pass

    def object_counted(self, **kwargs):
        pass

    def analysis_finished(self):
        pass