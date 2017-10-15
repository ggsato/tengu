#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguObserver(object):
    """
    This is a base class of all the Tengu observers including Detector, Tracker, Counter, CountReporter, and any other GUI classes.
    """
    def src_changed(self, **kwargs):
        pass

    def scene_change(self, **kwargs):
        pass

    def frame_changed(self, frame, frame_no):
        pass

    def objects_detected(self, **kwargs):
        pass

    def tracking_updated(self, **kwargs):
        pass

    def object_counted(self, **kwargs):
        pass