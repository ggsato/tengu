#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguObserver(object):
    """ a base class to register as an observer, and get notified when every frame gets analyzed
    """
    def frame_analyzed(self, event_dict):
        pass

    def analysis_finished(self):
        pass