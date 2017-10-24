#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..tengu_tracker import TenguTracker

class KCFTracker(TenguTracker):

    def __init__(self):
        super(OverlapRatioTracker, self).__init__()
    
    def resolve_trackings(self, detections):

        return super(OverlapRatioTracker, self).resolve_trackings(detections)