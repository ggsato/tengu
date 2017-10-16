#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tengu_observer import TenguObserver

class TenguTracker(TenguObserver):
    
    def update_trackings(self, detections):
        return []