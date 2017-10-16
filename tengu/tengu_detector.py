#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tengu_observer import TenguObserver

class TenguDetector(TenguObserver):
    
    def detect(self, frame):
        return []