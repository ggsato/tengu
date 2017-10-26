#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

class TenguCounter(object):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def count(self, tracked_objects):
        pass

class TenguObsoleteCounter(TenguCounter):

    def __init__(self):
        super(TenguObsoleteCounter, self).__init__()
        self._last_tracked_objects = None

    def count(self, tracked_objects):
        count = 0

        if self._last_tracked_objects is None:
            self._last_tracked_objects = tracked_objects
            return count

        for last_tracked_object in self._last_tracked_objects:
            confirmed = last_tracked_object.is_confirmed
            removed = not (last_tracked_object in tracked_objects)
            if confirmed and removed:
                # this means this tracked_object became obsolete
                count += 1
        
        self._last_tracked_objects = tracked_objects
        return count