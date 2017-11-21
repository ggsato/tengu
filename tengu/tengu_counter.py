#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from sets import Set

class TenguCounter(object):

    def __init__(self, reporter=None):
        self.logger = logging.getLogger(__name__)
        self._reporter = reporter
    
    def count(self, tracklets):
        pass

class TenguObsoleteCounter(TenguCounter):

    def __init__(self, **kwargs):
        super(TenguObsoleteCounter, self).__init__(**kwargs)
        self._last_tracklets = Set([])

    def count(self, tracklets):
        current_tracklets = Set(tracklets)

        if len(self._last_tracklets) == 0:
            self._last_tracklets = current_tracklets
            return 0

        obsolete_count = self._last_tracklets - current_tracklets
        
        self._last_tracklets = current_tracklets
        return obsolete_count