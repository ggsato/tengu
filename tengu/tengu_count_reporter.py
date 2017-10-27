#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

class TenguCountReporter(object):
    
    def update_counts(self, counts):
        pass

    def report(self):
        pass

class TenguTotalCountReporter(TenguCountReporter):

    def __init__(self):
        self.logger= logging.getLogger(__name__)
        self._total_counts = 0

    @property
    def total_counts(self):
        return self._total_counts

    def update_counts(self, counts):
        self._total_counts += counts

    def report(self):
        self.logger.info('TOTAL COUNTS = {}'.format(self._total_counts))