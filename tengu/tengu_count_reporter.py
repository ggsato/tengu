#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TenguAggregator(object):

    def aggregate(self, counts):
        pass

    @property
    def summary(self):
        pass

class TenguTotalAggregator(TenguAggregator):

    def __init__(self):
        self._total_counts = 0

    def aggregate(self, counts):
        self._total_counts += counts

    @property
    def summary(self):
        return self._total_counts

class TenguFormatter(object):

    def format(self, summary):
        return summary

class TenguConsoleWriter(object):

    def write(self, formatted_txt):
        print('TOTAL: {}'.format(formatted_txt))

class TenguCountReporter(object):

    def __init__(self, aggregator=TenguTotalAggregator(), formatter=TenguFormatter(), writer=TenguConsoleWriter()):
        self._aggregator = aggregator
        self._formatter = formatter
        self._writer = writer
    
    def update_counts(self, counts):
        self._aggregator.aggregate(counts)

    def report(self):
        self._writer.write(self._formatter.format(self._aggregator.summary))