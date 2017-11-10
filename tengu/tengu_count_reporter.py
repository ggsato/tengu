#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import StringIO
import logging

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

class TenguFpsAggregator(TenguAggregator):

    def __init__(self):
        self.current_frames = 0
        self.last_tick = -1
        self.fps_second = []
        self.fps_average = []
        self.total_elapsed = 0

    def aggregate(self, counts):
        if self.last_tick < 0:
            # for the first time
            self.last_tick = time.time()
            return

        self.current_frames += 1
        current_tick = time.time()
        elapsed = current_tick - self.last_tick
        logging.debug('elapsed = {}'.format(elapsed))
        if elapsed > 1:
            fps = int(self.current_frames / elapsed)
            self.fps_second.append(fps)
            if len(self.fps_second) > 60:
                self.fps_average.append([int(self.total_elapsed), int(sum(self.fps_second)/len(self.fps_second))])
                self.fps_second = []
            self.last_tick = current_tick
            self.current_frames = 0
            self.total_elapsed += elapsed

    @property
    def summary(self):
        logging.info('fps_second: {}, fps_average: {}'.format(self.fps_second, self.fps_average))
        return self.fps_average


class TenguFormatter(object):

    def format(self, summary):
        return summary

class TenguNumberListToCSVFormatter(TenguFormatter):

    def format(self, summary):
        sf = StringIO.StringIO()
        for item in summary:
            row = '{d[0]},{d[1]}\n'.format(d=item)
            sf.write(row)
        formatted_text = sf.getvalue()
        sf.close()
        logging.debug('formatted_text = {}'.format(formatted_text))
        return formatted_text

class TenguConsoleWriter(object):

    def write(self, formatted_txt):
        print('REPORT: {}'.format(formatted_txt))

class TenguFileWriter(TenguConsoleWriter):

    def __init__(self, filename):
        super(TenguFileWriter, self).__init__()
        self.filename = filename

    def write(self, formatted_txt):
        with open(self.filename, 'w') as f:
            f.write(formatted_txt)

class TenguCountReporter(object):

    def __init__(self, aggregator=TenguTotalAggregator(), formatter=TenguFormatter(), writer=TenguConsoleWriter()):
        self._aggregator = aggregator
        self._formatter = formatter
        self._writer = writer
        logging.debug("aggregator: {}, formatter:{}, writer:{}".format(self._aggregator, self._formatter, self._writer))
    
    def update_counts(self, counts):
        logging.debug('updating counts...')
        self._aggregator.aggregate(counts)

    def report(self):
        logging.info('making report...')
        self._writer.write(self._formatter.format(self._aggregator.summary))