#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time, sys, traceback
import cv2
from multiprocessing import Process, Queue, Value

class TenguSensor(Process):

    def __init__(self, input_queue_max_size=10, output_queue_max_size=20, output_queue_timeout_in_secs=10, **kwargs):
        super(TenguSensor, self).__init__(**kwargs)
        self.logger= logging.getLogger(__name__)

        self._output_queue_timeout_in_secs = output_queue_timeout_in_secs

        # items exchanged in these queue are of type TenguSensorItem
        # a queue to get some items as inputs for this sensor
        self._input_queue = Queue(maxsize=input_queue_max_size)
        # a queue to put some items as outputs from this sensor
        self._output_queue = Queue(maxsize=output_queue_max_size)

        self._finished = Value('i', 0)

    @property
    def needs_frame_input(self):
        return False

    @property
    def input_queue(self):
        return self._input_queue

    @property
    def output_queue(self):
        return self._output_queue

    def run(self):
        pass

    def finish(self):
        # then, mark finished
        self._finished.value = 1
        # consume all the items in both queues
        # input first
        self.logger.info('cleaning up input queue')
        while not self._input_queue.empty():
            self._input_queue.get_nowait()
        # output queue
        self.logger.info('cleaning up output queue')
        while not self._output_queue.empty():
            self._output_queue.get_nowait()

        # wait
        while self._finished.value != 2:
            self.logger.debug('waiting for exitting sensor loop, finished = {}'.format(self._finished.value))
            time.sleep(0.001)

class TenguObjectDetectionSensor(TenguSensor):

    def __init__(self, detector, **kwargs):
        super(TenguObjectDetectionSensor, self).__init__(**kwargs)
        self._detector = detector

    @property
    def needs_frame_input(self):
        return True

    def run(self):
        while self._finished.value == 0:
            sensored = 0
            while not self._input_queue.empty() and self._finished.value == 0:
                self.logger.info('got an input to sensor from a queue')
                sensor_input = self._input_queue.get_nowait()
                sensored += 1
                try:
                    img = cv2.imread(sensor_input.item)
                    detections, class_names = self._detector.detect(img)
                    sensor_output = TenguSensorItem(sensor_input.t, {'d': detections, 'n': class_names,'h': img.shape[0], 'w': img.shape[1]})
                    self.logger.info('detections at sensor = {}'.format(detections))
                    done = False
                    start = time.time()
                    elapsed = 0
                    while not done and elapsed < self._output_queue_timeout_in_secs and self._finished.value == 0:
                        try:
                            self.output_queue.put_nowait(sensor_output)
                            done = True
                        except:
                            time.sleep(0.001)
                        elapsed = (time.time() - start) / 1000
                except:
                    self.logger.error('sensor {} failed to get an image from {}'.format(self, sensor_input.item))
                    info = sys.exc_info()
                    self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
                    traceback.print_tb(info[2])
                if self._finished.value == 1:
                    break
            if sensored == 0:
                self.logger.info('no sensor input available, sleeping, finished? {}'.format(self._finished.value == 1))
                # no inputs were available, sleep a bit
                time.sleep(0.001)

        self._finished.value = 2

        self.logger.info('exitted sensor loop {}'.format(self._finished.value))

class TenguSensorItem(object):

    def __init__(self, t, sensor_item):
        super(TenguSensorItem, self).__init__()

        # time t when this item is associated
        self._t = t

        # a list of items
        self._item = sensor_item

    @property
    def t(self):
        return self._t

    @property
    def item(self):
        return self._item