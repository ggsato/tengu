#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
from multiprocessing import Process, Queue, Value

from tengu_flow_analyzer import TenguFlowAnalyzer
from tengu_sensor import TenguSensorItem

class TenguSceneModel(Process):
    """ TenguSceneModel repreesnts a model of scene, where a movement of each travelling Tracklet is predicted by a continuous sensor update respectively. 
    """

    def __init__(self, input_queue_max_size=20, output_queue_max_size=10, **kwargs):
        super(TenguSceneModel, self).__init__(**kwargs)

        self.logger= logging.getLogger(__name__)

        # a list of sensors
        self._frame_sensors = []
        self._extra_sensors = []

        self._flow_analyzer = TenguFlowAnalyzer()

        # queues
        self._intput_queue = Queue(maxsize=input_queue_max_size)
        self._output_queue = Queue(maxsize=output_queue_max_size)

        # time, incremented one by delta_t
        self._t = 0

        # finish if 1, this has to be Value because it is exchanged between processes
        self._finished = Value('i', 0)

    @property
    def input_queue(self):
        return self._intput_queue

    @property
    def output_queue(self):
        return self._output_queue

    def initialize(self, sensors):
        for sensor in sensors:
            if sensor.needs_frame_input:
                self._frame_sensors.append(sensor)
            else:
                self._extra_sensors.append(sensor)
            sensor.start()

    def run(self):
        """ start running the model until no sensor input arrives
        """
        while self._finished.value == 0:
            model_updated = False
            sensor_outputs = []
            if not self._intput_queue.empty():
                self.logger.info('getting a frame img path from an input queue')
                frame_img_path = self._intput_queue.get_nowait()
                frame_sensor_item = TenguSensorItem(self._t, frame_img_path)
                # feed first
                for frame_sensor in self._frame_sensors:
                    try:
                        frame_sensor.input_queue.put_nowait(frame_sensor_item)
                    except:
                        self.logger.info('failed to put a new frame in {}'.format(frame_sensor))

                # get sensor outputs, and associate it to a tracklet if available
                # extra sensor first
                for extra_sensor in self._extra_sensors:
                    # get a sensor output
                    sensor_output = self.get_sensor_output(extra_sensor)
                    if sensor_output is None:
                        continue

                    sensor_outputs.append(sensor_output)

                # frame fist
                for frame_sensor in self._frame_sensors:
                    # get a sensor input
                    sensor_output = self.get_sensor_output(frame_sensor)
                    if sensor_output is None:
                        continue

                    sensor_outputs.append(sensor_output)

                # update the model
                self.update_model(sensor_outputs)

                model_updated = True

            if not model_updated:
                self.logger.info('no frame img is avaialble in an input queue, sleeping, finished? {}'.format(self._finished.value == 1))
                time.sleep(0.001)

        self.logger.info('exitted loop {}'.format(self._finished.value))

    def get_sensor_output(self, sensor):
        """ get a sensor output given at a paricular time
        """
        self.logger.info('getting a sensor input from {}'.format(sensor))
        sensor_output = None

        while sensor_output is None and self._finished.value == 0:
            if not sensor.output_queue.empty():
                # get one by one until a sensor output taken at the time is taken
                sensor_item = sensor.output_queue.get_nowait()
                self.logger.info('got an item {}'.format(sensor_item))
                if sensor_item.t == self._t:
                    self.logger.info('found the matching item {} with time {}'.format(sensor_item.item, self._t))
                    sensor_output = sensor_item
                else:
                    self.logger.info('this is not the one looked for, this is time {}'.format(self._t))
            else:
                self.logger.debug('not yet sensor item of {} is available, sleeping'.format(sensor))
                # wait a bit
                time.sleep(0.001)

        return sensor_output

    def update_model(self, sensor_outputs):
        
        # TODO: how to update various outputs??
        detection_dict = sensor_outputs[0].item
        detections = detection_dict['d']
        class_names = detection_dict['n']
        h = detection_dict['h']
        w = detection_dict['w']

        # update model
        tracklets, scene = self._flow_analyzer.update_model((h, w), detections, class_names)

        # then, increment by one
        self._t += 1

    def finish(self):
        self._finished.value = 1

        # at first, cleanup all the items in queues
        self.logger.info('cleaning up scene model input queue')
        while not self._intput_queue.empty():
            self._intput_queue.get_nowait()

        # finish all
        self.logger.info('finishing frame sensors')
        for frame_sensor in self._frame_sensors:
            frame_sensor.finish()
        self.logger.info('finishing extra sensors')
        for extra_sensor in self._extra_sensors:
            extra_sensor.finish()

        self.logger.info('cleaning up scene model output queue')
        while not self._output_queue.empty():
            self._output_queue.get_nowait()

        # join sensors
        self.logger.info('joinning frame sensors')
        for frame_sensor in self._frame_sensors:
            frame_sensor.join()
        self.logger.info('joinning extra sensors')
        for extra_sensor in self._extra_sensors:
            extra_sensor.join()