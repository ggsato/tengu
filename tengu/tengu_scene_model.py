#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, sys, traceback, time
from multiprocessing import Process, Queue, Value
from Queue import Empty, Full

from tengu import Tengu
from detectnet_detector import DetectNetDetector
from tengu_tracker import TenguTracker
from tengu_flow_analyzer import TenguFlowAnalyzer
from tengu_scene_analyzer import TenguSceneAnalyzer
from tengu_sensor import TenguSensorItem, TenguObjectDetectionSensor

class TenguSceneModel(Process):
    """ TenguSceneModel manages a prediction model, in which objects are tracked and counted

    A prediction model conssits of 4 components.

    1. frame sensor   : detects objects in a frame
    2. tracker        : associate detections with existing tracklets or create new ones, then update each tracking Kalman Filter model with assigned detections
    3. flow analyzer  : collects and analyzes tracklet movements, and find traffic flows statistically
    4. scene analyzer : count objects for each flow

    """

    def __init__(self, input_queue_max_size=10, output_queue_max_size=10, output_queue_timeout_in_secs=10, scene_file=None, min_length=10, **kwargs):
        super(TenguSceneModel, self).__init__(**kwargs)

        self.logger= logging.getLogger(__name__)

        # sensor
        self._detector = DetectNetDetector(8890, interval=Value('i', 1), max_detection_size=192)
        self._frame_sensor = TenguObjectDetectionSensor(detector=self._detector)

        # flow analyzer
        self._flow_analyzer = TenguFlowAnalyzer(scene_file=scene_file)

        # tracker
        self._tracker = TenguTracker(self._flow_analyzer, min_length)

        # scene analyzer
        self._scene_analyzer = TenguSceneAnalyzer()

        # queues
        self._intput_queue = Queue(maxsize=input_queue_max_size)
        self._output_queue = Queue(maxsize=output_queue_max_size)
        self._output_queue_timeout_in_secs = output_queue_timeout_in_secs

        # time, incremented one by delta_t
        self._t = 0

        # finish if 1, this has to be Value because it is exchanged between processes
        self._finished = Value('i', -1)

    @property
    def detection_interval(self):
        return self._detector.interval

    def set_detection_interval(self, detection_interval):
        if self._detector is None:
            return

        self._detector.set_detection_interval(detection_interval)

    @property
    def input_queue(self):
        return self._intput_queue

    @property
    def output_queue(self):
        return self._output_queue

    def start_sensor(self):
        # start sensor
        self.logger.info('starting sensor')
        self._frame_sensor.start()

        while self._frame_sensor._finished.value == -1:
            self.logger.info('waiting for sensor running, process alive = {}, exitcode = {}'.format(self._frame_sensor.is_alive(), self._frame_sensor.exitcode))
            time.sleep(0.1)

    def run(self):
        """ start running the model until no sensor input arrives
        """
        # run
        self.logger.info('running scene model')
        self._finished.value = 0
        try:
            while self._finished.value == 0:
                model_update_start = time.time()
                model_updated = False
                sensor_output = None
                frame_img_path = None
                has_changed = False
                save_path = None
                try:
                    event_dict = self._intput_queue.get_nowait()
                    frame_img_path = event_dict[Tengu.EVENT_FRAME_CROPPED]
                    has_changed = event_dict[Tengu.EVENT_CAMERA_CHANGED]
                    if event_dict.has_key(Tengu.EVENT_SCENE_SAVE):
                        save_path = event_dict[Tengu.EVENT_SCENE_SAVE]
                except Empty:
                    pass
                except:
                    info = sys.exc_info()
                    self.logger.exception('Unknown Exception {}, {}, {}'.format(info[0], info[1], info[2]))
                    traceback.print_tb(info[2])
                if has_changed:
                    self.logger.info('camera change detected, resetting...')
                    # model has to be reset
                    self._flow_analyzer.reset()
                    self._scene_analyzer.reset_counter()
                elif save_path is not None:
                    self.logger.info('saving scene to {}...'.format(save_path))
                    self._flow_analyzer.save(save_path)
                    self.logger.info('saved the scene to {}'.format(save_path))
                if frame_img_path is not None:
                    self.logger.debug('got a frame img path from an input queue')
                    frame_sensor_item = TenguSensorItem(self._t, frame_img_path)
                    # feed first
                    try:
                        self._frame_sensor.input_queue.put_nowait(frame_sensor_item)
                    except Full:
                        self.logger.debug('failed to put a new frame in frame sensor queue')

                    # get sensor output
                    sensor_output = self.get_sensor_output(self._frame_sensor)
                    if sensor_output is None:
                        continue

                    # update the model
                    detections, class_names, tracklets, flows_updated = self.update_model(sensor_output)

                    if detections is None and self._finished.value != 0:
                        # shutdown in progress
                        self.logger.info('None detections found, shutting down...')
                        break

                    counted_tracklets = self._scene_analyzer.analyze_scene(self._flow_analyzer.scene)

                    # put in an output queue
                    done = False
                    start = time.time()
                    elapsed = 0
                    tracklet_dicts = []
                    for tracklet in tracklets:
                        tracklet_dicts.append(tracklet.to_dict())
                    flow_nodes_list = []
                    for flow_node in self._flow_analyzer.scene.updated_flow_nodes:
                        flow_nodes_list.append(flow_node.serialize())
                    flows_list = []
                    if flows_updated:
                        for flow in self._flow_analyzer.scene.flows:
                            flows_list.append(flow.serialize())
                    scene_dict = {'flow_nodes': flow_nodes_list, 'flows': flows_list}
                    output_dict = {'d': detections, 'c': class_names, 't': tracklet_dicts, 'n': self._t, 's': scene_dict, 'ct': counted_tracklets}
                    while not done and elapsed < self._output_queue_timeout_in_secs and self._finished.value == 0:
                        try:
                            self.logger.debug('putting output dict {} to an output queue'.format(output_dict))
                            self._output_queue.put_nowait(output_dict)
                            done = True
                        except Full:
                            self.logger.debug('failed to put in the output queue, sleeping')
                            time.sleep(0.001)
                        elapsed = time.time() - start

                    model_updated = True

                    self.logger.debug('model update at time {} took {} s with {} detections'.format(self._t, (time.time() - model_update_start), len(detections)))

                if not model_updated:
                    self.logger.debug('no frame img is avaialble in an input queue, queue size = {}, finished? {}'.format(self._intput_queue.qsize(), self._finished.value > 0))
                    time.sleep(0.001)
                else:
                    # then, increment by one
                    self._t += 1
        except:
            info = sys.exc_info()
            self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:
            self._scene_analyzer.finish_analysis()
            self._finished.value = 2
            if self._finished.value == 0:
                # finish is not called
                self.finish()
            self.logger.info('exitted scene model loop {}'.format(self._finished.value))

    def get_sensor_output(self, sensor):
        """ get a sensor output given at a paricular time
        """
        self.logger.debug('getting a sensor input from {}'.format(sensor))
        sensor_output = None
        start = time.time()

        while sensor_output is None and self._finished.value == 0:
            # get one by one until a sensor output taken at the time is taken
            sensor_item = None
            try:
                sensor_item = sensor.output_queue.get_nowait()
            except Empty:
                pass
            if sensor_item is not None:
                self.logger.debug('got an item {}'.format(sensor_item))
                if sensor_item.t == self._t:
                    self.logger.debug('found the matching item {} with time {}'.format(sensor_item.item, self._t))
                    sensor_output = sensor_item
                else:
                    self.logger.debug('this is not the one looked for, this is time {}'.format(self._t))
            else:
                self.logger.debug('not yet sensor item of {} is available, sleeping'.format(sensor))
                # wait a bit
                time.sleep(0.001)

        self.logger.debug('got an sensor output at time {} in {} s'.format(self._t, time.time() - start))

        return sensor_output

    def update_model(self, sensor_output):
        start = time.time()

        detection_dict = sensor_output.item
        detections = detection_dict['d']
        class_names = detection_dict['n']
        shape = detection_dict['s']

        self.logger.debug('detections at scene model = {}'.format(detections))

        if not self._flow_analyzer.initialized:
            self._flow_analyzer.initialize(shape)

        # update tracklets with new detections
        tracklets = self._tracker.resolve_tracklets(detections, class_names)

        # update flow graph
        flows_updated = self._flow_analyzer.update_flow_graph(tracklets)

        self.logger.debug('model was updated at time {} in {} s'.format(self._t, time.time() - start))

        return detections, class_names, tracklets, flows_updated

    def finish(self):
        if self._finished.value == -1:
            # this is not yet started
            return True

        if self._finished.value == 0:
            self._finished.value = 1

        # at first, cleanup all the items in queues
        self.logger.info('cleaning up scene model input queue')
        self._intput_queue.close()
        has_more = True
        while has_more:
            try:
                self._intput_queue.get_nowait()
            except:
                # more
                has_more = False

        # finish all
        self.logger.info('finishing frame sensor')
        if not self._frame_sensor.finish():
            self._frame_sensor.terminate()

        self.logger.info('cleaning up scene model output queue')
        self._output_queue.close()
        has_more = True
        while has_more:
            try:
                self._output_queue.get_nowait()
            except:
                # more
                has_more = False

        # wait
        start = time.time()
        elapsed = 0
        while elapsed < 1.0 and self._finished.value != 2:
            self.logger.info('waiting for exitting scene model loop, finished = {}'.format(self._finished.value))
            time.sleep(0.001)
            elapsed = time.time() - start

        return self._finished.value == 2