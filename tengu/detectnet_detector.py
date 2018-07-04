#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os, json, time
from io import open
import cv2
import requests
import random
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from multiprocessing import Value

from tengu_detector import TenguDetector

class DetectionServiceHandler():
    def __init__(self, port, alias=None):
        self.port = port
        if alias is None:
            alias = "{:0.5f}".format(random.random())
        self.alias = alias

        self.tmp_dir = 'uploads/'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def post(self, scene, index):
        #print('making a POST request for a scene detection')
        start = time.time()

        host = 'localhost'
        url = 'http://' + host + ':' + self.port + '/upload'
        # append file
        multiple_files = []
        filename = 'scene_{}_{}.jpg'.format(self.alias, index)
        path = self.tmp_dir + filename
        write_in_memory = True
        if write_in_memory:
            file_obj = BytesIO()
            rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(np.uint8(rgb))
            image.save(file_obj, "JPEG")
            # debug
            #image.save('debug.jpg', "JPEG")
            file_obj.seek(0)
        else:
            cv2.imwrite(path, scene)
            file_obj = open(path, 'rb')
        try:
            multiple_files.append(('images', (filename, file_obj)))
        except:
            logging.error('failed to create a file to post, retuns an empty detection')
            return {"c":[], "t":[], "b":[]}

        elapsed_time_0 = time.time()

        #print('making request...')
        r = requests.post(url, files=multiple_files)

        elapsed_time_1 = time.time()

        #print('received response')
        # TODO-OOH
        logging.debug(r.text)

        try:
            out = json.loads(r.text)
        except:
            out = {}
            logging.error('failed to parse a response from the server, returns an empty one safely...')

        elapsed_time_2 = time.time()

        # server reads img in memory, only client write to file
        if not write_in_memory:
            os.remove(path)

        logging.debug('detection serviced in ' + str(time.time() - start) + 'ms [' + str(elapsed_time_0-start) + ', ' + str(elapsed_time_1-elapsed_time_0) + ', ' + str(elapsed_time_2-elapsed_time_1) + ']')

        return out

class DetectNetDetector(TenguDetector):

    DETECTION_CLASS_SMALL = 'Small'
    DETECTION_CLASS_LARGE = 'Large'
    DETECTION_CLASS_BUS = 'Large(Bus)'

    def __init__(self, port, alias=None, max_detection_size=None, interval=Value('i', 1)):
        super(DetectNetDetector, self).__init__()
        self.detection_handler = DetectionServiceHandler(str(port), alias=alias)
        self._index = 0
        self._max_detection_size = max_detection_size
        self._interval = interval

    @property
    def interval(self):
        return self._interval.value

    def set_detection_interval(self, detection_interval):
        self._interval.value = detection_interval

    @staticmethod
    def class_names():
        return [DetectNetDetector.DETECTION_CLASS_SMALL, DetectNetDetector.DETECTION_CLASS_LARGE, DetectNetDetector.DETECTION_CLASS_BUS]

    def detect(self, frame):
        h = 480
        w = 640
        rects = []
        class_names = []
        start = time.time()

        # detect if ON
        if self._index % self._interval.value == 0:
            #print('index, interval = {}, {}'.format(self._index, self._interval.value))

            # crop frame if larger than shape
            offset_y = max(0, int((frame.shape[0] - h) / 2))
            offset_x = max(0, int((frame.shape[1] - w) / 2))
            cropped = frame[offset_y:offset_y+h, offset_x:offset_x+w, :]

            # if smaller, pad
            if cropped.shape[0] < h or cropped.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype=cropped.dtype)
                padded[0:cropped.shape[0], 0:cropped.shape[1], :] = cropped
                cropped = padded

            vehicles = self.detection_handler.post(cropped, self._index)
            for class_name in vehicles.keys():
                if class_name == 'c':
                    name = DetectNetDetector.DETECTION_CLASS_SMALL
                elif class_name == 't':
                    name = DetectNetDetector.DETECTION_CLASS_LARGE
                elif class_name == 'b':
                    name = DetectNetDetector.DETECTION_CLASS_BUS
                else:
                    logging.error('no such class {}'.format(class_name))
                    name = 'Unknown'
                for car in vehicles[class_name]:
                    if self._max_detection_size is not None:
                        if car[2] > self._max_detection_size or car[3] > self._max_detection_size:
                            continue
                    car[0] += offset_x
                    car[1] += offset_y
                    rects.append([car[0], car[1], car[2], car[3]])

                    class_names.append(name)

        self._index += 1

        #print('detected in {} ms'.format(time.time() - start))

        return rects, class_names