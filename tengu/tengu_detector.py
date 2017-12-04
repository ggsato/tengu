#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import cv2
import numpy as np

""" Reference implementation of TenguDetector
As a reference, yet sometimes useful enough, background subtraction based detection is implemented
"""

class TenguDetector(object):

    DETECTION_CLASS_ANY = 'any'

    def __init__(self):
        self.logger= logging.getLogger(__name__)

    def class_names(self):
        return [TenguDetector.DETECTION_CLASS_ANY]
    
    def detect(self, frame):
        """ detect objects in a frame, then retrun their detections as rectangle by their class
        when there are one or more classes are supported, such detections should include all the classes with an empty array for not found classes
        """
        detections = []
        class_names = []
        return detections, class_names

class TenguBackgroundSubtractionDetector(TenguDetector):

    def __init__(self, use_gaussian=True, use_dilation=True, debug=False):
        super(TenguBackgroundSubtractionDetector, self).__init__()
        self.avg_frame = None
        self.use_gaussian = use_gaussian
        self.use_dilation = use_dilation
        self.debug = debug

    def detect(self, frame):
        start = cv2.getTickCount()

        # gray and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # reset if the shape is different
        if self.avg_frame is not None and self.avg_frame.shape != blurred.shape:
            self.avg_frame = None
        
        # create avg image
        if self.avg_frame is None:
            self.avg_frame = np.float32(blurred)

        cv2.accumulateWeighted(blurred, self.avg_frame, 0.3)

        # thresholding to make a binary image
        # threshold binary image
        abs_pos = cv2.convertScaleAbs(self.avg_frame)
        delta = cv2.absdiff(abs_pos, blurred)

        if self.use_gaussian:
            thresh = cv2.threshold(delta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)[1]
        else:
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

        if self.use_dilation:
            thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # rectangles
        detections = []
        class_names = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            detections.append((x, y, w, h))
            class_names.append(TenguDetector.DETECTION_CLASS_ANY)

        end = cv2.getTickCount()
        time = (end - start)/ cv2.getTickFrequency()
        self.logger.debug('{} detected in {} ms'.format(len(detections), time))

        if self.debug:
            # show opencv window
            for rect in detections[TenguDetector.DETECTION_CLASS_ANY]:
                cv2.rectangle(thresh, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 3)
            cv2.imshow('TenguDetector', thresh)
            cv2.waitKey(1)

        return detections, class_names