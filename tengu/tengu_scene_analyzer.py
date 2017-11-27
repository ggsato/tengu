#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging, math
import cv2
import numpy as np

class TenguSceneAnalyzer(object):

    def __init__(self, **kwargs):
        super(TenguSceneAnalyzer, self).__init__(**kwargs)

    def analyze_scene(self, scene):
        """ analyze scene, and outputs a count report at intervals
        """

        for name in scene.flow_names:
            for named_flow in scene.named_flows(name):
                pass