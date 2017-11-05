#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging, copy

"""The default implementation of TenguTracker
The default implementation of TenguTracker is based on o¥overlaps.
If the currently tracked object's rectangle overlaps over a threshold is considered identical.
"""

class TrackedObject(object):

    obj_id = -1

    def __init__(self, rect, min_confirmation_updates=1):
        self.logger = logging.getLogger(__name__)
        TrackedObject.obj_id += 1
        # incremental unique object id
        self._obj_id = TrackedObject.obj_id
        self._overlapped_detections = [rect]
        self._last_updated_at = TenguTracker._global_updates
        self._min_confirmation_updates = min_confirmation_updates

    @property
    def rect(self):
        return self._overlapped_detections[-1]

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def is_confirmed(self):
        return len(self._overlapped_detections) > self._min_confirmation_updates

    def update_tracking(self, rect):
        self._overlapped_detections.append(rect)
        self._last_updates = TenguTracker._global_updates
        self.logger.debug('{}: updating track with {} at {}'.format(self, rect, self._last_updated_at))

class TenguTracker(object):

    _global_updates = -1

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._tracked_objects = []

    @property
    def tracked_objects(self):
        return self._tracked_objects

    def resolve_trackings(self, detections):
        TenguTracker._global_updates += 1

        # update tracked_objects

        return self._tracked_objects

class OverlapRatioTracker(TenguTracker):

    def __init__(self, min_overlap_ratio=0.3, updates_to_expire=10):
        super(OverlapRatioTracker, self).__init__()
        self._min_overlap_ratio = min_overlap_ratio
        self._updates_to_expire = updates_to_expire
    
    def resolve_trackings(self, detections):

        TenguTracker._global_updates += 1

        if len(self._tracked_objects) == 0:
            self.initialize_tracked_objects(detections)
            return copy.copy(self._tracked_objects)

        self.prepare_updates()

        matrix = self.overlap_matrix(detections)

        self.update_trackings(detections, matrix)

        self.obsolete_trackings()

        self.logger.debug('resolved, and now {} tracked objects'.format(len(self._tracked_objects)))

        return copy.copy(self._tracked_objects)

    def prepare_updates(self):
        pass

    def new_tracked_object(self, detection):
        return TrackedObject(detection)

    def initialize_tracked_objects(self, detections):
        for detection in detections:
            self._tracked_objects.append(self.new_tracked_object(detection))

    def overlap_matrix(self, detections):
        """ Calculates overlap ratio matrix
        overlap_matrix calculates a matrix of overlap ratio as DxT, where D is detections, and T is tracked objects.
        """
        if len(self._tracked_objects) == 0:
            return None

        shape = (len(detections), max(len(self._tracked_objects), 2))
        self.logger.debug('shape: {}'.format(shape))
        matrix = np.zeros(shape, dtype=np.float32)
        self.logger.debug('shape of matrix: {}'.format(matrix.shape))
        for d, detection in enumerate(detections):
            for t, tracked_object in enumerate(self._tracked_objects):
                overlap_ratio = OverlapRatioTracker.calculate_overlap_ratio(detection, tracked_object.rect)
                matrix[d][t] = overlap_ratio
        return matrix

    @staticmethod
    def calculate_overlap_ratio(rect_a, rect_b):
        """ Calculates a overlap ratio against rect_b with respect to rect_a
        """
        ratio = 0.0
        right_a = rect_a[0] + rect_a[2]
        right_b = rect_b[0] + rect_b[2]
        bottom_a = rect_a[1] + rect_a[3]
        bottom_b = rect_b[1] + rect_b[3]
        dx = min(right_a, right_b) - max(rect_a[0], rect_b[0])
        if dx <= 0:
            return ratio
        dy = min(bottom_a, bottom_b) - max(rect_a[1], rect_b[1])
        if dy <= 0:
            return ratio
        # finally
        ratio = (dx * dy) / (rect_a[2] * rect_a[3])
        return ratio

    def update_trackings(self, detections, matrix):
        """ Resolve tracked objects based on the matrix
        """
        # update based on votes
        votes = np.argmax(matrix, axis=1)
        sums = np.sum(matrix, axis=1)
        self.logger.debug('argmax:{}'.format(votes))
        for s, asum in enumerate(sums):
            if asum == 0:
                self._tracked_objects.append(self.new_tracked_object(detections[s]))
                continue
            max_overlap_ratio = matrix[s, votes[s]]
            if max_overlap_ratio < self._min_overlap_ratio:
                continue
            tracked_object = self._tracked_objects[votes[s]]
            tracked_object.update_tracking(detections[s])

    def obsolete_trackings(self):
        """ Filters old trackings
        """
        removed = 0
        for tracked_object in self._tracked_objects:
            if self.is_obsolete(tracked_object):
                del self._tracked_objects[self._tracked_objects.index(tracked_object)]
                removed += 1
        self.logger.debug('removed {} tracked objects due to obsoletion'.format(removed))

    def is_obsolete(self, tracked_object):
        diff = TenguTracker._global_updates - tracked_object.last_updated_at
        if not tracked_object.is_confirmed:
            return diff > 0

        return diff > self._updates_to_expire