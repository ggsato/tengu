#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import logging, copy
from scipy.optimize import linear_sum_assignment

"""The default implementation of TenguTracker
The default implementation of TenguTracker is based on o¥overlaps.
If the currently tracked object's rectangle overlaps over a threshold is considered identical.
"""

class TrackedObject(object):

    obj_id = -1
    min_confirmation_updates = 1

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        TrackedObject.obj_id += 1
        # incremental unique object id
        self._obj_id = TrackedObject.obj_id
        self._last_updated_at = TenguTracker._global_updates
        self._assignments = []
    
    # tracklet properties
    @property
    def position(self):
        pass

    @property
    def size(self):
        pass

    @property
    def velocity(self):
        pass

    @property
    def rect(self):
        #self.logger.info('retruning rect from {}'.format(self))
        return self._assignments[-1]

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def is_confirmed(self):
        return len(self._assignments) > TrackedObject.min_confirmation_updates

    @property
    def last_assignment(self):
        return self._assignments[-1]

    def update_with_assignment(self, assignment):
        self._assignments.append(assignment)
        self._last_updates = TenguTracker._global_updates
        self.logger.debug('{}: updating track with {} at {}'.format(self, assignment, self._last_updated_at))

class TenguCostMatrix(object):

	def __init__(self, assignments, cost_matrix):
		super(TenguCostMatrix, self).__init__()
		self.assignments = assignments
		self.cost_matrix = cost_matrix

class TenguTracker(object):

    _global_updates = -1
    _min_value = 0.0001

    def __init__(self, obsoletion=100):
        self.logger = logging.getLogger(__name__)
        self._tracked_objects = []
        self._obsoletion = obsoletion

    @property
    def tracked_objects(self):
        return copy.copy(self._tracked_objects)

    def resolve_trackings(self, detections):
        TenguTracker._global_updates += 1

        if len(self._tracked_objects) == 0:
            self.initialize_tracked_objects(detections)
            return self.tracked_objects

        if len(detections) == 0:
        	return self.tracked_objects

        self.prepare_updates(detections)

        tengu_cost_matrix = self.calculate_cost_matrix(detections)
        row_ind, col_ind = TenguTracker.optimize_and_assign(tengu_cost_matrix.cost_matrix)
        min_cost = tengu_cost_matrix.cost_matrix[row_ind, col_ind].sum()
        self.logger.debug('min optimzied cost of {} = {}'.format(tengu_cost_matrix.cost_matrix, min_cost))
        self.update_trackings_with_optimized_assignments(tengu_cost_matrix.assignments, row_ind, col_ind)

        # TODO: Create new trackings
        # Obsolete old ones
        self.obsolete_trackings()

        self.logger.debug('resolved, and now {} tracked objects'.format(len(self._tracked_objects)))

        return self.tracked_objects

    def prepare_updates(self, detections):
        pass

    def new_tracked_object(self, detection):
        to = TrackedObject()
        to.update_with_assignment(detection)
        return to

    def initialize_tracked_objects(self, detections):
        for detection in detections:
            self._tracked_objects.append(self.new_tracked_object(detection))

    def calculate_cost_matrix(self, detections):
    	""" Calculates cost matrix
        Given the number of tracked objects, m, and the number of detections, n,
        cost matrix consists of mxn matrix C.
        Cmn: a cost at (m, n), -Infinity<=Cmn<=0
		
		Then, such a cost matrix is optimized to produce a combination of minimum cost assignments.
		For more information, see Hungarian algorithm(Wikipedia), scipy.optimize.linear_sum_assignment
        """
       	cost_matrix = self.create_empty_cost_matrix(len(detections))
        for t, tracked_object in enumerate(self._tracked_objects):
            for d, detection in enumerate(detections):
                cost = self.calculate_cost_by_overlap_ratio(detection, tracked_object.rect)
                cost_matrix[t][d] = cost
        return TenguCostMatrix(detections, cost_matrix)

    def create_empty_cost_matrix(self, cols):
    	if len(self._tracked_objects) == 0:
            return None

        shape = (len(self._tracked_objects), cols)
        self.logger.debug('shape: {}'.format(shape))
        cost_matrix = np.zeros(shape, dtype=np.float32)
        if len(shape) == 1:
        	# the dimesion should be forced
        	cost_matrix = np.expand_dims(cost_matrix, axis=1)

        self.logger.debug('shape of cost_matrix: {}'.format(cost_matrix.shape))

       	return cost_matrix

    def calculate_cost_by_overlap_ratio(self, rect_a, rect_b):
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
        self.logger.debug('ratio of {} and {} = {}'.format(rect_a, rect_b, ratio))

        return -1 * math.log(max(ratio, TenguTracker._min_value))

    @staticmethod
    def optimize_and_assign(cost_matrix):
    	row_ind, col_ind = linear_sum_assignment(cost_matrix)
    	return row_ind, col_ind

    def update_trackings_with_optimized_assignments(self, assignments, row_ind, col_ind):
    	for ix, row in enumerate(row_ind):
    		self._tracked_objects[row].update_with_assignment(assignments[col_ind[ix]])

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

        return diff > self._obsoletion