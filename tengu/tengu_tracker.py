#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, time
import numpy as np
import logging, copy
from scipy.optimize import linear_sum_assignment

"""The default implementation of TenguTracker
The default implementation of TenguTracker is based on o¥overlaps.
If the currently tracked object's rectangle overlaps over a threshold is considered identical.
"""

class Tracklet(object):

    _class_obj_id = -1
    _min_confirmation_updates = 10
    _estimation_decay = 0.5
    _disable_estimation = True
    _recent_updates_length = 10

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        Tracklet._class_obj_id += 1
        # incremental unique object id
        self._obj_id = Tracklet._class_obj_id
        self._last_updated_at = -1
        self._assignments = []
        self._rect = None
        self._recent_updates = ['N/A']

    @property
    def obj_id(self):
        return self._obj_id
    
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
        return self._rect

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def updated(self):
        return self._last_updated_at == Tracklet._class_obj_id

    @property
    def is_confirmed(self):
        return len(self._assignments) > Tracklet._min_confirmation_updates

    @property
    def last_assignment(self):
        return self._assignments[-1]

    @property
    def last_update_pattern(self):
        return self._recent_updates[-1]

    def update_with_assignment(self, assignment):
        """
        update tracklet with the new assignment
        note that this could be called multiple times in case multiple cost matrixes are used
        """
        if len(self._assignments) > 0:
            self.logger.debug('{}@{}: updating with {} from {} at {}'.format(id(self), self.obj_id, assignment, self._assignments[-1], self._last_updated_at))
        
        if self._last_updated_at == TenguTracker._global_updates:
            # nothing to do here
            pass

        self._assignments.append(assignment)
        self._rect = assignment
        self._last_updated_at = TenguTracker._global_updates

        self.recent_updates_by('1')


    def update_without_assignment(self):
        """
        no update was available
        so update by an estimation if possible
        """
        if Tracklet._disable_estimation:
            return

        if len(self._assignments) == 1:
            # no speed calculation possible
            return

        prev = self._assignments[-1]
        prev2 = self._assignments[-2]

        last_move_x, last_move_y = Tracklet.movement_from_rects(prev, prev2)
        
        new_x = self._rect[0] + last_move_x * Tracklet._estimation_decay
        new_y = self._rect[1] + last_move_y * Tracklet._estimation_decay
        self._rect = (new_x, new_y, self._rect[2], self._rect[3])

        self.recent_updates_by('2')

    def recent_updates_by(self, update_pattern):
        self._recent_updates.append(update_pattern)
        if len(self._recent_updates) > Tracklet._recent_updates_length:
            del self._recent_updates[0]

    @staticmethod
    def movement_from_rects(prev, prev2):
        move_x = int((prev[0]+prev[2]/2) - (prev2[0]+prev2[2]/2))
        move_y = int((prev[1]+prev[3]/2) - (prev2[1]+prev2[3]/2))
        return move_x, move_y

class TenguCostMatrix(object):

    def __init__(self, assignments, cost_matrix):
        super(TenguCostMatrix, self).__init__()
        self.assignments = assignments
        self.cost_matrix = cost_matrix
        # row_ind, col_ind
        self.ind = None

class TenguTracker(object):

    _global_updates = -1
    _min_value = 0.0001
    _confident_min_cost = 0.3

    def __init__(self, obsoletion=50):
        self.logger = logging.getLogger(__name__)
        self._tracklets = []
        self._obsoletion = obsoletion

    @property
    def tracklets(self):
        return copy.copy(self._tracklets)

    def resolve_trackings(self, detections):
        TenguTracker._global_updates += 1

        if len(self._tracklets) == 0:
            self.initialize_tracklets(detections)
            return self.tracklets

        if len(detections) == 0:
            return self.tracklets

        start = time.time()

        self.prepare_updates(detections)
        lap1 = time.time()
        self.logger.debug('prepare_updates took {} s'.format(lap1 - start))

        tengu_cost_matrix_list = self.calculate_cost_matrix(detections)
        lap2 = time.time()
        self.logger.debug('calculate_cost_matrix took {} s'.format(lap2 - lap1))

        TenguTracker.optimize_and_assign(tengu_cost_matrix_list)
        lap3 = time.time()
        self.logger.debug('optimize_and_assign took {} s'.format(lap3 - lap2))

        self.update_trackings_with_optimized_assignments(tengu_cost_matrix_list)
        lap4 = time.time()
        self.logger.debug('update_trackings_with_optimized_assignments took {} s'.format(lap4 - lap3))

        # Obsolete old ones
        self.obsolete_trackings()
        lap5 = time.time()
        self.logger.debug('obsolete_trackings took {} s'.format(lap5 - lap4))

        end = time.time()
        self.logger.debug('resolved, and now {} tracked objects at {}, executed in {} s'.format(len(self._tracklets), TenguTracker._global_updates, end-start))

        return self.tracklets

    def prepare_updates(self, detections):
        pass

    def new_tracklet(self, assignment):
        to = Tracklet()
        to.update_with_assignment(assignment)
        return to

    def initialize_tracklets(self, detections):
        for detection in detections:
            self._tracklets.append(self.new_tracklet(detection))

    def calculate_cost_matrix(self, detections):
        """ Calculates cost matrix
        Given the number of tracked objects, m, and the number of detections, n,
        cost matrix consists of mxn matrix C.
        Cmn: a cost at (m, n), -Infinity<=Cmn<=0
        
        Then, such a cost matrix is optimized to produce a combination of minimum cost assignments.
        For more information, see Hungarian algorithm(Wikipedia), scipy.optimize.linear_sum_assignment
        """
        cost_matrix = self.create_empty_cost_matrix(len(detections))
        for t, tracklet in enumerate(self._tracklets):
            for d, detection in enumerate(detections):
                cost = self.calculate_cost_by_overlap_ratio(tracklet.rect, detection)
                cost_matrix[t][d] = cost
                self.logger.debug('cost at [{}][{}] of ({}, {}) = {}'.format(t, d, id(tracklet), id(detection), cost))
        return [TenguCostMatrix(detections, cost_matrix)]

    def create_empty_cost_matrix(self, cols):
        if len(self._tracklets) == 0:
            return None

        shape = (len(self._tracklets), cols)
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
            pass
        else:
            dy = min(bottom_a, bottom_b) - max(rect_a[1], rect_b[1])
            if dy <= 0:
                pass
            else:
                # finally
                ratio = (dx * dy) / (rect_a[2] * rect_a[3])
        self.logger.debug('ratio of {} and {} = {}'.format(rect_a, rect_b, ratio))

        return -1 * math.log(max(ratio, TenguTracker._min_value))

    @staticmethod
    def optimize_and_assign(tengu_cost_matrix_list):
        for tengu_cost_matrix in tengu_cost_matrix_list:
            row_ind, col_ind = linear_sum_assignment(tengu_cost_matrix.cost_matrix)
            tengu_cost_matrix.ind = (row_ind, col_ind)

    def update_trackings_with_optimized_assignments(self, tengu_cost_matrix_list):
        for m, tengu_cost_matrix in enumerate(tengu_cost_matrix_list):
            self.logger.debug('updating based on {} cost matrix, ind ={}'.format(m, tengu_cost_matrix.ind))
            for ix, row in enumerate(tengu_cost_matrix.ind[0]):
                cost = tengu_cost_matrix.cost_matrix[row][tengu_cost_matrix.ind[1][ix]]
                tracklet = self._tracklets[row]
                if cost > TenguTracker._confident_min_cost:
                    self.logger.debug('{} is not updated due to too high cost {}'.format(tracklet.obj_id, cost))
                    continue
                new_assignment = tengu_cost_matrix.assignments[tengu_cost_matrix.ind[1][ix]]
                self.logger.info('updating tracked object {} of id={} having {} with {} at {}'.format(id(tracklet), tracklet.obj_id, tracklet.rect, new_assignment, TenguTracker._global_updates))
                tracklet.update_with_assignment(new_assignment)

            # create new ones
            for ix, assignment in enumerate(tengu_cost_matrix.assignments):
                if not ix in tengu_cost_matrix.ind[1]:
                    # this is not assigned, create a new one
                    to = self.new_tracklet(assignment)
                    self._tracklets.append(to)
                    self.logger.debug('created tracked object {} of id={} at {}'.format(to, to.obj_id, TenguTracker._global_updates))

        for tracklet in self._tracklets:
            if not tracklet.updated:
                tracklet.update_without_assignment()

    def obsolete_trackings(self):
        """ Filters old trackings
        """
        removed = 0
        for tracklet in self._tracklets:
            if self.is_obsolete(tracklet):
                del self._tracklets[self._tracklets.index(tracklet)]
                removed += 1
        self.logger.debug('removed {} tracked objects due to obsoletion'.format(removed))

    def is_obsolete(self, tracklet):
        diff = TenguTracker._global_updates - tracklet.last_updated_at
        if not tracklet.is_confirmed:
            return diff > 0

        return diff > self._obsoletion