#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, time
import numpy as np
import logging, copy
from scipy.optimize import linear_sum_assignment

from tengu_object import TenguObject

"""The default implementation of TenguTracker
The default implementation of TenguTracker is based on o¥overlaps.
If the currently tracked object's rectangle overlaps over a threshold is considered identical.
"""

class Tracklet(TenguObject):

    _class_obj_id = -1
    _min_confidence = 0.5
    _estimation_decay = 0.9
    _recent_updates_length = 2

    # see speed for details
    average_real_size = 4.5

    def __init__(self, **kwargs):
        super(Tracklet, self).__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        Tracklet._class_obj_id += 1
        # incremental unique object id
        self._obj_id = Tracklet._class_obj_id
        self._last_updated_at = -1
        self._assignments = []
        self._rect = None
        self._movement = None
        self._confidence = 0.
        self._recent_updates = ['N/A']
        # used by flow analyzer for building flow graph
        self._path = []
        # (center, rect, frame_no)
        self._milestones = []
        # used by flow analyzer for updating a scene
        self._current_flow = None
        self._dist_to_sink = 0
        self._flow_similarity = 0
        self._removed = False
        self._left = False
        self._passed_flow = None
        # for classification
        self._class_map = {}

        # debug
        self._shortest_path_for_debug = None

    def __repr__(self):
        return 'id:{}, obj_id:{}, confidence:{}, speed:{}, flow:{}, left:{}, class:{}'.format(id(self), self._obj_id, self._confidence, self.speed, self.current_flow_name, self._left, self.class_name)

    # Tracklet Properties
    @property
    def obj_id(self):
        return self._obj_id
    
    @property
    def movement(self):
        return self._movement

    @property
    def rect(self):
        return self._rect

    @property
    def center(self):
        """ calculate the center of this tracklet
        the true location of this object is lower than the simple center of this rectangle
        center = (int(self._rect[0]+self._rect[2]/2), int(self._rect[1]+self._rect[3]/2))
        location is a bit lower than the center y by 1/4 of the hight

        this is used to locate a flow node
        """
        return (int(self._rect[0]+self._rect[2]/2), int(self._rect[1]+self._rect[3]/2+self._rect[3]/4))

    @property
    def confidence(self):
        return self._confidence

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def updated(self):
        return self._last_updated_at == TenguTracker._global_updates

    @property
    def is_confirmed(self):
        return self._confidence > Tracklet._min_confidence

    @property
    def last_assignment(self):
        return self._assignments[-1]

    @property
    def last_update_pattern(self):
        return self._recent_updates[-1]

    def similarity(self, assignment):
        """
        calculate similarity of assignment to self
        """
        if self._rect is None:
            return 0

        return TenguTracker.calculate_overlap_ratio(self._rect, assignment)

    def update_properties(self, lost=True):
        assignment = self._assignments[-1]
        self._movement = self.last_movement()
        new_confidence = self.similarity(assignment)
        if new_confidence < self._confidence:
            # instead of directly replacing the value, use decay
            self._confidence = self._confidence * Tracklet._estimation_decay
        else:
            self._confidence = new_confidence
        # rect has to be updated after similarity calculation
        self._rect = assignment
        if not lost:
            self._last_updated_at = TenguTracker._global_updates

    def update_with_assignment(self, assignment, class_name):
        """
        update tracklet with the new assignment
        note that this could be called multiple times in case multiple cost matrixes are used
        """
        if len(self._assignments) > 0:
            self.logger.debug('{}@{}: updating with {} from {} at {}'.format(id(self), self.obj_id, assignment, self._assignments[-1], self._last_updated_at))
        
        if self._last_updated_at == TenguTracker._global_updates:
            # nothing to do here
            pass

        if len(self._assignments) > 0 and self.similarity(assignment) < Tracklet._min_confidence:
            # do not accept this
            self.update_without_assignment()
        else:
            self._assignments.append(assignment)
            self.update_properties(lost=False)
            self.recent_updates_by('1')
            if not self._class_map.has_key(class_name):
                self._class_map[class_name] = 0
            self._class_map[_class_map] += 1

    def update_without_assignment(self):
        """
        no update was available
        so create a possible assignment to update
        """
        last_movement = self.last_movement()
        if last_movement is None:
            return
        
        new_x = self._rect[0] + last_movement[0] * Tracklet._estimation_decay
        new_y = self._rect[1] + last_movement[1] * Tracklet._estimation_decay
        possible_assignment = (new_x, new_y, self._rect[2], self._rect[3])
        self._assignments.append(possible_assignment)
        self.update_properties()
        self.recent_updates_by('2')

    def last_movement(self):
        return super(Tracklet, self).movement

    def recent_updates_by(self, update_pattern):
        self._recent_updates.append(update_pattern)
        if len(self._recent_updates) > Tracklet._recent_updates_length:
            del self._recent_updates[0]

    @staticmethod
    def movement_from_rects(prev, prev2):
        move_x = int((prev[0]+prev[2]/2) - (prev2[0]+prev2[2]/2))
        move_y = int((prev[1]+prev[3]/2) - (prev2[1]+prev2[3]/2))
        return move_x, move_y

    @property
    def current_flow_name(self):
        if self._current_flow is None:
            return '-'
        return self._current_flow.name

    @property
    def path(self):
        return self._path

    @property
    def source(self):
        if len(self._path) < 1:
            return None
        return self._path[0]

    @property
    def sink(self):
        if len(self._path) < 1:
            return None
        return self._path[-1]

    def add_flow_node_to_path(self, flow_node):
        self._path.append(flow_node)
        self._milestones.append([self.center, self.rect, TenguTracker._global_updates])

    @property
    def speed(self):
        """calculates speed of this tracklet based on milestones using Tracklet.average_real_size
        """

        if len(self._milestones) < 2:
            return -1

        prev_stone = None
        observations = []
        for milestone in self._milestones:
            if prev_stone is None:
                prev_stone = milestone
                continue
            distance = Tracklet.compute_distance(prev_stone[0], milestone[0])
            real_size_per_pixel = Tracklet.average_real_size / max(milestone[1][2], milestone[1][3])
            real_distance = distance * real_size_per_pixel
            real_distance_per_frame = real_distance / (milestone[2] - prev_stone[2])
            self.logger.debug('observation of speed distance={}, real_size_per_pixel={}, real_distance={}, real_distance_per_frame={}'.format(distance, real_size_per_pixel, real_distance, real_distance_per_frame))
            observations.append(real_distance_per_frame)
            prev_stone = milestone

        # finally, from the last to the current location
        if prev_stone[2] < TenguTracker._global_updates:
            distance = Tracklet.compute_distance(prev_stone[0], self.center)
            real_size_per_pixel = Tracklet.average_real_size / max(self.rect[2], self.rect[3])
            real_distance = distance * real_size_per_pixel
            real_distance_per_frame = real_distance / (TenguTracker._global_updates - prev_stone[2])
            self.logger.debug('current observation of speed distance={}, real_size_per_pixel={}, real_distance={}, real_distance_per_frame={}'.format(distance, real_size_per_pixel, real_distance, real_distance_per_frame))
            observations.append(real_distance_per_frame)

        estimated_speed = float(sum(observations)) / len(observations)
        self.logger.debug('estimated speed per frame = {}, observations = {}'.format(estimated_speed, observations))

        return estimated_speed

    @staticmethod
    def compute_distance(pos0, pos1):
        return math.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)

    @property
    def last_flow(self):
        if len(self._path) < 1:
            return None
        return self._path[-1]

    @property
    def dist_to_sink(self):
        if self._dist_to_sink is None:
            return '-'
        return self._dist_to_sink

    @property
    def flow_similarity(self):
        return self._flow_similarity

    def set_flow(self, flow, distance_to_sink, similarity, shortest_path_for_debug=None):
        self._current_flow = flow
        self._dist_to_sink = distance_to_sink
        self._flow_similarity = similarity
        self._shortest_path_for_debug = shortest_path_for_debug

    @property
    def removed(self):
        return self._removed

    def mark_removed(self):
        self._removed = True

    @property
    def has_left(self):
        return self._left

    def mark_left(self):
        self._left = True

    @property
    def class_name(self):
        return sorted(self._class_map.keys(), key=self._class_map.__getitem__, reverse=True)[0]

    @property
    def milestones(self):
        return self._milestones

    def mark_flow_passed(self, flow):
        self._passed_flow = flow

    @property
    def passed_flow(self):
        return self._passed_flow

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
    # this is the -1 * log value
    # 0.5 = 0.7 => 50% overlap
    # 0.2 = 1.61
    # 0.1 = 2.3
    # 0.01 = 4.6 => 1% overlap
    _confident_min_cost = 4.6

    def __init__(self, obsoletion=100):
        self.logger = logging.getLogger(__name__)
        self._tracklets = []
        self._obsoletion = obsoletion
        self._tengu_flow_analyer = None
        self._min_length = None

    def set_flow_analyzer(self, tengu_flow_analyzer, min_length):
        self._tengu_flow_analyer = tengu_flow_analyzer
        self._min_length = min_length

    @property
    def tracklets(self):
        return copy.copy(self._tracklets)

    def resolve_tracklets(self, detections, class_names):
        """ update existing tracklets with new detections
        This is the main task of Tracker, which consists of the following jobs:
        1. tracklet initialization and any other preparations required to complete this task
        2. calculate cost matrix of existing tracklets and new detections
        3. create a plan for new detections assignments
        4. update existing tracklets with the assignment plan

        Also, Tracklet does the following book keeping jobs of tracklets
        5. update existing tracklets that had no assignment
        6. find obsolete tracklets among existing tracklets, and clean them up if any
        """
        TenguTracker._global_updates += 1

        if len(self._tracklets) == 0:
            self.initialize_tracklets(detections, class_names)
            return self.tracklets

        start = time.time()

        if len(detections) > 0:

            self.prepare_updates(detections)
            lap1 = time.time()
            self.logger.debug('prepare_updates took {} s'.format(lap1 - start))

            tengu_cost_matrix = self.calculate_cost_matrix(detections)
            lap2 = time.time()
            self.logger.debug('calculate_cost_matrix took {} s'.format(lap2 - lap1))

            TenguTracker.optimize_and_assign(tengu_cost_matrix)
            lap3 = time.time()
            self.logger.debug('optimize_and_assign took {} s'.format(lap3 - lap2))

            self.update_trackings_with_optimized_assignments(tengu_cost_matrix, class_names)
            lap4 = time.time()
            self.logger.debug('update_trackings_with_optimized_assignments took {} s'.format(lap4 - lap3))

        # update without detections
        self.update_tracklets()

        # Obsolete old ones
        self.obsolete_trackings()
        lap5 = time.time()
        self.logger.debug('obsolete_trackings took {} s'.format(lap5 - lap4 if len(detections) > 0 else start))

        end = time.time()
        self.logger.debug('resolved, and now {} tracked objects at {}, executed in {} s'.format(len(self._tracklets), TenguTracker._global_updates, end-start))

        return self.tracklets

    def prepare_updates(self, detections):
        pass

    def new_tracklet(self, assignment, class_name):
        to = Tracklet()
        to.update_with_assignment(assignment, class_name)
        return to

    def initialize_tracklets(self, detections, class_names):
        for d, detection in enumerate(detections):
            self._tracklets.append(self.new_tracklet(detection, class_names[d]))

    def calculate_cost_matrix(self, detections):
        """ Calculates cost matrix
        Given the number of tracked objects, m, and the number of detections, n,
        cost matrix consists of mxn matrix C.
        Cmn: a cost at (m, n), 0<=Cmn<=Infinity
        
        Then, such a cost matrix is optimized to produce a combination of minimum cost assignments.
        For more information, see Hungarian algorithm(Wikipedia), scipy.optimize.linear_sum_assignment
        """
        cost_matrix = TenguTracker.create_empty_cost_matrix(len(self._tracklets), len(detections))
        for t, tracklet in enumerate(self._tracklets):
            if tracklet.has_left:
                continue
            for d, detection in enumerate(detections):
                cost = self.calculate_cost_by_overlap_ratio(tracklet.rect, detection)
                cost_matrix[t][d] = cost
                self.logger.debug('cost at [{}][{}] of ({}, {}) = {}'.format(t, d, id(tracklet), id(detection), cost))
        return TenguCostMatrix(detections, cost_matrix)

    @staticmethod
    def create_empty_cost_matrix(rows, cols):
        if rows == 0:
            return None

        shape = (rows, cols)
        cost_matrix = np.zeros(shape, dtype=np.float32)
        if len(shape) == 1:
            # the dimesion should be forced
            cost_matrix = np.expand_dims(cost_matrix, axis=1)
        return cost_matrix

    def calculate_cost_by_overlap_ratio(self, rect_a, rect_b):
        """ Calculates a overlap ratio against rect_b with respect to rect_a
        """
        ratio = TenguTracker.calculate_overlap_ratio(rect_a, rect_b)

        return -1 * math.log(max(ratio, TenguTracker._min_value))

    @staticmethod
    def calculate_overlap_ratio(rect_a, rect_b):
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
        return ratio

    @staticmethod
    def optimize_and_assign(tengu_cost_matrix):
        row_ind, col_ind = linear_sum_assignment(tengu_cost_matrix.cost_matrix)
        tengu_cost_matrix.ind = (row_ind, col_ind)

    def update_trackings_with_optimized_assignments(self, tengu_cost_matrix, class_names):
        self.logger.debug('updating trackings based on cost matrix, ind ={}'.format(tengu_cost_matrix.ind))
        assigned = []
        for ix, row in enumerate(tengu_cost_matrix.ind[0]):
            cost = tengu_cost_matrix.cost_matrix[row][tengu_cost_matrix.ind[1][ix]]
            tracklet = self._tracklets[row]
            new_assignment = tengu_cost_matrix.assignments[tengu_cost_matrix.ind[1][ix]]
            if cost > TenguTracker._confident_min_cost:
                self.logger.debug('{} is not updated due to too high cost {}'.format(tracklet.obj_id, cost))
                continue
            assigned.append(new_assignment)
            self.logger.debug('updating tracked object {} of id={} having {} with {} at {}'.format(id(tracklet), tracklet.obj_id, tracklet.rect, new_assignment, TenguTracker._global_updates))
            self.assign_new_to_tracklet(new_assignment, class_names[tengu_cost_matrix.assignments.index(new_assignment)], tracklet)

        # create new ones
        for ix, assignment in enumerate(tengu_cost_matrix.assignments):
            if not assignment in assigned:
                # this is not assigned, create a new one
                # but do not create one contains existing tracklets
                to = self.new_tracklet(assignment, class_names[ix])
                contains = False
                for tracklet in self._tracklets:
                    if self.rect_contains(to.rect, tracklet.rect):
                        contains = True
                        break
                if contains:
                    self.logger.debug('skipping creating a new, but disturbing tracklet {}'.format(to))
                    continue
                self._tracklets.append(to)
                self.logger.debug('created tracked object {} of id={} at {}'.format(to, to.obj_id, TenguTracker._global_updates))

    def update_tracklets(self):
        for tracklet in self._tracklets:
            if not tracklet.updated:
                self.assign_new_to_tracklet(None, None, tracklet)

    def assign_new_to_tracklet(self, new_assignment, class_name, tracklet):
        if new_assignment is None:
            tracklet.update_without_assignment()
        else:
            tracklet.update_with_assignment(new_assignment, class_names)

    @staticmethod
    def rect_contains(rect_a, rect_b):
        x, y = rect_a[0], rect_a[1]
        contains_upper_left = (x > rect_b[0] and x < (rect_b[0]+rect_b[2])) and (y > rect_b[1] and y < (rect_b[1]+rect_b[3]))
        x, y = x + rect_a[2], y + rect_a[3]
        contains_lower_right = (x > rect_b[0] and x < (rect_b[0]+rect_b[2])) and (y > rect_b[1] and y < (rect_b[1]+rect_b[3]))
        return contains_upper_left and contains_lower_right

    def obsolete_trackings(self):
        """ Filters old trackings
        """
        new_tracklet = []
        for tracklet in self._tracklets:
            if not self.is_obsolete(tracklet):
                self.logger.debug('{} is not obsolete yet, diff = {}'.format(tracklet, TenguTracker._global_updates - tracklet.last_updated_at))
                new_tracklet.append(tracklet)
            else:
                self.logger.debug('{} became obsolete'.format(tracklet))
        removed = len(self._tracklets) - len(new_tracklet)
        self._tracklets = new_tracklet
        self.logger.debug('removed {} tracked objects due to obsoletion'.format(removed))

        # actively merge existing tracklets
        if len(self._tracklets) < 1:
            return

        duplicated = []
        for tracklet in self._tracklets:
            for another in self._tracklets:
                if tracklet == another:
                    continue
                cost = self.calculate_cost_by_overlap_ratio(tracklet.rect, another.rect)
                if cost < 0.2:
                    length = len(tracklet._assignments)
                    length_another = len(another._assignments)
                    if length >= length_another:
                        if not another in duplicated:
                            duplicated.append(another)
                    else:
                        if not tracklet in duplicated:
                            duplicated.append(tracklet)
                    break

        self.logger.debug('removing duplicates {}'.format(duplicated))
        for duplicate in duplicated:
            if duplicate in self._tracklets:
                del self._tracklets[self._tracklets.index(duplicate)]

    def is_obsolete(self, tracklet):
        diff = TenguTracker._global_updates - tracklet.last_updated_at
        if tracklet.speed < 0:
            # quickly remove unreliable tracklet
            return diff > 0

        return diff > self._obsoletion

    """ this is called from Flow Analyzer to check if this tracklet should be considered
    """
    def ignore_tracklet(self, tracklet):
        return False