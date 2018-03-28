#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, time
import cv2
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

    # angle movement when not enough records exist
    angle_movement_not_available = 9.9

    def __init__(self, tracker, **kwargs):
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
        # used by Clustered KLT Tracker
        self.tracker = tracker
        self._w_hist = []
        self._h_hist = []
        self._hist = None
        self._centers = []
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
        return Tracklet.center_from_rect(self._rect)

    @staticmethod
    def center_from_rect(rect):
        """ calculate the center of this tracklet
        the true location of this object is lower than the simple center of this rectangle
        center = (int(self._rect[0]+self._rect[2]/2), int(self._rect[1]+self._rect[3]/2))
        location is a bit lower than the center y by 1/4 of the hight
        """
        return (int(rect[0]+rect[2]/2), int(rect[1]+rect[3]/2+rect[3]/4))

    @property
    def confidence(self):
        return self._confidence

    @property
    def last_updated_at(self):
        return self._last_updated_at

    @property
    def uptodate(self):
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
        1. rect similarity(position, size similrity)
        2. histogram similarity
        """


        if self._rect is None:
            # this means this is called for the first time
            assignment.hist, assignment.img = self.histogram(assignment.detection)
            return 1.0

        # 1. rect similarity
        rect_similarity = TenguTracker.calculate_overlap_ratio(self._rect, assignment.detection)

        # 2. histogram similarity
        hist0 = self._hist
        hist1, assignment.img = self.histogram(assignment.detection)
        assignment.hist = hist1
        hist_similarity = cv2.compareHist(hist0, hist1, cv2.HISTCMP_CORREL)

        disable_similarity = False
        if disable_similarity:
            rect_similarity = 1.0
            hist_similarity = 1.0

        similarity = [rect_similarity, hist_similarity]
        
        self.logger.debug('similarity = {}'.format(similarity))

        return min(similarity)

    def histogram(self, rect):
        frame = self.tracker._tengu_flow_analyzer._last_frame
        bigger_ratio = 0.0
        from_y = int(rect[1])
        from_y = max(0, int(from_y - rect[3] * bigger_ratio))
        to_y = int(rect[1]+rect[3])
        to_y = min(frame.shape[0], int(to_y + rect[3] * bigger_ratio))
        from_x = int(rect[0])
        from_x = max(0, int(from_x - rect[2] * bigger_ratio))
        to_x = int(rect[0]+rect[2])
        to_x = min(frame.shape[1], int(to_x + rect[2] * bigger_ratio))
        img = frame[from_y:to_y, from_x:to_x, :]
        hist = cv2.calcHist([img], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        flattened = hist.flatten()
        return flattened, img.copy()

    def update_properties(self):
        assignment = self._assignments[-1]
        self._movement = self.last_movement()
        new_confidence = self.similarity(assignment)
        if new_confidence < self._confidence:
            # instead of directly replacing the value, use decay
            self._confidence = self._confidence * Tracklet._estimation_decay
        else:
            self._confidence = new_confidence
        # hist is calculated at similarity
        self._hist = assignment.hist
        self._w_hist.append(assignment.detection[2])
        if len(self._w_hist) > 10:
            del self._w_hist[0]
        self._h_hist.append(assignment.detection[3])
        if len(self._h_hist) > 10:
            del self._h_hist[0]
        # rect is temporarily set
        self._rect = assignment.detection
        self._centers.append(self.center)
        # update
        self._last_updated_at = TenguTracker._global_updates

    def update_with_assignment(self, assignment, class_name):
        if len(self._assignments) > 0:
            self.logger.debug('{}@{}: updating with {} from {} at {}'.format(id(self), self.obj_id, assignment, self._assignments[-1], self._last_updated_at))

        if len(self._assignments) > 0 and self.similarity(assignment) < Tracklet._min_confidence:
            # do not accept this
            pass
        elif not self.accept_measurement(Tracklet.center_from_rect((assignment.detection))):
            # not acceptable
            self.logger.debug('{} is not an acceptable measurement to update {}'.format(assignment.detection, self))
        elif self.has_left:
            # no more update
            self.update_without_assignment()
        else:
            # update
            self._assignments.append(assignment)
            self.update_properties()
            self.recent_updates_by('1')
            if not self._class_map.has_key(class_name):
                self._class_map[class_name] = 0
            self._class_map[class_name] += 1
            self.update_location(self._centers[-1])

    def update_without_assignment(self):
        """
        no update was available
        """

        self.logger.debug('updating without {}@{}'.format(id(self), self.obj_id))

        self.recent_updates_by('2')
        self.update_location(None)

    def update_location(self, z):
        super(Tracklet, self).update_location(z)

        # update rect
        location = self.location
        w = sum(self._w_hist) / len(self._w_hist)
        h = sum(self._h_hist) / len(self._h_hist)
        x = location[0] - w/2
        y = location[1] - h*3/4
        self._rect = (x, y, w, h)

        # update confidence based on variance
        variance = self.variance
        if variance is None:
            return
        confidence_x = min(1.0, w/2 / (variance[0]*3))
        confidence_y = min(1.0, h/2 / (variance[1]*3))
        self._confidence = min(confidence_x, confidence_y)

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

    @property
    def angle_movement(self):
        """ returns the movement of angles
        """
        if len(self._milestones) < 3:
            return Tracklet.angle_movement_not_available

        first_angle = Tracklet.get_angle(self._milestones[0][0], self._milestones[2][0])
        last_angle = Tracklet.get_angle(self._milestones[-3][0], self._milestones[-1][0])

        diff_angle = last_angle - first_angle

        if diff_angle < -1 * math.pi:
            diff_angle = 2 * math.pi + diff_angle
        elif diff_angle > math.pi:
            diff_angle = diff_angle - 2 * math.pi

        return diff_angle

    @staticmethod
    def get_angle(p_from, p_to):
        diff_x = p_to[0] - p_from[0]
        diff_y = p_to[1] - p_from[1]
        # angle = (-pi, pi)
        angle = math.atan2(diff_y, diff_x)
        return angle

    def mark_flow_passed(self, flow):
        self._passed_flow = flow

    @property
    def passed_flow(self):
        return self._passed_flow

    def cropped_images(self, max_size):
        """get intermediate images
        """
        cropped_images = []
        use_intermediate = True
        if use_intermediate:
            diff = len(self._assignments) - max_size
            offset = 0
            if diff > 0:
                # take intermediate images
                offset = diff / 2
            index = 1
            while index <= max_size and (index+offset) <= len(self._assignments):
                assignment = self._assignments[-1*(index+offset)]
                if hasattr(assignment, 'img') and assignment.img is not None:
                    cropped_images.append(assignment.img)
                index += 1
        else:
            # use from the first
            index = 0
            while len(cropped_images) < len(self._assignments) and len(cropped_images) < max_size:
                assignment = self._assignments[index]
                if hasattr(assignment, 'img') and assignment.img is not None:
                    cropped_images.append(assignment.img)
                index += 1

        return cropped_images

class Assignment(object):

    def __init__(self, detection):
        super(Assignment, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.detection = detection
        self.hist = None

    def __repr__(self):
        return 'detection={}'.format(self.detection)

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

    def __init__(self, obsoletion=100, ignore_direction_ranges=None, R_std=10., P=25., Q=0.01, **kwargs):
        super(TenguTracker, self).__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self._tracklets = []

        self._obsoletion = obsoletion
        # (start, end], -pi <= ignored_range < pi
        self._ignore_direction_ranges = ignore_direction_ranges
        self._R_std = R_std
        self._P = P
        self._Q = Q

        self._tengu_flow_analyzer = None
        self._min_length = None

    def set_flow_analyzer(self, tengu_flow_analyzer, min_length):
        self._tengu_flow_analyzer = tengu_flow_analyzer
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
        # TODO: R_std and P can be set from saved values in a flow graph
        to = Tracklet(self, R_std=self._R_std, P=self._P, Q=self._Q)
        to.update_with_assignment(Assignment(assignment), class_name)
        return to

    def initialize_tracklets(self, detections, class_names):
        
        self.prepare_updates(detections)

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
            if not tracklet.uptodate:
                self.assign_new_to_tracklet(None, None, tracklet)

    def assign_new_to_tracklet(self, new_assignment, class_name, tracklet):
        if new_assignment is None:
            tracklet.update_without_assignment()
        else:
            tracklet.update_with_assignment(Assignment(new_assignment), class_name)

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
                self.logger.info('{} became obsolete'.format(tracklet))
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

        # obsolete if not is_confirmed
        new_tracklet = []
        for tracklet in self._tracklets:
            if tracklet.is_confirmed:
                new_tracklet.append(tracklet)
            else:
                self.logger.debug('{} is marked as obsolete, not confirmed anymore'.format(tracklet))
        removed = len(self._tracklets) - len(new_tracklet)
        self._tracklets = new_tracklet
        self.logger.debug('removed {} tracked objects due to obsoletion'.format(removed))

        for tracklet in self._tracklets:
            # check if it has already left
            if tracklet.has_left:
                continue
            # check if any of rect corners has left
            frame_shape = self._tengu_flow_analyzer._frame_shape
            has_left = tracklet.rect[0] <= 0 or tracklet.rect[1] <= 0 \
                    or (tracklet.rect[0] + tracklet.rect[2] >= frame_shape[1]) \
                    or (tracklet.rect[1] + tracklet.rect[3] >= frame_shape[0])
            if has_left:
                tracklet.mark_left()
                continue

    def is_obsolete(self, tracklet):
        diff = TenguTracker._global_updates - tracklet.last_updated_at
        #if tracklet.speed < 0:
        #    # quickly remove unreliable tracklet
        #    return diff > 0

        return diff > self._obsoletion

    def ignore_tracklet(self, tracklet):
        """ this is called from Flow Analyzer to check if this tracklet should be considered
        """
        ignore_tracklet = False
        if self._ignore_direction_ranges is not None:
            if tracklet.direction is None:
                self.logger.debug('{} has no direction yet, will be ignored'.format(tracklet))
                ignore_tracklet = True
            else:
                for ignore_direction_range in self._ignore_direction_ranges:
                    if tracklet.direction >= ignore_direction_range[0] and tracklet.direction < ignore_direction_range[1]:
                        self.logger.debug('{} is moving towards the direction between ignored ranges'.format(tracklet))
                        ignore_tracklet = True
                        break
        return ignore_tracklet