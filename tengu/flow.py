#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

class FlowBlock:

    def __init__(self):
        self.bins = 10
        self.bin_angle = 2 * math.pi / self.bins
        # hist ranges from 0 to 2*pi
        self.angle_hist = np.zeros(self.bins, dtype=np.uint8)
        self.flow_length = 0.
        # color
        self.b = 0
        self.g = 0
        self.r = 0
        self.last_max_angle = 0

    @staticmethod
    def color_by_angle(angle):
        # bgr calculation
        offset = 0
        # b
        offset = math.pi / 2
        b = math.fabs(angle + offset) / math.pi
        # g
        offset = math.pi / 6
        g = (math.pi - math.fabs(angle + offset)) / math.pi
        # r
        offset = -1 * math.pi / 6
        r = math.fabs(angle + offset) / math.pi

        #print('angle: ' + str(angle/math.pi) + ' = (' + str(int(b*255)) + ',' + str(int(g*255)) + ',' + str(int(r*255)) + ')')
        return b,g,r

    def update_block(self, angle, flow_length):
        # increment
        self.flow_length += flow_length
        # calculate bin, (-pi, pi) + pi => (0, 2pi)
        bin_ix = int((angle + math.pi) / self.bin_angle)
        #print('updating block with angle=' + str(angle) + ', bin_ix=' + str(bin_ix))
        self.angle_hist[bin_ix] += 1
        # update color
        max_angle = self.get_max_angle()
        b,g,r = FlowBlock.color_by_angle(max_angle)
        self.b = b
        self.g = g
        self.r = r
        self.last_max_angle = max_angle

    def get_max_angle(self):
        return self.angle_hist.argmax() * self.bin_angle - math.pi

    def size(self):
        return sum(self.angle_hist)

class SceneFlow:

    def __init__(self, scene_shape, flow_block_size=10):
        self.flows = np.ones(scene_shape, dtype=np.uint8) * 255
        # initialize other variables
        self._flow_blocks = {}
        self.flow_block_size = flow_block_size
        self.flow_block_radius = int(self.flow_block_size/2)
        self.max_recent_size = scene_shape[0] * scene_shape[1]
        self.recent_max_angle_size = [1]
        self.radius = int(scene_shape[1] / 20)
        self.diameter = self.radius*2
        self.color_compass_image = SceneFlow.create_color_compass_image(self.radius)
        self.max_super_block_size = int(flow_block_size/2)

    @staticmethod
    def create_color_compass_image(radius):
        diameter = radius * 2
        color_compass = np.zeros((diameter, diameter, 3), np.uint8)
        for degree in range(0, 180):
            theta = float(degree) / 180 * math.pi
            to_x = int(math.cos(theta) * radius)
            to_y = int(math.sin(theta) * radius)
            tr = [(0,0), (to_x, to_y)]
            b,g,r = SceneFlow.dir_color(tr)
            cv2.line(color_compass,(radius,radius),(radius+to_x, radius+to_y),(b*255,g*255,r*255),1)
            to_x = int(math.cos(-1*theta) * radius)
            to_y = int(math.sin(-1*theta) * radius)
            tr = [(0,0), (to_x, to_y)]
            b,g,r = SceneFlow.dir_color(tr)
            cv2.line(color_compass,(radius,radius),(radius+to_x, radius+to_y),(b*255,g*255,r*255),1)

        return color_compass

    @staticmethod
    def get_angle(tr):
        p_from = tr[0]
        p_to = tr[-1]
        diff_x = p_to[0] - p_from[0]
        diff_y = p_to[1] - p_from[1]
        # angle = (-pi, pi)
        angle = math.atan2(diff_y, diff_x)
        return angle

    @staticmethod
    def dir_color(tr):
        angle = SceneFlow.get_angle(tr)
        return FlowBlock.color_by_angle(angle)

    @staticmethod
    def compute_flow_length(tr):
        return math.sqrt((tr[-1][0]-tr[0][0])**2+(tr[-1][1]-tr[0][1])**2)

    def update_flow(self, tr):
        if len(tr) == 1:
            return
        flow_length = SceneFlow.compute_flow_length(tr)

        if flow_length < self.flow_block_size * 3:
            # do not update
            return

        angle = SceneFlow.get_angle(tr)
        max_angle_size = max(self.recent_max_angle_size)
        current_max_angle_size = 1
        for point in tr:
            x_blk = self.get_x_blk(point[0])
            y_blk = self.get_y_blk(point[1])
            blk = self.get_flow_block(x_blk, y_blk)
            blk.update_block(angle, flow_length)
            # update flow
            blk_img = np.zeros((self.flow_block_size,self.flow_block_size,3), dtype=np.uint8)
            color = (int(blk.b*255), int(blk.g*255), int(blk.r*255))
            cv2.rectangle(blk_img,(0,0),(self.flow_block_size,self.flow_block_size),color,-1)
            # calculate v
            use_hsv = False
            if use_hsv and blk.size() > 0:
                # find max in its neightbours
                hsv = cv2.cvtColor(blk_img, cv2.COLOR_BGR2HSV)
                avg_flow_length = blk.flow_length/blk.size()
                max_avg_flow_length = avg_flow_length
                from_x = max(0, x_blk-self.max_super_block_size)
                from_y = max(0, y_blk-self.max_super_block_size)
                to_x = min(int(self.flows.shape[1]/10), x_blk+self.max_super_block_size)
                to_y = min(int(self.flows.shape[0]/10), y_blk+self.max_super_block_size)
                for x in xrange(from_x, to_x+1):
                    for y in xrange(from_y, to_y+1):
                        tmp_blk = self.get_flow_block(x, y)
                        if tmp_blk.size() == 0:
                            continue
                        tmp_avg_flow_length = tmp_blk.flow_length/tmp_blk.size()
                        if tmp_avg_flow_length > max_avg_flow_length:
                            max_avg_flow_length = tmp_avg_flow_length
                        tmp_bin_ix = int((tmp_blk.last_max_angle + math.pi) / tmp_blk.bin_angle)
                        tmp_angle_size = tmp_blk.angle_hist[tmp_bin_ix]
                        if current_max_angle_size < tmp_angle_size:
                            current_max_angle_size = tmp_angle_size
                v = avg_flow_length/max_avg_flow_length
                # and check if the angle is stable
                #bin_ix = int((blk.last_max_angle + math.pi) / blk.bin_angle)
                #v *= blk.angle_hist[bin_ix] / max_angle_size
                hsv[:,:,2] = (hsv[:,:,2] * v).astype(np.uint8)
                blk_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # draw direction
            blk_dir_from = (self.flow_block_radius, self.flow_block_radius)
            blk_dir_to = (self.flow_block_radius + int(self.flow_block_radius * math.cos(blk.last_max_angle)), self.flow_block_radius + int(self.flow_block_radius * math.sin(blk.last_max_angle)))
            cv2.arrowedLine(blk_img, blk_dir_from, blk_dir_to, (255, 255, 255))
            # draw
            blk_from_x = x_blk*self.flow_block_size
            blk_from_y = y_blk*self.flow_block_size
            try:
                self.flows[blk_from_y:blk_from_y+self.flow_block_size,blk_from_x:blk_from_x+self.flow_block_size,:] = blk_img
            except:
                print('blk_img failed at ' + str(blk_from_y) + ',' + str(blk_from_x))

        self.recent_max_angle_size.append(current_max_angle_size)
        if len(self.recent_max_angle_size) > int(self.flows.shape[0] * self.flows.shape[1] / 10):
            del self.recent_max_angle_size[0]

    def get_x_blk(self, x):
        x = min(max(0., x), self.flows.shape[1])
        return int(x / self.flow_block_size)

    def get_y_blk(self, y):
        y = min(max(0., y), self.flows.shape[0])
        return int(y / self.flow_block_size)

    def get_flow_block(self, x_blk, y_blk):
        if not self._flow_blocks.has_key(x_blk):
            self._flow_blocks[x_blk] = {}

        if not self._flow_blocks[x_blk].has_key(y_blk):
            self._flow_blocks[x_blk][y_blk] = FlowBlock()

        blk = self._flow_blocks[x_blk][y_blk]
        return blk

    def get_angle_at(self, point):
        x_blk = self.get_x_blk(point[0])
        y_blk = self.get_y_blk(point[1])
        blk = self.get_flow_block(x_blk, y_blk)
        return blk.get_max_angle()

    def get_flows(self, with_color_compass=True):
        # return only copy to avoid accessing partial array(??)
        # e.g.     self.flows[blk_from_y:blk_from_y+self.flow_block_size,blk_from_x:blk_from_x+self.flow_block_size,:] = blk_img
        #          ValueError: could not broadcast input array from shape (10,10,3) into shape (0,10,3)
        copy = self.flows.copy()
        if with_color_compass:
            # add color compass
            copy[self.radius:self.radius+self.diameter,self.radius:self.radius+self.diameter,] = self.color_compass_image
        return copy
