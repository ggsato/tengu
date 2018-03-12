#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class TenguObject(object):
    """ TenguObject is a base class of any travelling object classes.

    The purpose of this class is to give the best estimate of a current location based on a predicted location and an observed location.
    Also, this class provides subclasses with additional benefits such as speed and direction that can be calculated from a time series of locations.
    
    To achieve the first purpose, this class uses a KalmanFilter.
    """

    def __init__(self, R_std=10., Q=.0001, dt=1, P=100.):
        self._filter = TenguObject.create_filter(R_std, Q, dt, P)
        self._zs = []
        self._xs = []
        self._covs = []

    @property
    def location(self):
        """ the current position of this TenguObject instance

        An estimated current location based on this filter's prediction and a predicted position at time t

        returns (x, y) or None if no location history
        """
        if len(self._xs) == 0:
            return (self._filter.x[0], self._filter.x[3])

        return (self._xs[-1][0], self._xs[-1][3])

    @property
    def measurement(self):
        if len(self._zs) == 0:
            return (self._filter.x[0], self._filter.x[3])

        return (self._zs[-1][0], self._zs[-1][1])

    @property
    def direction(self):
        """ the direction of this TenguObject instance

        A direction ranges from -pi(-180) to pi(180),
        in the OpenCV coordinate system, the upper left corner is the origin.

        The difference between a direction and a heading is that 
        heading is the current direction, which could vary frequently,
        while direction is more stable because it is computed from the first to the last.

        return a float value (-pi, pi)  
        """
        if len(self._xs) < 2:
            return None

        return math.atan2(self._xs[-1][3] - self._xs[0][3], self._xs[-1][0] - self._xs[0][0])

    @property
    def heading(self):
        """ the current direction of this TenguObject instance

        A direction ranges from -pi(-180) to pi(180),
        in the OpenCV coordinate system, the upper left corner is the origin.

        return a float value (-pi, pi)  
        """
        if len(self._xs) < 2:
            return None

        return math.atan2(self._xs[-1][3] - self._xs[-2][3], self._xs[-1][0] - self._xs[-2][0])

    @property
    def speed(self):
        """ pixels/frame

        returns a float value >= 0
        """
        movement = self.movement
        print('getting speed from the last movement {}'.format(movement))

        if movement is None:
            return None

        return math.sqrt(movement[0]**2 + movement[1]**2)

    @property
    def movement(self):
        """ the last movement of x, y, which is the speed in x and y
        """
        if len(self._xs) == 0:
            return None

        return (self._xs[-1][1], self._xs[-1][4])

    def update_location(self, z):
        """ update the current location by a predicted location and the given z(observed_location)


        """
        if len(self._zs) == 0:
            if z is None:
                return
            # initialize
            self._filter.x = np.array([[z[0], 0., 0., z[1], 0., 0.,]]).T
        else:
            # predict
            self._filter.predict()
            if z is not None:
                self._filter.update(z)
            self._xs.append(self._filter.x)
            self._covs.append(self._filter.P)
        self._zs.append(z)

    def similarity(self, another):
        """ calculate a similarity between self and another instance of TenguObject 

        a similarity
        """
        pass

    @staticmethod
    def create_filter(R_std, Q, dt, P):
        """ creates a second order Kalman Filter

        R_std: float, a standard deviation of measurement error
        Q    : float, a covariance of process noise
        dt   : int, a time unit
        P    : float, a maximum initial covariance
        """
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        kf.P = np.eye(6) * P
        kf.R = np.eye(2) * R_std**2
        q = Q_discrete_white_noise(3, dt, Q)
        kf.Q = block_diag(q, q)
        kf.F = np.array([[1., dt, .5*dt*dt, 0., 0., 0.],
                         [0., 1., dt, 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 1., dt, .5*dt*dt],
                         [0., 0., 0., 0., 1., dt],
                         [0., 0., 0., 0., 0., 1.]])
        kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0.]])
        return kf