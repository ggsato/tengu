#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class TenguObject(object):
    """ TenguObject is a base class of any travelling object classes in Tengu framework.

    The purpose of this class is to give the best estimate of a current location based on a predicted location and an observed location.
    To achieve this purpose, this class uses a KalmanFilter.
    """

    _very_small_value = 0.000001

    def __init__(self, x0, R_std=10., Q=.0001, dt=1, P=100., std_devs=None):
        super(TenguObject, self).__init__()
        self._filter = TenguObject.create_filter(x0, R_std, Q, dt, P, std_devs)
        self._zs = []
        self._xs = []
        self._covs = []
        self._last_accepted_residual = None
        self._R_std = R_std

    @property
    def location(self):
        """ the current position of this TenguObject instance

        An estimated current location based on this filter's prediction and a predicted position at time t

        returns (x, y) or None if no location history
        """
        if len(self._xs) == 0:
            return (self._filter.x[0], self._filter.x[3])

        return [self._xs[-1][0], self._xs[-1][3]]

    @property
    def measurement(self):
        if len(self._zs) == 0:
            return (self._filter.x[0], self._filter.x[3])

        return [self._zs[-1][0], self._zs[-1][1]]

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

    @property
    def observed_travel_distance(self):
        """ the travel distance of observations
        """
        if len(self._zs) == 0:
            return -1

        start, end = None, None
        for i in range(len(self._zs)):
            if start is None:
                start = self._zs[i]
            if end is None:
                end = self._zs[-1-1*i]
            if start is not None and end is not None:
                break

        if start is None or end is None:
            return -1

        travel_distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        return travel_distance

    def update_location(self, z):
        """ update the current location by a predicted location and the given z(observed_location)
        """
        # predict
        self._filter.predict()
        if z is not None:
            self._filter.update(z)
        self._xs.append(self._filter.x)
        self._covs.append(self._filter.P)
        self._zs.append(z)

    def accept_measurement(self, z):
        """ check if a given measurement is aacceptable

        please override as required
        """
        # accept until enough time steps have passed
        if self.direction is None:
            return True

        last_x = self._xs[-1]
        variance_speed = self.variance_speed
        variance_accl = self.variance_accel
        variance = [variance_speed[0] + 0.5 * variance_accl[0], variance_speed[1] + 0.5 * variance_accl[1]]

        next_x = last_x[0] + last_x[1] + 0.5 * last_x[2]
        next_y = last_x[3] + last_x[4] + 0.5 * last_x[5]

        residual_x = math.sqrt((z[0] - next_x)**2)
        residual_y = math.sqrt((z[1] - next_y)**2)

        most_likely_x = variance[0]*3
        most_likely_y = variance[1]*3

        residual_ratio = [residual_x / most_likely_x, residual_y / most_likely_y]

        # accept if the ratio is smailer than 1
        accept = max(residual_ratio) < 1.0
        self._last_accepted_residual = [residual_ratio[0], residual_ratio[1]]

        return accept

    def similarity(self, another):
        """ calculate a similarity between self and another instance of TenguObject 

        a similarity
        """
        pass

    @property
    def variance(self):
        """ the variance of x """
        if len(self._covs) == 0:
            return None

        return [max(TenguObject._very_small_value, math.sqrt(self._covs[-1][0][0])), max(TenguObject._very_small_value, math.sqrt(self._covs[-1][3][3]))]

    @property
    def variance_speed(self):
        """ the variance of x speed """
        if len(self._covs) == 0:
            return None

        return [math.sqrt(self._covs[-1][1][1]), math.sqrt(self._covs[-1][4][4])]

    @property
    def variance_accel(self):
        """ the variance of x accel """
        if len(self._covs) == 0:
            return None

        return [math.sqrt(self._covs[-1][2][2]), math.sqrt(self._covs[-1][5][5])]

    @property
    def R_std(self):
        return self._R_std

    @staticmethod
    def create_filter(x0, R_std, Q, dt, P, std_devs):
        """ creates a second order Kalman Filter

        R_std: float, a standard deviation of measurement errors
        Q    : float, a covariance of process noises
        dt   : int, a time unit
        P    : float, a maximum initial variance of all states to build a covariance matrix
        x0   : 1x6 array, initial values of x(x0)
        """
        kf = KalmanFilter(dim_x=6, dim_z=2)
        kf.R = np.eye(2) * R_std**2
        if std_devs is None:
            kf.P = np.eye(6) * P
            q = Q_discrete_white_noise(3, dt, Q)
            kf.Q = block_diag(q, q)
        else:
            kf.P = np.array([[std_devs[0]*3**2, 0., 0., 0., 0., 0.],
                         [0., std_devs[1]*3**2, 0., 0., 0., 0.],
                         [0., 0., std_devs[2]*3**2, 0., 0., 0.],
                         [0., 0., 0., std_devs[3]*3**2, 0., 0.],
                         [0., 0., 0., 0., std_devs[4]*3**2, 0.],
                         [0., 0., 0., 0., 0., std_devs[5]*3**2]])
            kf.Q = np.array([[0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [0., 0., std_devs[2]*3**2, 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., std_devs[5]*3**2]])
        kf.F = np.array([[1., dt, .5*dt*dt, 0., 0., 0.],
                         [0., 1., dt, 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 1., dt, .5*dt*dt],
                         [0., 0., 0., 0., 1., dt],
                         [0., 0., 0., 0., 0., 1.]])
        kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0.]])
        # initial x
        kf.x = np.array([x0]).T
        return kf