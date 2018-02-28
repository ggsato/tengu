#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class TenguObject(object):
    """ TenguObject is a base class of any travelling object classes.

    The purpose of this class is to give the best estimate of a current location based on a predicted location and an observed location.
    Also, this class provides subclasses with additional benefits such as speed and direction that can be calculated from a time series of locations.
    
    To achieve the first purpose, this class uses a KalmanFilter.
    """

    DEFAULT_LOCATION_VALUE = -1.0

    def __init__(self, fps=25, keep_locations_in_seconds=10):
        self._fps = fps
        self._keep_locations_in_seconds = keep_locations_in_seconds
        self._locations = np.ones((keep_locations_in_seconds, fps, 2), np.dtype=np.float) * TenguObject.DEFAULT_LOCATION_VALUE
        self._location_ix = -1

    @property
    def path(self):
        """ a path is a history of past locations

        returns a list of (x, y)
        """
        return copy.copy(self._locations)

    @property
    def location(self):
        """ the current position of this TenguObject instance

        An estimated current location based on this filter's prediction and a predicted position at time t

        returns (x, y) or None if no location history
        """
        if len(self._locations) == 0:
            return None

        return self._locations[-1]

    @property
    def direction(self):
        """ the current direction of this TenguObject instance

        A direction ranges from -pi(-180) to pi(180),
        in the OpenCV coordinate system, the upper left corner is the origin.

        return a float value (-pi, pi)  
        """
        pass

    @property
    def speed(self):
        """ pixels/frame

        returns a float value >= 0
        """
        pass

    def update_location(self, observed_location):
        """ update the current location by a predicted location and the given observed_location


        """
        pass

    def similarity(self, another):
        """ calculate a similarity between self and another instance of TenguObject 

        a similarity
        """
        pass