#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import time
import math
import logging
import os, shutil
import threading
from Queue import Queue

from tengu_observer import TenguObserver
from tengu_scene_model import TenguSceneModel

class Tengu(object):

    # dictionary to set camera settings
    # e.g. width {3, 1280}
    PREFERRED_CAMERA_SETTINGS = None

    # event keys
    EVENT_FRAME = 'ef'
    EVENT_FRAME_NO = 'efn'
    EVENT_FRAME_CROPPED = 'efc'

    EVENT_DETECTIONS = 'ed'
    EVENT_DETECTION_CLASSES = 'edc'

    EVENT_TRACKLETS = 'et'
    EVENT_SCENE = 'es'

    # tmpfs to exchange an image between processes
    TMPFS_DIR = '/dev/shm/tengu'

    def __init__(self):
        self.logger= logging.getLogger(__name__)

        self._scene_model = TenguSceneModel()
        self._camera_reader = None

        """ terminate if True
        TODO: this works only in the same process.
        Otherwise, this has to be something else to communicate between processes
        """
        self._stopped = False

    @property
    def frame_no(self):
        return self._camera_reader.current_frame

    """ start running analysis on src
    """
    def run(self, src=None, roi=None, scale=1.0, every_x_frame=1, rotation=0, skip_to=-1, frame_queue_timeout_in_secs=10, sensors=[], queue=None):

        if src is None:
            self.logger.error('src has to be set')
            return

        # start camera
        self._camera_reader = CameraReader(src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, queue, self._scene_model.input_queue)

        try:
            # start reading camera
            self._camera_reader.start()

            # initialize scene model
            self._scene_model.initialize(sensors)
            self._scene_model.start()

            # run loop
            while not self._stopped:

                frame_analysis_start = time.time()

                # output event dictionary
                # input event dictionary is separately set from camera reader
                event_dict = {}
                
                # get an output
                done = False
                start = time.time()
                elapsed = 0
                output_dict = None
                while not done and elapsed < frame_queue_timeout_in_secs and not self._stopped:
                    try:
                        self.logger.debug('getting an output dict from a queue')
                        output_dict = self._scene_model.output_queue.get_nowait()
                        self.logger.debug('got an output dict {}'.format(output_dict))
                        done = True
                    except:
                        self.logger.debug('failed to get an output dict from a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = (time.time() - start) / 1000

                if output_dict is None:
                    self.logger.error('failed to get any output from an output queue within {} seconds'.format(frame_queue_timeout_in_secs))
                    break
                else:
                    event_dict[Tengu.EVENT_DETECTIONS] = output_dict['d']
                    event_dict[Tengu.EVENT_DETECTION_CLASSES] = output_dict['c']
                    event_dict[Tengu.EVENT_TRACKLETS] = output_dict['t']
                    event_dict[Tengu.EVENT_FRAME_NO] = output_dict['n']

                # put in the queue
                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < frame_queue_timeout_in_secs and not self._stopped:
                    try:
                        self.logger.debug('putting an event dict to a queue')
                        queue.put_nowait(event_dict)
                        done = True
                    except:
                        self.logger.debug('failed to put an event dict in a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = (time.time() - start) / 1000

                if not done:
                    self.logger.error('failed to put an output dict in a queue within {} seconds'.format(frame_queue_timeout_in_secs))
                    break

                self.logger.info('analyzed frame no {} in {} ms'.format(event_dict[Tengu.EVENT_FRAME_NO], (time.time() - frame_analysis_start)))

        except:
            self.logger.exception('Unknow Exception {}'.format(sys.exc_info()))

        self.logger.info('exitted run loop, exitting...')
        if self._camera_reader is not None:
            self._camera_reader.finish()
            self._camera_reader.join()
        self._scene_model.finish()
        self._scene_model.join()
        self._stopped = True

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.info('stopping...')
        self._camera_reader.finish()
        self._stopped = True

class CameraReader(threading.Thread):
    def __init__(self, video_src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, queue, scene_input_queue, tmpfs_cleanup_interval_in_frames=1*25):
        super(CameraReader, self).__init__()
        self.logger= logging.getLogger(__name__)
        self._video_src = video_src
        # TODO: check these values are correct
        self._roi = roi
        self._scale = scale
        self._every_x_frame = every_x_frame
        self._rotation = rotation
        self._skip_to = skip_to
        self._frame_queue_timeout_in_secs = frame_queue_timeout_in_secs
        self._queue = queue
        self._scene_input_queue = scene_input_queue
        self._tmpfs_cleanup_interval_in_frames = tmpfs_cleanup_interval_in_frames
        # transient
        self._cam = None
        self._current_frame = 0
        self._finished = False

    @property
    def current_frame(self):
        return self._current_frame

    def setup(self):
        try:
            self._cam = cv2.VideoCapture(int(self._video_src))
        except:
            self._cam = cv2.VideoCapture(self._video_src)        
        if self._cam is None or not self._cam.isOpened():
            self.logger.error(self._video_src + ' is not available')

        if Tengu.PREFERRED_CAMERA_SETTINGS is not None:
            for key in Tengu.PREFERRED_CAMERA_SETTINGS:
                self.logger.info('set a user defined camera setting {} of {}'.format(Tengu.PREFERRED_CAMERA_SETTINGS[key], key))
                self._cam.set(key, Tengu.PREFERRED_CAMERA_SETTINGS[key])

        # check tmpfs directory exists
        shutil.rmtree(Tengu.TMPFS_DIR)
        os.makedirs(Tengu.TMPFS_DIR)

    def run(self):
        self.setup()
        # read frames
        while not self._finished:
            
            self.logger.info('reading the next frame')
            ret, frame = self._cam.read()

            # finished
            if not ret:
                self.logger.info('no frame is avaiable')
                break

            # camera event dictionary
            event_dict = {Tengu.EVENT_FRAME_NO: self._current_frame}

            # rotate
            if self._rotation != 0:
                rows, cols, channels = frame.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), self._rotation, 1)
                frame = cv2.warpAffine(frame, M, (cols, rows))

            # use copy for gui use, which is done asynchronously, meaning may corrupt buffer during camera updates
            event_dict[Tengu.EVENT_FRAME] = frame.copy()

            # preprocess
            cropped = self.preprocess(frame, self._roi, self._scale)
            cropped_copy = cropped.copy()
            event_dict[Tengu.EVENT_FRAME_CROPPED] = cropped_copy

            # put in a queue
            done = False
            start = time.time()
            elapsed = 0
            while not done and elapsed < self._frame_queue_timeout_in_secs and not self._finished:
                try:
                    self.logger.debug('putting a frame image in a queue')
                    self._queue.put_nowait(event_dict)
                    done = True
                except:
                    self.logger.debug('failed to put event dict in a queue, sleeping')
                    time.sleep(0.001)
                elapsed = (time.time() - start) / 1000

            # skip if necessary
            if (self._every_x_frame > 1 and self._current_frame % self._every_x_frame != 0) or (self._skip_to > 0 and self._current_frame < self._skip_to):

                self.logger.debug('skipping frame at {}'.format(self._current_frame))
                continue

            img_path = os.path.join(Tengu.TMPFS_DIR, 'frame-{}.jpg'.format(self._current_frame))
            cv2.imwrite(img_path, cropped_copy)
            self.logger.debug('wrote a frame image {}'.format(img_path))

            done = False
            start = time.time()
            elapsed = 0
            while not done and elapsed < self._frame_queue_timeout_in_secs and not self._finished:
                try:
                    self.logger.debug('putting a frame image path {} in a queue'.format(img_path))
                    self._scene_input_queue.put_nowait(img_path)
                    done = True
                except:
                    self.logger.debug('failed to put {} in a queue, sleeping'.format(img_path))
                    time.sleep(0.001)
                elapsed = (time.time() - start) / 1000

            if self._current_frame % self._tmpfs_cleanup_interval_in_frames == 0:
                self.logger.info('cleaning up tmp images')
                self.cleanup_tmp_images()

            # increment
            self._current_frame += 1

        self._finished = True
        # cleanup queues
        self.logger.info('cleaning up queue')
        while not self._queue.empty():
            self._queue.get_nowait()
            self.logger.info('cleaning up scene input queue')
        while not self._scene_input_queue.empty():
            self._scene_input_queue.get_nowait()
        self.logger.info('finished reading camera')

    def preprocess(self, frame, roi, scale):

        cropped = None
        
        if scale == 1.0:
            resized = frame
        else:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if roi is not None:
            # crop
            cropped = resized[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        else:
            cropped = resized
        
        return cropped

    def cleanup_tmp_images(self):
        # delete image files created before the last interval
        files = os.listdir(Tengu.TMPFS_DIR)
        for file in files:
            if not file.endswith('.jpg'):
                continue
            img_frame_no = int(file[file.index('-')+1:file.index('.')])
            if (self._current_frame - img_frame_no) > self._tmpfs_cleanup_interval_in_frames:
                self.logger.info('removing tmp image file {}'.format(file))
                os.remove(os.path.join(Tengu.TMPFS_DIR, file))
            else:
                self.logger.info('keep tmp image file {}'.format(file))

    def finish(self):
        self.logger.info('finishing camera reading')
        self._finished = True

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
