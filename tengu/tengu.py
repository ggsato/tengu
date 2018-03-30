#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import time
import math
import logging
import os, shutil

from weakref import WeakValueDictionary

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

    def __init__(self, tmpfs_cleanup_interval_in_frames=1*25):
        self.logger= logging.getLogger(__name__)
        self._tmpfs_cleanup_interval_in_frames = tmpfs_cleanup_interval_in_frames

        self._observers = WeakValueDictionary()
        self._current_frame = -1

        self._scene_model = TenguSceneModel()

        """ terminate if True
        TODO: this works only in the same process.
        Otherwise, this has to be something else to communicate between processes
        """
        self._stopped = False

    @property
    def frame_no(self):
        return self._current_frame

    def add_observer(self, observer):
        if observer == None:
            return
        if not isinstance(observer, TenguObserver):
            self.logger.info('{} is not an instance of TenguObserver')
            return
        observer_id = id(observer)
        if self._observers.has_key(observer_id):
            return
        self._observers[observer_id] = observer

    def remove_observer(self, observer):
        if observer == None:
            return
        observer_id = id(observer)
        if self._observers.has_key(observer_id):
            del self._observers[observer_id]

    def _notify_frame_analyzed(self, event_dict):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            observer.frame_analyzed(event_dict)

    def _notify_analysis_finished(self):
        for observer_id in self._observers:
            observer = self._observers[observer_id]
            observer.analysis_finished()

    """ start running analysis on src
    """
    def run(self, src=None, roi=None, scale=None, every_x_frame=1, rotation=0, queue=None, max_queue_wait=10, skip_to=-1, sensors=[], frame_queue_timeout_in_secs=10):
        self.logger.info('running with roi={}, scale={}, skipping to {}'.format(roi, scale, skip_to))
        
        if src is None:
            self.logger.error('src has to be set')
            return

        try:
            cam = cv2.VideoCapture(int(src))
        except:
            cam = cv2.VideoCapture(src)        
        if cam is None or not cam.isOpened():
            self.logger.debug(self._src + ' is not available')
            return

        if Tengu.PREFERRED_CAMERA_SETTINGS is not None:
            for key in Tengu.PREFERRED_CAMERA_SETTINGS:
                self.logger.info('set a user defined camera setting {} of {}'.format(Tengu.PREFERRED_CAMERA_SETTINGS[key], key))
                cam.set(key, Tengu.PREFERRED_CAMERA_SETTINGS[key])

        # check tmpfs directory exists
        shutil.rmtree(Tengu.TMPFS_DIR)
        os.makedirs(Tengu.TMPFS_DIR)

        try:
            # initialize scene model
            self._scene_model.initialize(sensors)
            self._scene_model.start()

            # run loop
            while not self._stopped:

                self.logger.info('reading the next frame')
                ret, frame = cam.read()
                #self.logger.info('frame shape = {}'.format(frame.shape))
                if not ret:
                    self.logger.info('no frame is avaiable')
                    break

                event_dict = {}

                # increment
                self._current_frame += 1
                event_dict[Tengu.EVENT_FRAME_NO] = self._current_frame

                # rotate
                if rotation != 0:
                    rows, cols, channels = frame.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
                    frame = cv2.warpAffine(frame, M, (cols, rows))

                # use copy for gui use, which is done asynchronously, meaning may corrupt buffer during camera updates
                event_dict[Tengu.EVENT_FRAME] = frame.copy()

                # preprocess
                cropped = self.preprocess(frame, roi, scale)
                cropped_copy = cropped.copy()
                event_dict[Tengu.EVENT_FRAME_CROPPED] = cropped_copy

                # synchronize with a client if necessary
                if queue is not None:
                    done = False
                    while not done and not self._stopped:
                        # wait until queue becomes ready
                        # NOTE: this queue is from Queue module, not from multiprocessing
                        try:
                            self.logger.info('putting a cropped copy in to a queue')
                            queue.put_nowait(cropped_copy)
                            done = True
                        except:
                            self.logger.info('queue is full, sleeping...')
                            # max 100 FPS
                            time.sleep(0.01)

                    if self._stopped:
                        break

                # skip if necessary
                if (every_x_frame > 1 and self._current_frame % every_x_frame != 0) or (skip_to > 0 and self._current_frame < skip_to):
                    self.logger.debug('skipping frame at {}'.format(self._current_frame))
                    event_dict[Tengu.EVENT_DETECTIONS] = []
                    event_dict[Tengu.EVENT_DETECTION_CLASSES] = []
                    event_dict[Tengu.EVENT_TRACKLETS] = []
                    self._notify_frame_analyzed(event_dict)
                    continue

                # # flow analysis
                # if tengu_flow_analyzer is not None:
                #     detections, class_names, tracklets, scene = tengu_flow_analyzer.analyze_flow(cropped, self._current_frame)
                #     event_dict[Tengu.EVENT_DETECTIONS] = detections
                #     event_dict[Tengu.EVENT_DETECTION_CLASSES] = class_names
                #     event_dict[Tengu.EVENT_TRACKLETS] = tracklets
                #     event_dict[Tengu.EVENT_SCENE] = scene

                #     # scene analysis
                #     if tengu_scene_analyzer is not None:
                #         self.logger.debug('calling scene analyzer')
                #         tengu_scene_analyzer.analyze_scene(cropped, scene)

                #     else:
                #         self.logger.debug('skip calling scene analyzer')

                # create a sensor input to pass an image as a file
                img_path = os.path.join(Tengu.TMPFS_DIR, 'frame-{}.jpg'.format(self._current_frame))
                cv2.imwrite(img_path, cropped_copy)
                self.logger.info('wrote a frame image {}'.format(img_path))

                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < frame_queue_timeout_in_secs:
                    try:
                        self.logger.info('putting a frame image path {} in a queue'.format(img_path))
                        self._scene_model.input_queue.put_nowait(img_path)
                        done = True
                    except:
                        self.logger.info('failed to put {} in a queue, sleeping'.format(img_path))
                        time.sleep(0.001)
                    elapsed = (time.time() - start) / 1000
                
                # then??

                if self._current_frame % self._tmpfs_cleanup_interval_in_frames == 0:
                    self.logger.info('cleaning up tmp images')
                    self.cleanup_tmp_images()

                # temporary
                event_dict[Tengu.EVENT_DETECTIONS] = []
                event_dict[Tengu.EVENT_DETECTION_CLASSES] = []
                event_dict[Tengu.EVENT_TRACKLETS] = []

                # notify observers
                self._notify_frame_analyzed(event_dict)
                self.logger.info('analyzed frame no {}'.format(self._current_frame))

        except:
            self.logger.exception('Unknow Exception {}'.format(sys.exc_info()))

        self.logger.info('exitted run loop, exitting...')
        self._scene_model.finish()
        self._scene_model.join()
        self._stopped = True

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

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.info('stopping...')
        self._stopped = True

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
