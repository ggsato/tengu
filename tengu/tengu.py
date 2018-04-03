#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import time
import math
import logging
import os, shutil
import traceback
from multiprocessing import Process, Value
import threading

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
        self._tmp_image_cleaner = None

        """ terminate if True
        TODO: this works only in the same process.
        Otherwise, this has to be something else to communicate between processes
        """
        self._stopped = Value('i', 0)

    @property
    def frame_no(self):
        return self._camera_reader.current_frame

    """ start running analysis on src
    """
    def run(self, src=None, roi=None, scale=1.0, every_x_frame=1, rotation=0, skip_to=-1, frame_queue_timeout_in_secs=10, queue=None, tmpfs_cleanup_interval_in_frames=10*25):

        if src is None:
            self.logger.error('src has to be set')
            return

        # check tmpfs directory exists
        if os.path.exists(Tengu.TMPFS_DIR):
            shutil.rmtree(Tengu.TMPFS_DIR)
        os.makedirs(Tengu.TMPFS_DIR)

        try:

            # start reading camera
            self.logger.info('starting camera reader')
            self._camera_reader = CameraReader(src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, queue, self._scene_model.input_queue)
            self._camera_reader.start()
            while self._camera_reader._finished.value == -1:
                # this means not started yet
                self.logger.info('waiting for camera running')
                time.sleep(0.1)

            # initialize scene model
            self.logger.info('starting scene model')
            self._scene_model.start_sensor()
            self._scene_model.start()
            while self._scene_model._finished.value == -1:
                # this means not started yet
                self.logger.info('waiting for scene model running, process alive = {}, exitcode = {}'.format(self._scene_model.is_alive(), self._scene_model.exitcode))
                time.sleep(0.1)

            # start cleaner
            self.logger.info('starting tmp image cleaner')
            # tmp cleanup
            self._tmp_image_cleaner = TmpImageCleaner(tmpfs_cleanup_interval_in_frames)
            self._tmp_image_cleaner.start()

            # run loop
            while self._stopped.value == 0:

                frame_analysis_start = time.time()

                # output event dictionary
                # input event dictionary is separately set from camera reader
                event_dict = {}
                
                # get an output
                done = False
                start = time.time()
                elapsed = 0
                output_dict = None
                while not done and elapsed < frame_queue_timeout_in_secs and self._stopped.value == 0:
                    try:
                        self.logger.debug('getting an output dict from a queue')
                        output_dict = self._scene_model.output_queue.get_nowait()
                        self.logger.debug('got an output dict {}'.format(output_dict))
                        done = True
                    except:
                        self.logger.debug('failed to get an output dict from a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start

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
                while not done and elapsed < frame_queue_timeout_in_secs and self._stopped.value == 0:
                    try:
                        self.logger.debug('putting an event dict to a queue')
                        queue.put_nowait(event_dict)
                        done = True
                    except:
                        self.logger.debug('failed to put an event dict in a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start

                if not done:
                    if self._stopped.value == 1:
                        self.logger.info('breaking tengu loop...')
                    else:
                        self.logger.error('failed to put an output dict in a queue within {} seconds'.format(frame_queue_timeout_in_secs))
                    break

                self.logger.info('analyzed frame no {} in {} s'.format(event_dict[Tengu.EVENT_FRAME_NO], (time.time() - frame_analysis_start)))

                # update time of tmp cleaner
                while self._tmp_image_cleaner.current_frame < self._camera_reader.current_frame:
                    self.logger.info('incrementing cleaner current frame {}, camera reader {}'.format(self._tmp_image_cleaner.current_frame, self._camera_reader.current_frame))
                    self._tmp_image_cleaner.increment_current_frame()

        except:
            info = sys.exc_info()
            self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:

            if self._camera_reader is not None:
                if self._camera_reader.finish():
                    self._camera_reader.join()
                else:
                    raise Exception('camera reader did not stop!')

            if self._scene_model is not None:
                if self._scene_model.finish():
                    self._scene_model.join()
                else:
                    self._scene_model.terminate()

            # at this point, only queue that communicates with a client has not yet been cleaned up, and which should be done by the clietn

            if self._tmp_image_cleaner is not None:
                if self._tmp_image_cleaner.finish():
                    self._tmp_image_cleaner.join()
                else:
                    self._tmp_image_cleaner.terminate()

            self._stopped.value = 2
            self.logger.info('exitted run loop, exitting... {}'.format(self._stopped.value))

    def save(self, model_folder):
        self.logger.debug('saving current models in {}...'.format(model_folder))

    def load(self, model_folder):
        self.logger.debug('loading models from {}...'.format(model_folder))

    def stop(self):
        self.logger.info('stopping...')
        if self._stopped.value == 0:
            self._stopped.value = 1

        while self._stopped.value != 2:
            self.logger.info('waiting for exitting tengu loop {}'.format(self._stopped.value))
            time.sleep(0.01)


class CameraReader(threading.Thread):
    def __init__(self, video_src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, queue, scene_input_queue, **kwargs):
        super(CameraReader, self).__init__(**kwargs)
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
        # transient
        self._cam = None
        self._current_frame = Value('i', 0)
        self._finished = Value('i', -1)

    @property
    def current_frame(self):
        return self._current_frame.value

    def setup(self):
        self.logger.info('setting up camera reader')
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
        self._finished.value = 0

    def run(self):
        self.logger.info('running camera reader')
        self.setup()
        # read frames
        try:
            while self._finished.value == 0:

                frame_start = time.time()
                
                self.logger.info('reading the next frame')
                ret, frame = self._cam.read()

                # finished
                if not ret:
                    self.logger.info('no frame is avaiable')
                    break

                # camera event dictionary
                event_dict = {Tengu.EVENT_FRAME_NO: self._current_frame.value}

                # rotate
                if self._rotation != 0:
                    rows, cols, channels = frame.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), self._rotation, 1)
                    frame = cv2.warpAffine(frame, M, (cols, rows))

                # preprocess
                cropped = self.preprocess(frame, self._roi, self._scale)

                # write an image to exchange between processes
                start = time.time()
                img_path = os.path.join(Tengu.TMPFS_DIR, 'frame-{}.jpg'.format(self._current_frame.value))
                cv2.imwrite(img_path, cropped)
                self.logger.info('wrote a frame image {} in {} s'.format(img_path, time.time() - start))

                event_dict[Tengu.EVENT_FRAME_CROPPED] = img_path

                # put in a queue, this is shared with a GUI client
                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < self._frame_queue_timeout_in_secs and self._finished.value == 0:
                    try:
                        self.logger.info('putting a frame image path {} in a queue'.format(img_path))
                        self._queue.put_nowait(event_dict)
                        done = True
                    except:
                        self.logger.debug('failed to put event dict in a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start

                # skip if necessary
                if (self._every_x_frame > 1 and self._current_frame.value % self._every_x_frame != 0) or (self._skip_to > 0 and self._current_frame.value < self._skip_to):

                    self.logger.info('skipping frame at {}'.format(self._current_frame.value))
                    continue

                # put in a scene input queue, this is shared with scene model
                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < self._frame_queue_timeout_in_secs and self._finished.value == 0:
                    try:
                        self.logger.info('putting a frame image path {} in a queue'.format(img_path))
                        self._scene_input_queue.put_nowait(img_path)
                        done = True
                    except:
                        self.logger.info('failed to put {} in a queue, sleeping'.format(img_path))
                        time.sleep(0.001)
                    elapsed = time.time() - start

                self.logger.info('put frame img and its path at time {} in {} s'.format(self._current_frame.value, time.time() - frame_start))

                # increment
                self._current_frame.value += 1
        except:
            info = sys.exc_info()
            self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:
            # cleanup queues
            self.logger.info('cleaning up queue')
            self._queue.close()
            while not self._queue.empty():
                self._queue.get_nowait()
            self.logger.info('cleaning up scene input queue')
            self._scene_input_queue.close()
            while not self._scene_input_queue.empty():
                self._scene_input_queue.get_nowait()
            # mark exit
            self._finished.value = 2
            self.logger.info('exitted camera loop {}'.format(self._finished.value))
        self.logger.info('finished reading camera')

    def preprocess(self, frame, roi, scale):

        start = time.time()

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

        self.logger.info('preprocessing done in {} s'.format(time.time() - start))
        
        return cropped

    def finish(self):
        self.logger.info('finishing camera reading')
        if self._finished.value == 0:
            self._finished.value = 1

        start = time.time()
        elapsed = 0
        while elapsed < 1.0 and self._finished.value != 2:
            self.logger.info('waiting for exitting camera loop, finished = {}'.format(self._finished.value))
            time.sleep(0.001)
            elapsed = time.time() - start

        return self._finished.value == 2

class TmpImageCleaner(Process):

    def __init__(self, tmpfs_cleanup_interval_in_frames, **kwargs):
        super(TmpImageCleaner, self).__init__(**kwargs)
        self.logger= logging.getLogger(__name__)
        self._current_frame = Value('i', 0)
        self._tmpfs_cleanup_interval_in_frames = Value('i', tmpfs_cleanup_interval_in_frames)
        self._finished = Value('i', 0)

    @property
    def current_frame(self):
        return self._current_frame.value

    def increment_current_frame(self):
        self._current_frame.value += 1

    def run(self):
        self.logger.info('running tmp image cleaner')
        try:
            while self._finished.value == 0:

                self.cleanup_tmp_images()

                start = time.time()
                elapsed = 0
                while elapsed < 10 and self._finished.value == 0:
                    time.sleep(0.1)
                    elapsed = time.time() - start
        except:
            info = sys.exc_info()
            self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:
            self._finished.value = 2
            self.logger.info('exitting cleanup tmp image loop {}'.format(self._finished.value))

    def cleanup_tmp_images(self):
        self.logger.info('cleaning up tmp images...')
        # delete image files created before the last interval
        start = time.time()
        files = os.listdir(Tengu.TMPFS_DIR)
        for file in files:
            if not file.endswith('.jpg'):
                continue
            img_frame_no = int(file[file.index('-')+1:file.index('.')])
            if (self._current_frame.value - img_frame_no) > self._tmpfs_cleanup_interval_in_frames:
                self.logger.info('removing tmp image file {}'.format(file))
                os.remove(os.path.join(Tengu.TMPFS_DIR, file))
            else:
                self.logger.info('keep tmp image file {}'.format(file))
        self.logger.info('cleaned up tmp images at time {} in {} ms'.format(self._current_frame.value, time.time() - start))

    def finish(self):
        self.logger.info('finishing cleanup tmp images')
        if self._finished.value == 0:
            self._finished.value = 1

        start = time.time()
        elapsed = 0
        while elapsed < 1.0 and self._finished.value != 2:
            self.logger.info('waiting for exitting cleaner loop, finished = {}'.format(self._finished.value))
            time.sleep(0.001)
            elapsed = time.time() - start

        return self._finished.value == 2

def main():
    print(sys.argv)
    video_src = sys.argv[1]
    
    tengu = Tengu(video_src)
    
if __name__ == '__main__':
    main()
