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
from Queue import Full, Empty

class Tengu(object):

    """ A main program to track and count objects

    First of all, Tengu manages the following three processes:
    1. Camera Reader
    2. Temp Image Cleaner
    3. Scene Model

    And communications between processes are done by queues.
    A. output_event_queue[CameraReader => SceneModel => GUI]: various events are raised for visualization and debugging
    B. input_event_queue[GUI => CameraReader(=> SceneModel)]: various control events from GUI(a user) is passed

    """

    # dictionary to set camera settings
    # e.g. width {3, 1280}
    PREFERRED_CAMERA_SETTINGS = None

    # event keys
    EVENT_FRAME_SHAPE = 'event_frame_shape'
    EVENT_FRAME_NO = 'event_frame_no'
    EVENT_FRAME_CROPPED = 'event_frame_cropped'

    EVENT_DETECTIONS = 'event_detections'
    EVENT_DETECTION_CLASSES = 'event_detection_classes'

    EVENT_TRACKLETS = 'event_tracklets'
    EVENT_FLOW_NODES = 'event_flow_nodes'
    EVENT_FLOWS = 'event_flows'
    EVENT_COUNTED_TRACKLETS = 'event_counted_tracklets'
    EVENT_SCENE_SAVE = 'event_scene_save'

    EVENT_CAMERA_SRC = 'event_camera_src'
    EVENT_CAMERA_ROI = 'event_camera_roi'
    EVENT_CAMERA_SCALE = 'event_camera_scale'
    EVENT_CAMERA_CHANGED = 'event_camera_changed'
    EVENT_CAMERA_NEEDS_ONLY_FRAME = 'event_camera_only_frame'

    # tmpfs to exchange an image between processes
    TMPFS_DIR = '/dev/shm/tengu'

    def __init__(self, scene_file=None):
        self.logger= logging.getLogger(__name__)

        """ a model that consumes camera frames and produces something meaningful as scene
        """
        # deferred import to prevent a circular import, Tengu -> TenguSceneModel -> Tengu
        from tengu_scene_model import TenguSceneModel
        self._scene_model = TenguSceneModel(scene_file=scene_file)
        self._camera_reader = None
        self._tmp_image_cleaner = None

        """ terminate if not 0 """
        self._stopped = Value('i', 0)

        self._current_model_frame = Value('i', 0)

    @property
    def camera_frame_no(self):
        return self._camera_reader.current_frame

    @property
    def model_frame_no(self):
        return self._current_model_frame.value

    @property
    def detection_interval(self):
        if self._scene_model is None:
            return -1
        return self._scene_model.detection_interval

    def set_detection_interval(self, detection_interval):
        if self._scene_model is None:
            return
        self._scene_model.set_detection_interval(detection_interval)

    
    def start_processes(self, src=None, roi=None, scale=None, every_x_frame=1, rotation=0, skip_to=-1, frame_queue_timeout_in_secs=10, output_event_queue=None, input_event_queue=None, tmpfs_cleanup_interval_in_frames=10*25):
        # check tmpfs directory exists
        if os.path.exists(Tengu.TMPFS_DIR):
            shutil.rmtree(Tengu.TMPFS_DIR)
        os.makedirs(Tengu.TMPFS_DIR)

        # start scene model that spawns Processes
        # IMPORTANT: Fork first, then Thread
        # so this should be called BEFORE creating any threads on a main process
        # start reading camera
        self.logger.info('starting camera reader')
        self._camera_reader = CameraReader(src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, output_event_queue, input_event_queue, self._scene_model.input_queue)
        self._camera_reader.start()
        while self._camera_reader._finished.value == -1:
            # this means not started yet
            self.logger.info('waiting for camera running')
            time.sleep(0.1)

        # start cleaner
        self.logger.info('starting tmp image cleaner')
        # tmp cleanup
        self._tmp_image_cleaner = TmpImageCleaner(tmpfs_cleanup_interval_in_frames)
        self._tmp_image_cleaner.start()
        while self._tmp_image_cleaner._finished.value == -1:
            # this means not started yet
            self.logger.info('waiting for image cleaner running')
            time.sleep(0.1)

        # start scene model
        self.logger.info('starting scene model')
        self._scene_model.start_sensor()
        self._scene_model.start()
        while self._scene_model._finished.value == -1:
            # this means not started yet
            self.logger.info('waiting for scene model running, process alive = {}, exitcode = {}'.format(self._scene_model.is_alive(), self._scene_model.exitcode))
            time.sleep(0.1)

    def stop_processes(self):

        if self._camera_reader is not None:
            if self._camera_reader.finish():
                self._camera_reader.join()
            else:
                raise Exception('camera reader did not stop!')

        if self._scene_model is not None:
            if not self._scene_model.finish():
                self._scene_model.terminate()

        # at this point, only queue that communicates with a client has not yet been cleaned up, and which should be done by the clietn

        if self._tmp_image_cleaner is not None:
            if self._tmp_image_cleaner.finish():
                self._tmp_image_cleaner.join()
            else:
                raise Exception('cleaner did not stop!')

    def run(self, frame_queue_timeout_in_secs=10, output_event_queue=None, tmpfs_cleanup_interval_in_frames=10*25, **kwargs):

        try:
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
                    except Empty:
                        self.logger.debug('failed to get an output dict from a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start

                if output_dict is None:
                    self.logger.error('failed to get any output from an output queue within {} seconds'.format(frame_queue_timeout_in_secs))
                    continue
                else:
                    event_dict[Tengu.EVENT_DETECTIONS] = output_dict['d']
                    event_dict[Tengu.EVENT_DETECTION_CLASSES] = output_dict['c']
                    event_dict[Tengu.EVENT_TRACKLETS] = output_dict['t']
                    event_dict[Tengu.EVENT_FRAME_NO] = output_dict['n']
                    event_dict[Tengu.EVENT_COUNTED_TRACKLETS] = output_dict['ct']

                    scene = output_dict['s']
                    event_dict[Tengu.EVENT_FLOW_NODES] = scene['flow_nodes']
                    event_dict[Tengu.EVENT_FLOWS] = scene['flows']


                # put in the queue
                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < frame_queue_timeout_in_secs and self._stopped.value == 0:
                    try:
                        self.logger.debug('putting an event dict to a queue')
                        output_event_queue.put_nowait(event_dict)
                        done = True
                    except Full:
                        self.logger.debug('failed to put an event dict in a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start

                if not done:
                    if self._stopped.value == 1:
                        self.logger.info('breaking tengu loop...')
                    else:
                        self.logger.error('failed to put an output dict in a queue within {} seconds'.format(frame_queue_timeout_in_secs))
                    break

                self._current_model_frame.value = event_dict[Tengu.EVENT_FRAME_NO]
                self.logger.debug('analyzed frame no {} in {} s'.format(self._current_model_frame.value, (time.time() - frame_analysis_start)))

                # update time of tmp cleaner
                while self._tmp_image_cleaner.current_frame < self._camera_reader.current_frame:
                    self.logger.debug('incrementing cleaner current frame {}, camera reader {}'.format(self._tmp_image_cleaner.current_frame, self._camera_reader.current_frame))
                    self._tmp_image_cleaner.increment_current_frame()

        except:
            info = sys.exc_info()
            self.logger.exception('Unknow Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:
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


class CameraReader(Process):
    def __init__(self, video_src, roi, scale, every_x_frame, rotation, skip_to, frame_queue_timeout_in_secs, output_event_queue, input_event_queue, scene_input_queue, **kwargs):
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
        self._output_event_queue = output_event_queue
        self._input_event_queue = input_event_queue
        self._scene_input_queue = scene_input_queue
        # transient
        self._cam = None
        self._current_frame = Value('i', 0)
        self._finished = Value('i', -1)
        self._return_only_frame = False

    @property
    def video_src(self):
        return self._video_src

    @property
    def roi(self):
        return self._roi

    @property
    def scale(self):
        return self._scale

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
            self.logger.error('{} is not available'.format(self._video_src))

        if Tengu.PREFERRED_CAMERA_SETTINGS is not None:
            for key in Tengu.PREFERRED_CAMERA_SETTINGS:
                self.logger.info('set a user defined camera setting {} of {}'.format(Tengu.PREFERRED_CAMERA_SETTINGS[key], key))
                self._cam.set(key, Tengu.PREFERRED_CAMERA_SETTINGS[key])

    def run(self):
        self.logger.info('running camera reader')
        self._finished.value = 0
        # read frames
        try:
            while self._finished.value == 0:

                has_changed = False
                save_path = None

                if self._input_event_queue is not None:
                    # check if camera settings are passed as events
                    try:
                        event = self._input_event_queue.get_nowait()
                        self.logger.info('new event found {}'.format(event))
                        # ok, found an event
                        for key in event:
                            if key == Tengu.EVENT_CAMERA_SRC:
                                new_src = event[Tengu.EVENT_CAMERA_SRC]
                                self.logger.info('video_src is switching to {} from {}'.format(new_src, self._video_src))
                                self._video_src = new_src
                                has_changed = True
                            elif key == Tengu.EVENT_CAMERA_ROI:
                                new_roi = event[Tengu.EVENT_CAMERA_ROI]
                                # make sure roi is int
                                new_roi = [int(new_roi[0]), int(new_roi[1]), int(new_roi[2]), int(new_roi[3])]
                                self.logger.info('roi is changed to {}'.format(new_roi))
                                self._roi = new_roi
                                has_changed = True
                            elif key == Tengu.EVENT_CAMERA_SCALE:
                                new_scale = event[Tengu.EVENT_CAMERA_SCALE]
                                self.logger.info('scale is changed to {}'.format(new_scale))
                                self._scale = new_scale
                                has_changed = True
                            elif key == Tengu.EVENT_CAMERA_NEEDS_ONLY_FRAME:
                                needs_only_frame = event[Tengu.EVENT_CAMERA_NEEDS_ONLY_FRAME]
                                self.logger.info('camera will return only frame? = {}'.format(needs_only_frame))
                                self._return_only_frame = needs_only_frame
                            elif key == Tengu.EVENT_SCENE_SAVE:
                                save_path = event[Tengu.EVENT_SCENE_SAVE]
                                self.logger.info('will forward an event to save a scene to {}...'.format(save_path))
                            else:
                                self.logger.info('{} is not a known event key for camera'.format(key))

                        if has_changed and self._video_src is not None:
                            self.logger.info('resetting camera...')
                            self._current_frame.value = 0
                            self.setup()
                        else:
                            # this is strange
                            self.logger.info('keys = {}'.format(event.keys()))
                            self.logger.info('values = {}'.format(event.values()))

                    except Empty:
                        # no event
                        pass
                    except:
                        info = sys.exc_info()
                        self.logger.exception('Unknown Exception {}, {}, {}'.format(info[0], info[1], info[2]))
                        traceback.print_tb(info[2])

                if self._video_src is None:
                    # no src is set yet
                    time.sleep(1)
                    continue

                if self._cam is None:
                    self.setup()

                frame_start = time.time()
                
                start = time.time()
                ret, frame = self._cam.read()
                self.logger.debug('read the next frame in {} s'.format(time.time() - start))

                # finished
                if not ret:
                    self.logger.info('no frame is available')
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
                self.logger.debug('wrote a frame image {} in {} s'.format(img_path, time.time() - start))

                event_dict[Tengu.EVENT_FRAME_SHAPE] = [frame.shape[0], frame.shape[1]]
                event_dict[Tengu.EVENT_FRAME_CROPPED] = img_path
                event_dict[Tengu.EVENT_CAMERA_CHANGED] = has_changed

                # put in a queue, this is shared with a GUI client
                done = False
                start = time.time()
                elapsed = 0
                while not done and elapsed < self._frame_queue_timeout_in_secs and self._finished.value == 0:
                    try:
                        self.logger.debug('putting a frame image path {} in a queue'.format(img_path))
                        self._output_event_queue.put_nowait(event_dict)
                        done = True
                    except Full:
                        self.logger.debug('failed to put event dict in a queue, sleeping')
                        time.sleep(0.001)
                    elapsed = time.time() - start
                self.logger.debug('took {} to put a frame image path in a queue'.format(elapsed))

                # skip if necessary
                if (self._every_x_frame > 1 and self._current_frame.value % self._every_x_frame != 0) or (self._skip_to > 0 and self._current_frame.value < self._skip_to):

                    self.logger.debug('skipping frame at {}'.format(self._current_frame.value))
                    continue

                # put in a scene input queue, this is shared with scene model
                if not self._return_only_frame:
                    done = False
                    start = time.time()
                    elapsed = 0
                    while not done and elapsed < self._frame_queue_timeout_in_secs and self._finished.value == 0:
                        try:
                            self.logger.debug('putting a frame image path {} in a scene model queue'.format(img_path))
                            if save_path is not None:
                                event_dict[Tengu.EVENT_SCENE_SAVE] = save_path
                            self._scene_input_queue.put_nowait(event_dict)
                            done = True
                        except Full:
                            self.logger.debug('failed to put {} in a queue, sleeping'.format(event_dict))
                            time.sleep(0.001)
                        elapsed = time.time() - start
                    self.logger.debug('took {} to put a frame image path in a scene queue'.format(elapsed))

                self.logger.debug('put frame img and its path at time {} in {} s'.format(self._current_frame.value, time.time() - frame_start))

                # increment
                self._current_frame.value += 1
        except:
            info = sys.exc_info()
            self.logger.exception('Unknown Exception {}, {}, {}'.format(info[0], info[1], info[2]))
            traceback.print_tb(info[2])
        finally:
            # cleanup queues
            self.logger.info('cleaning up queue')
            self._output_event_queue.close()
            while not self._output_event_queue.empty():
                self._output_event_queue.get_nowait()
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

        self.logger.debug('preprocessing done with {}, {} in {} s'.format(roi, scale, time.time() - start))
        
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
        self._tmpfs_cleanup_interval_in_frames = tmpfs_cleanup_interval_in_frames
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
            #self.logger.info('frame no of {} is {}'.format(file, img_frame_no))
            if (self.current_frame - img_frame_no) > self._tmpfs_cleanup_interval_in_frames:
                self.logger.debug('removing tmp image file {}'.format(file))
                os.remove(os.path.join(Tengu.TMPFS_DIR, file))
            else:
                self.logger.debug('keep tmp image file {}, {} - {} <= {}'.format(file, self.current_frame, img_frame_no, self._tmpfs_cleanup_interval_in_frames))
        self.logger.info('cleaned up tmp images at time {} in {} ms'.format(self.current_frame, time.time() - start))

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
