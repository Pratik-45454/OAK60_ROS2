#!/usr/bin/env python3

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from oakd_roi_detector_interfaces.srv import GetDetections



ROI_1 = (
    ((174, 519), (368, 663)),
    ((174, 519), (368, 663)),
    ((174, 519), (368, 663)),
    ((174, 519), (368, 663)),
)

ROI_2 = (
    ((505, 574), (672, 707)),
    ((505, 574), (672, 707)),
    ((505, 574), (672, 707)),
    ((505, 574), (672, 707)),
)

ROI_3 = (
    ((847, 499), (991, 641)),
    ((847, 499), (991, 641)),
    ((847, 499), (991, 641)),
    ((847, 499), (991, 641)),
)

ROI1_ACTIVE = 1
ROI2_ACTIVE = 2
ROI3_ACTIVE = 3

ROI_TIMEOUT = 0.4
ROI_TIMEOUT_SECOND = 0.20
LOCK_FRAMES = 2



COLOR_ROI_COUNT = 6
COLOR_ROI_SIZE = 120
COLOR_ROI_Y = 520
COLOR_ROI_X_START = 300
COLOR_ROI_X_GAP = 180
ROI_SHIFT_CONST = 40
COLOR_PIXEL_THRESHOLD = 0.01


def crop_roi(frame, roi):
    (x1, y1), (x2, y2) = roi
    return frame[y1:y2, x1:x2]


class OAKDDetectorNode(Node):

    def __init__(self):
        super().__init__('oakd_roi_detector')

        model_path = os.path.join(
            get_package_share_directory('oakd_roi_detector'),
            'models',
            'best.pt'
        )
        self.model = YOLO(model_path)

        self.stage = 1
        self.last_stage = None

        self.state = ROI1_ACTIVE

        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"

        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()

        self.color_state = 0

        self.color_locked = [False] * COLOR_ROI_COUNT
        self.color_result = ["no"] * COLOR_ROI_COUNT

        self.color_count = 0
        self.color_last = None
        self.color_start_time = time.monotonic()

        self.srv = self.create_service(
            GetDetections,
            '/get_detections',
            self.get_detections_callback
        )
        self.pipeline = dai.Pipeline()
        self.setup_pipeline()
        self.start_device()


    def full_reset(self):

        self.state = ROI1_ACTIVE
        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()

        self.color_locked = [False] * COLOR_ROI_COUNT
        self.color_result = ["no"] * COLOR_ROI_COUNT
        self.color_state = 0
        self.color_count = 0
        self.color_last = None
        self.color_start_time = time.monotonic()


    def setup_pipeline(self):
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(1920, 1080)
        cam.setInterleaved(False)

        ctrl = dai.CameraControl()
        ctrl.setContrast(1)
        ctrl.setSaturation(1)
        ctrl.setSharpness(1)
        ctrl.setLumaDenoise(1)
        ctrl.setChromaDenoise(4)
        cam.initialControl.set(ctrl.get())

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.video.link(xout.input)

    def start_device(self):
        self.device = dai.Device(self.pipeline)

        calib = self.device.readCalibration()
        w, h = 1920, 1080

        K = np.array(calib.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, w, h))
        D = np.array(calib.getDistortionCoefficients(
            dai.CameraBoardSocket.RGB))

        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2
        )

        self.queue = self.device.getOutputQueue(
            "rgb", maxSize=4, blocking=False
        )

        self.create_timer(0.04, self.processing_loop)


    def processing_loop(self):

        msg = self.queue.tryGet()
        if msg is None:
            return

        frame = msg.getCvFrame()
        frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        if self.stage <= 4:

            if time.monotonic() - self.roi_start_time > ROI_TIMEOUT:
                self.advance_state()
                return

            if self.state == ROI1_ACTIVE:
                roi = ROI_1[self.stage - 1]
            elif self.state == ROI2_ACTIVE:
                roi = ROI_2[self.stage - 1]
            else:
                roi = ROI_3[self.stage - 1]

            roi_img = crop_roi(frame, roi)
            results = self.model(roi_img, verbose=False)

            detected = None
            for r in results:
                detected = self.get_best_class(r)

            if detected is None:
                self.roi_count = 0
                self.roi_last = None
            else:
                if detected == self.roi_last:
                    self.roi_count += 1
                else:
                    self.roi_last = detected
                    self.roi_count = 1

                if self.roi_count >= LOCK_FRAMES:
                    if self.state == ROI1_ACTIVE:
                        self.roi1_class = detected
                        self.state = ROI2_ACTIVE
                    elif self.state == ROI2_ACTIVE:
                        self.roi2_class = detected
                        self.state = ROI3_ACTIVE
                    else:
                        self.roi3_class = detected
                        self.state = ROI1_ACTIVE

                    self.roi_count = 0
                    self.roi_last = None
                    self.roi_start_time = time.monotonic()

        elif 5 <= self.stage <= 10:

            rois = [self.get_dynamic_color_roi(i)
                    for i in range(COLOR_ROI_COUNT)]

            active_roi = rois[self.color_state]

            if active_roi is None:
                self.advance_color_state()

            elif not self.color_locked[self.color_state]:

                if time.monotonic() - self.color_start_time > ROI_TIMEOUT_SECOND:
                    self.advance_color_state()
                else:
                    roi_img = crop_roi(frame, active_roi)
                    detected = self.detect_red_color(roi_img)

                    if detected == self.color_last:
                        self.color_count += 1
                    else:
                        self.color_last = detected
                        self.color_count = 1

                    if self.color_count >= LOCK_FRAMES and detected:
                        self.color_result[self.color_state] = "yes"
                        self.color_locked[self.color_state] = True
                        self.advance_color_state()

    def advance_state(self):
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()
        self.state = {
            ROI1_ACTIVE: ROI2_ACTIVE,
            ROI2_ACTIVE: ROI3_ACTIVE,
            ROI3_ACTIVE: ROI1_ACTIVE
        }[self.state]

    def advance_color_state(self):
        self.color_count = 0
        self.color_last = None
        self.color_start_time = time.monotonic()
        self.color_state = (self.color_state + 1) % COLOR_ROI_COUNT

    def get_dynamic_color_roi(self, idx):
        shift = int((self.stage - 7.5) * ROI_SHIFT_CONST)

        x1 = COLOR_ROI_X_START + shift + idx * COLOR_ROI_X_GAP
        y1 = COLOR_ROI_Y
        x2 = x1 + COLOR_ROI_SIZE
        y2 = y1 + COLOR_ROI_SIZE

        if x1 < 0 or y1 < 0 or x2 > 1920 or y2 > 1080:
            return None

        return ((x1, y1), (x2, y2))

    def detect_red_color(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / mask.size
        return ratio >= COLOR_PIXEL_THRESHOLD

    def get_best_class(self, r):
        if r.obb is None or len(r.obb) == 0:
            return None
        confs = r.obb.conf.cpu().numpy()
        classes = r.obb.cls.cpu().numpy().astype(int)
        return self.model.names[classes[np.argmax(confs)]]

    def get_detections_callback(self, request, response):

        if request.reset:
            self.full_reset()

        if self.last_stage != request.stage:
            self.full_reset()

            # Set cycling start based on stage
            start_index = (request.stage - 5) % 6
            self.color_state = start_index

        self.stage = request.stage
        self.last_stage = request.stage

        response.roi1_class = self.roi1_class
        response.roi2_class = self.roi2_class
        response.roi3_class = self.roi3_class

        response.roi1_martial = self.color_result[0]
        response.roi2_martial = self.color_result[1]
        response.roi3_martial = self.color_result[2]
        response.roi4_martial = self.color_result[3]
        response.roi5_martial = self.color_result[4]
        response.roi6_martial = self.color_result[5]

        response.success = True
        response.status = "completed"

        return response


def main(args=None):
    rclpy.init(args=args)
    node = OAKDDetectorNode()
    executor = SingleThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
