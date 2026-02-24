#!/usr/bin/env python3

import threading
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
LOCK_FRAMES = 2

COLOR_ROI_COUNT = 6
COLOR_ROI_SIZE = 154
COLOR_ROI_Y = 750
COLOR_ROI_X_START = 729
COLOR_ROI_X_GAP = 270
ROI_SHIFT_CONST = 232

COLOR_PIXEL_RATIO_THRESHOLD = 0.8  # black/brown ratio threshold


MANUAL_MARTIAL_ROIS = {
    5: [((700,700),(854,854)), ((980,700),(1134,854)), ((1260,700),(1414,854)),
        ((1540,700),(1694,854)), ((1820,700),(1974,854)), ((2100,700),(2254,854))],
    6: [((710,710),(864,864)), ((990,710),(1144,864)), ((1270,710),(1424,864)),
        ((1550,710),(1704,864)), ((1830,710),(1984,864)), ((2110,710),(2264,864))],
    7: [((720,720),(874,874)), ((1000,720),(1154,874)), ((1280,720),(1434,874)),
        ((1560,720),(1714,874)), ((1840,720),(1994,874)), ((2120,720),(2274,874))],
    8: [((730,730),(884,884)), ((1010,730),(1164,884)), ((1290,730),(1444,884)),
        ((1570,730),(1724,884)), ((1850,730),(2004,884)), ((2130,730),(2284,884))],
    9: [((740,740),(894,894)), ((1020,740),(1174,894)), ((1300,740),(1454,894)),
        ((1580,740),(1734,894)), ((1860,740),(2014,894)), ((2140,740),(2294,894))],
    10:[((750,750),(904,904)), ((1030,750),(1184,904)), ((1310,750),(1464,904)),
        ((1590,750),(1744,904)), ((1870,750),(2024,904)), ((2150,750),(2304,904))],
}


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
        self.state = ROI1_ACTIVE
        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"
        self.last_stage = None
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()

        self.color_state = 0
        self.color_locked = [False]*COLOR_ROI_COUNT
        self.color_result = ["no"]*COLOR_ROI_COUNT
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

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

    def full_reset(self):
        self.state = ROI1_ACTIVE
        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()

        self.color_locked = [False]*COLOR_ROI_COUNT
        self.color_result = ["no"]*COLOR_ROI_COUNT
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

        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h))
        D = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K,D,np.eye(3),newK,(w,h),cv2.CV_16SC2)

        self.queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.create_timer(0.04, self.processing_loop)

    def processing_loop(self):
        msg = self.queue.tryGet()
        if msg is None:
            return

        frame = msg.getCvFrame()
        frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        debug_frame = frame.copy()

        if self.stage <= 4:
            elapsed = time.monotonic() - self.roi_start_time
            if elapsed > ROI_TIMEOUT:
                self.advance_state()
                return

            roi = {ROI1_ACTIVE: ROI_1, ROI2_ACTIVE: ROI_2, ROI3_ACTIVE: ROI_3}[self.state][self.stage-1]
            roi_img = crop_roi(frame, roi)
            results = self.model(roi_img, verbose=False)
            detected = None
            for r in results:
                detected = self.get_best_class(r)

            for idx, coords in enumerate((ROI_1[self.stage-1], ROI_2[self.stage-1], ROI_3[self.stage-1]), start=1):
                (x1, y1), (x2, y2) = coords
                color = (0,255,255) if idx==1 else (255,255,0) if idx==2 else (255,0,255)
                cv2.rectangle(debug_frame, (x1,y1),(x2,y2), color,3)
                cv2.putText(debug_frame,f"ROI{idx}",(x1,y1-15),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

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
            rois = MANUAL_MARTIAL_ROIS.get(self.stage, [None]*COLOR_ROI_COUNT)
            for i, roi in enumerate(rois):
                if roi is None:
                    continue
                (x1,y1),(x2,y2) = roi
                color = (0,255,255) if i==self.color_state else (0,0,255) if self.color_locked[i] else (0,255,0)
                cv2.rectangle(debug_frame,(x1,y1),(x2,y2),color,3)
                cv2.putText(debug_frame,f"CROI{i+1}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

            active_roi = rois[self.color_state]
            if active_roi is not None and not self.color_locked[self.color_state]:
                elapsed = time.monotonic()-self.color_start_time
                if elapsed > ROI_TIMEOUT:
                    self.advance_color_state()
                else:
                    roi_img = crop_roi(frame, active_roi)
                    if roi_img is not None and roi_img.size > 0:
                        detected = self.detect_black_brown_ratio(roi_img)
                    else:
                        detected = False
                    if detected == self.color_last:
                        self.color_count += 1
                    else:
                        self.color_last = detected
                        self.color_count = 1
                    if self.color_count >= LOCK_FRAMES and detected:
                        self.color_result[self.color_state] = "yes"
                        self.color_locked[self.color_state] = True
                        self.advance_color_state()

        with self.frame_lock:
            self.latest_frame = debug_frame.copy()

    def advance_state(self):
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()
        self.state = {ROI1_ACTIVE: ROI2_ACTIVE, ROI2_ACTIVE: ROI3_ACTIVE, ROI3_ACTIVE: ROI1_ACTIVE}[self.state]

    def advance_color_state(self):
        self.color_count = 0
        self.color_last = None
        self.color_start_time = time.monotonic()
        self.color_state = (self.color_state+1)%COLOR_ROI_COUNT

    def get_best_class(self, r):
        if r.obb is None or len(r.obb)==0:
            return None
        confs = r.obb.conf.cpu().numpy()
        classes = r.obb.cls.cpu().numpy().astype(int)
        return self.model.names[classes[np.argmax(confs)]]

    def detect_black_brown_ratio(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,60])
        lower_brown = np.array([5,50,50])
        upper_brown = np.array([20,255,200])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        black_pixels = np.count_nonzero(mask_black)
        brown_pixels = np.count_nonzero(mask_brown)
        if brown_pixels==0:
            return False
        ratio = black_pixels/brown_pixels
        return ratio >= COLOR_PIXEL_RATIO_THRESHOLD

    def display_loop(self):
        cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
        while rclpy.ok():
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is not None:
                cv2.imshow("Debug View", frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
        cv2.destroyAllWindows()

    def get_detections_callback(self, request, response):
        if request.reset:
            self.full_reset()
        if self.last_stage != request.stage:
            self.full_reset()
            start_index = (request.stage-5)%6
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


if __name__=='__main__':
    main()