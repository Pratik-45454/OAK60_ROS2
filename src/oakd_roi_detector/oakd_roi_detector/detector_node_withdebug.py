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


# ROI_1 = (
#     ((785, 436), (940, 596)),
#     ((524, 712), (642, 845)),
#     ((483, 615), (634, 786)),
#     ((618, 641), (747, 756)),
# )

# ROI_2 = (
#     ((398, 508), (552, 676)),
#     ((965, 489), (1147, 683)),
#     ((953, 514), (1117, 684)),
#     ((0, 0), (0, 0)),
# )

# ROI_3 = (
#     ((26, 427), (190, 621)),
#     ((1542, 359), (1734, 562)),
#     ((1441, 595), (1607, 708)),
#     ((1327, 647), (1440, 720)),
# )

ROI_1 = {
    "red": (
        ((785, 436), (940, 596)),
        ((1495, 319), (1753, 597)),
        ((1472, 588), (1707, 793)),
        ((1287, 607), (1439, 746)),
    ),
    "red": (
        ((100, 100), (200, 200)),
        ((300, 300), (400, 400)),
        ((500, 500), (600, 600)),
        ((700, 700), (800, 800)),
    ),
}

ROI_2 = {
    "red": (
        ((398, 508), (552, 676)),
        ((801, 459), (1037, 719)),
        ((776, 452), (1002, 701)),
        ((0, 0), (0, 0)),
    ),
    "red": (
        ((100, 100), (200, 200)),
        ((300, 300), (400, 400)),
        ((500, 500), (600, 600)),
        ((700, 700), (800, 800)),
    ),
}

ROI_3 = {
    "red": (
        ((26, 427), (190, 621)),
        ((152, 569), (378, 770)),
        ((129, 576), (364, 824)),
        ((506, 606), (632, 715)),
    ),
    "red": (
        ((100, 100), (200, 200)),
        ((300, 300), (400, 400)),
        ((500, 500), (600, 600)),
        ((700, 700), (800, 800)),
    ),
}

ROI1_ACTIVE = 1
ROI2_ACTIVE = 2
ROI3_ACTIVE = 3

ROI_TIMEOUT = 0.4
LOCK_FRAMES = 2


DEPTH_TOLERANCE_FOR_FOREST = 15 


DEPTH_TARGETS_FOR_FOREST = [
    [4.10, 3.55, 3.38],
    [2.32, 2.2, 2.25],
    [2.68, 2.59, 2.73],  
    [3.65, 0.55, 3.53],
]

# DEPTH_TARGETS_FOR_FOREST = [
#     [99.10, 99.55, 99.38],
#     [99.32, 99.2, 99.25],
#     [99.68, 99.59, 99.73],  
#     [99.65, 99.55, 99.53],
# ]


COLOR_PIXEL_RATIO_THRESHOLD = 0.8

# ── Stage 11 ──────────────────────────────────────────────────────────────────
# TODO: replace dummy ROI coordinates with real measured values
STAGE11_ROIS = {
    "red": [
        ((558, 433), (798, 630)),
        ((807, 432), (1056, 630)),
        ((1063, 429), (1321, 630)),
        ((541, 108), (796, 339)),
        ((803, 118), (1053, 332)),
        ((1064, 125), (1323, 328)),
    ],

    "blue": [
        ((100, 100), (200, 200)),
        ((250, 100), (350, 200)),
        ((400, 100), (500, 200)),
        ((100, 250), (200, 350)),
        ((250, 250), (350, 350)),
        ((400, 250), (500, 350)),
    ]
}

STAGE11_DEPTH_TARGETS   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # metres — tune later
STAGE11_DEPTH_TOLERANCE = 100                               # metres — tune later
STAGE11_BLUE_THRESHOLD  = 0.25   # min blue-pixel ratio — tune later
STAGE11_RED_THRESHOLD   = 0.25   # min red-pixel ratio — tune later
# ─────────────────────────────────────────────────────────────────────────────

BLACK_RANGE     = ([25, 0, 0],    [112, 172, 91])
BROWN_RANGE     = ([5, 50, 50],   [20, 255, 200])
RED_RANGE       = ([0, 131, 36],  [4, 255, 255])
GREEN_RANGE     = ([40, 50, 50],  [80, 255, 255])
BLUE_RANGE      = ([100, 50, 50], [130, 255, 255])



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
        # Warm up on CPU — forces memory allocation at startup, not mid-competition
        _ = self.model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False, device='cpu')
        self.stage = 1
        self.state = ROI1_ACTIVE
        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"
        self.last_stage = None
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()

        self.current_color = "red"

        # Stage 11 state — 6 independently-locked ROIs, same pattern as stages 5-10
        self.stage11_locked = [False] * 6
        self.stage11_result = ["no"] * 6
        self.stage11_state  = 0          # index of the currently active ROI
        self.stage11_count  = 0
        self.stage11_last   = None
        self.stage11_start_time = time.monotonic()
        self.stage11_color = "red"       # which color counts as "yes"; set via request.color

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

    def is_roi_outside(self, roi, frame):
        (x1, y1), (x2, y2) = roi
        h, w = frame.shape[:2]
        return x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2

    def full_reset(self):
        self.state = ROI1_ACTIVE

        self.roi1_class = "none"
        self.roi2_class = "none"
        self.roi3_class = "none"

        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()


        self.stage11_locked = [False] * 6
        self.stage11_result = ["no"] * 6
        self.stage11_state  = 0
        self.stage11_count  = 0
        self.stage11_last   = None
        self.stage11_start_time = time.monotonic()

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

        monoL = self.pipeline.create(dai.node.MonoCamera)
        monoR = self.pipeline.create(dai.node.MonoCamera)

        monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)

        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)

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
        self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

        self.create_timer(0.04, self.processing_loop)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_depth(self, depthFrame, roi):
        """Return (depth_m_left, depth_m_right) for a given ROI centre."""
        (x1, y1), (x2, y2) = roi
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        roi_width = x2 - x1
        offset = int(0.10 * roi_width)

        dh, dw = depthFrame.shape
        xl = max(0, min(dw - 1, cx - offset))
        xr = max(0, min(dw - 1, cx + offset))
        cy = max(0, min(dh - 1, cy))

        dl = depthFrame[cy, xl]
        dr = depthFrame[cy, xr]
        return (dl / 1000.0 if dl > 0 else 0.0,
                dr / 1000.0 if dr > 0 else 0.0)

    def _detect_target_color(self, roi_img, target_color, blue_threshold, red_threshold):
        """
        Stage 11 color check. `target_color` ("red" or "blue") selects which
        color counts as "yes"; the opposite color counts as "blocked".
        Returns "no" if neither is found, or if target_color is invalid.
        The target color is checked first — if both somehow trigger, target wins.
        """
        if target_color == "red":
            yes_range, yes_threshold = RED_RANGE, red_threshold
            blocked_range, blocked_threshold = BLUE_RANGE, blue_threshold
        elif target_color == "blue":
            yes_range, yes_threshold = BLUE_RANGE, blue_threshold
            blocked_range, blocked_threshold = RED_RANGE, red_threshold
        else:
            return "no"

        yes_detected = self.detect_color_ratio(
            roi_img,
            colorA_range=yes_range,
            threshold=yes_threshold,
            mode="percent"
        )
        if yes_detected:
            return "yes"

        blocked_detected = self.detect_color_ratio(
            roi_img,
            colorA_range=blocked_range,
            threshold=blocked_threshold,
            mode="percent"
        )
        if blocked_detected:
            return "blocked"

        return "no"

    # ── main processing loop ──────────────────────────────────────────────────

    def processing_loop(self):
        msg = self.queue.tryGet()
        depth_msg = self.depth_queue.tryGet()

        if msg is None or depth_msg is None:
            return

        frame = msg.getCvFrame()
        depthFrame = depth_msg.getCvFrame() 

        frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        debug_frame = frame.copy()

        if self.stage <= 4:
            if self.stage == 4 and self.state == ROI2_ACTIVE:
                self.roi2_class = "none"
                self.state = ROI3_ACTIVE
                self.roi_start_time = time.monotonic()
                return
            elapsed = time.monotonic() - self.roi_start_time
            if elapsed > ROI_TIMEOUT:
                self.advance_state()
            
            else:
                roi = {
                    ROI1_ACTIVE: ROI_1,
                    ROI2_ACTIVE: ROI_2,
                    ROI3_ACTIVE: ROI_3,
                }[self.state][self.current_color][self.stage - 1]
                if self.is_roi_outside(roi, frame):
                    if self.state == ROI1_ACTIVE:
                        self.roi1_class = "outside"
                        self.state = ROI2_ACTIVE
                    elif self.state == ROI2_ACTIVE:
                        self.roi2_class = "outside"
                        self.state = ROI3_ACTIVE
                    else:
                        self.roi3_class = "outside"
                        self.state = ROI1_ACTIVE
                    self.roi_count = 0
                    self.roi_last = None
                    self.roi_start_time = time.monotonic()
                    return 

                roi_img = crop_roi(frame, roi)

                (x1,y1),(x2,y2) = roi

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                roi_width = (x2 - x1)
                offset = int(0.10 * roi_width)

                x_left = cx - offset
                x_right = cx + offset

                h, w = depthFrame.shape
                x_left = max(0, min(w-1, x_left))
                x_right = max(0, min(w-1, x_right))
                cy = max(0, min(h-1, cy))

                depth_mm_left = depthFrame[cy, x_left]
                depth_mm_right = depthFrame[cy, x_right]

                depth_m_left = depth_mm_left / 1000.0 if depth_mm_left > 0 else 0
                depth_m_right = depth_mm_right / 1000.0 if depth_mm_right > 0 else 0

                roi_index = self.state - 1
                target_depth = DEPTH_TARGETS_FOR_FOREST[self.stage-1][roi_index]

                depth_valid = (
                    abs(depth_m_left - target_depth) <= DEPTH_TOLERANCE_FOR_FOREST
                    or
                    abs(depth_m_right - target_depth) <= DEPTH_TOLERANCE_FOR_FOREST
                )

                detected = None
                if depth_valid:
                    results = self.model(roi_img, verbose=False, device='cpu')
                    for r in results:
                        detected = self.get_best_class(r)

                for idx, coords in enumerate(
                    (
                        ROI_1[self.current_color][self.stage - 1],
                        ROI_2[self.current_color][self.stage - 1],
                        ROI_3[self.current_color][self.stage - 1],
                    ),
                    start=1,
                ):
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

        elif self.stage == 11:
            # ── Stage 11: six independently-locked ROIs, red/blue colour ratio ──
            # `self.stage11_color` (set via request.color: "red" or "blue")
            # selects which color returns "yes" and which returns "blocked".

            active_idx = self.stage11_state
            stage11_rois = STAGE11_ROIS[self.current_color]

            # Draw all six ROI boxes
            for i, roi in enumerate(stage11_rois):
                (x1, y1), (x2, y2) = roi
                box_color = (
                    (0, 255, 255) if i == active_idx else        # active  — yellow
                    (0, 0, 255)   if self.stage11_locked[i] else  # locked  — red
                    (0, 255, 0)                                   # waiting — green
                )
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(debug_frame, f"S11_ROI{i+1}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Only process the currently active (unlocked) ROI
            if active_idx < 6 and not self.stage11_locked[active_idx]:
                active_roi = stage11_rois[active_idx]

                if self.is_roi_outside(active_roi, frame):
                    self.stage11_result[active_idx] = "outside"
                    self.stage11_locked[active_idx] = True
                    self._advance_stage11()
                else:
                    elapsed = time.monotonic() - self.stage11_start_time

                    if elapsed > ROI_TIMEOUT:
                        # Timed out without a definitive detection — move on
                        self._advance_stage11()
                    else:
                        (x1, y1), (x2, y2) = active_roi
                        depth_m_left, depth_m_right = self._sample_depth(depthFrame, active_roi)
                        target_depth = STAGE11_DEPTH_TARGETS[active_idx]

                        depth_valid = (
                            abs(depth_m_left  - target_depth) <= STAGE11_DEPTH_TOLERANCE
                            or
                            abs(depth_m_right - target_depth) <= STAGE11_DEPTH_TOLERANCE
                        )

                        cv2.putText(debug_frame,
                                    f"D:{depth_m_left:.2f}m",
                                    (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        roi_img = crop_roi(frame, active_roi)

                        if roi_img is not None and roi_img.size > 0 and depth_valid:
                            detected = self._detect_target_color(
                                roi_img,
                                target_color=self.stage11_color,
                                blue_threshold=STAGE11_BLUE_THRESHOLD,
                                red_threshold=STAGE11_RED_THRESHOLD
                            )
                        else:
                            detected = "no"

                        cv2.putText(debug_frame,
                                    f"Color: {detected}",
                                    (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        if detected == self.stage11_last:
                            self.stage11_count += 1
                        else:
                            self.stage11_last = detected
                            self.stage11_count = 1

                        # Latch on yes or blocked; ignore transient "no"
                        if self.stage11_count >= LOCK_FRAMES and detected != "no":
                            self.stage11_result[active_idx] = detected
                            self.stage11_locked[active_idx] = True
                            self._advance_stage11()

        with self.frame_lock:
            self.latest_frame = debug_frame

    def advance_state(self):
        self.roi_count = 0
        self.roi_last = None
        self.roi_start_time = time.monotonic()
        self.state = {ROI1_ACTIVE: ROI2_ACTIVE, ROI2_ACTIVE: ROI3_ACTIVE, ROI3_ACTIVE: ROI1_ACTIVE}[self.state]



    def _advance_stage11(self):
        self.stage11_count = 0
        self.stage11_last  = None
        self.stage11_start_time = time.monotonic()
        self.stage11_state = (self.stage11_state + 1) % 6

    def get_best_class(self, r):
        if r.obb is None or len(r.obb)==0:
            return None
        confs = r.obb.conf.cpu().numpy()
        classes = r.obb.cls.cpu().numpy().astype(int)
        return self.model.names[classes[np.argmax(confs)]]

    def detect_color_ratio(self, roi_img, colorA_range, colorB_range=None, threshold=0.5, mode="ratio"):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        lowerA, upperA = colorA_range
        maskA = cv2.inRange(hsv, np.array(lowerA), np.array(upperA))
        pixelsA = np.count_nonzero(maskA)

        total_pixels = roi_img.shape[0] * roi_img.shape[1]

        if mode == "pixel":
            return pixelsA >= threshold

        if mode == "percent":
            ratio = pixelsA / total_pixels
            return ratio >= threshold

        if colorB_range is None:
            return False

        lowerB, upperB = colorB_range
        maskB = cv2.inRange(hsv, np.array(lowerB), np.array(upperB))
        pixelsB = np.count_nonzero(maskB)

        if pixelsB == 0:
            return False

        ratio = pixelsA / pixelsB
        return ratio >= threshold

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
        if request.color in ("blue", "red"):
            self.current_color = request.color
            self.stage11_color = request.color
        self.stage = request.stage
        self.last_stage = request.stage

        response.roi1_class = self.roi1_class
        response.roi2_class = self.roi2_class
        response.roi3_class = self.roi3_class
        response.roi1_martial = "none"
        response.roi2_martial = "none"
        response.roi3_martial = "none"
        response.roi4_martial = "none"
        response.roi5_martial = "none"
        response.roi6_martial = "none"
        response.stage11_roi1 = self.stage11_result[0]
        response.stage11_roi2 = self.stage11_result[1]
        response.stage11_roi3 = self.stage11_result[2]
        response.stage11_roi4 = self.stage11_result[3]
        response.stage11_roi5 = self.stage11_result[4]
        response.stage11_roi6 = self.stage11_result[5]
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