#!/usr/bin/env python3

import threading
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from oakd_roi_detector_interfaces.srv import GetDetections

# ----------------- ROIs (adjusted for typical 1280x720 webcam feed) -----------------
# Feel free to adjust these coordinates to match your webcam resolution and desired areas
ROI_1 = ((200, 200), (600, 550))   # Left ROI
ROI_2 = ((680, 200), (1080, 550))  # Right ROI

# ----------------- States -----------------
IDLE = 0
ROI1_ACTIVE = 1
ROI1_LOCKED = 2
ROI2_ACTIVE = 3
ROI2_LOCKED = 4

def crop_roi(frame, roi):
    (x1, y1), (x2, y2) = roi
    return frame[y1:y2, x1:x2], x1, y1

class WebcamDetectorNode(Node):

    def __init__(self):
        super().__init__('webcam_roi_detector')

        # Load YOLO model
        package_name = 'oakd_roi_detector'  # Keep name or change if you rename package
        model_path = os.path.join(
            get_package_share_directory(package_name),
            'models',
            'best.pt'
        )
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found at {model_path}")
            raise FileNotFoundError(model_path)

        self.model = YOLO(model_path)
        self.get_logger().info("YOLO model loaded")

        # State variables
        self.state = IDLE
        self.roi1_class = ""
        self.roi2_class = ""

        # Debug display
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.bridge = CvBridge()

        # Service
        self.srv = self.create_service(
            GetDetections, '/get_detections', self.get_detections_callback)

        # Webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open webcam")
            raise RuntimeError("Webcam not accessible")

        # Set resolution (optional — adjust to your camera capability)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.get_logger().info("Webcam opened successfully")

        # Start debug display thread
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()
        self.get_logger().info("Debug display thread started")

        # Start processing timer (~30-60 Hz is enough for webcam)
        self.create_timer(0.033, self.processing_loop)  # ~30 Hz

        # Begin detection
        self.state = ROI1_ACTIVE
        self.get_logger().info("Started in ROI1_ACTIVE state")

    def processing_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        # Optional: resize if needed (uncomment if your cam gives different size)
        # frame = cv2.resize(frame, (1280, 720))

        # Prepare debug frame
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (200, 200), (600, 550), (0, 255, 255), 3)
        cv2.putText(debug_frame, "ROI1", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.rectangle(debug_frame, (680, 200), (1080, 550), (255, 255, 0), 3)
        cv2.putText(debug_frame, "ROI2", (680, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # State text
        state_text = {
            ROI1_ACTIVE: "STATE: Waiting for ROI1",
            ROI1_LOCKED: "STATE: ROI1 Locked",
            ROI2_ACTIVE: "STATE: Waiting for ROI2",
            ROI2_LOCKED: "STATE: Both Locked"
        }.get(self.state, "STATE: Unknown")
        cv2.putText(debug_frame, state_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store for display thread
        with self.frame_lock:
            self.latest_frame = debug_frame.copy()

        # Detection logic
        if self.state == ROI1_ACTIVE:
            roi_img, _, _ = crop_roi(frame, ROI_1)
            results = self.model(roi_img, stream=False, verbose=False)
            for r in results:
                if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                    cls_id = int(r.obb.cls[0])
                    cls_name = self.model.names[cls_id]
                    self.roi1_class = cls_name
                    self.get_logger().info(f"ROI 1 LOCKED: {cls_name}")
                    self.state = ROI1_LOCKED
                    self.state = ROI2_ACTIVE  # Auto transition
                    break

        elif self.state == ROI2_ACTIVE:
            roi_img, _, _ = crop_roi(frame, ROI_2)
            results = self.model(roi_img, stream=False, verbose=False)
            for r in results:
                if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                    cls_id = int(r.obb.cls[0])
                    cls_name = self.model.names[cls_id]
                    self.roi2_class = cls_name
                    self.get_logger().info(f"ROI 2 LOCKED: {cls_name}")
                    self.state = ROI2_LOCKED
                    break

    def display_loop(self):
        cv2.namedWindow("Debug View - Webcam", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug View - Webcam", 1280, 720)

        while rclpy.ok():
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_show = self.latest_frame.copy()
                else:
                    frame_to_show = None

            if frame_to_show is not None:
                cv2.imshow("Debug View - Webcam", frame_to_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit key pressed in debug window")
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def get_detections_callback(self, request, response):
        # Handle reset
        if request.reset:
            self.state = ROI1_ACTIVE
            self.roi1_class = ""
            self.roi2_class = ""
            self.get_logger().info("State machine RESET via service call")

        # Handle trigger_roi
        if request.trigger_roi.lower() == "roi1":
            self.state = ROI1_ACTIVE
            self.get_logger().info("Forced to ROI1_ACTIVE via service")
        elif request.trigger_roi.lower() == "roi2":
            self.state = ROI2_ACTIVE
            self.get_logger().info("Forced to ROI2_ACTIVE via service")

        # Fill response
        if self.state == ROI2_LOCKED:
            response.roi1_class = self.roi1_class
            response.roi2_class = self.roi2_class
            response.success = True
            response.status = "both_locked"
        else:
            response.roi1_class = ""
            response.roi2_class = ""
            response.success = False
            if self.state == ROI1_ACTIVE:
                response.status = "waiting_for_roi1"
            elif self.state == ROI2_ACTIVE:
                response.status = "waiting_for_roi2"
            else:
                response.status = "idle_or_error"

        return response

    def destroy_node(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamDetectorNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()