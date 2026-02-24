#!/usr/bin/env python3

import time
import serial

import rclpy
from rclpy.node import Node

from oakd_roi_detector_interfaces.srv import GetDetections

START_BYTE = 0xA5


class DetectionSerialBridge(Node):

    def __init__(self):
        super().__init__("detection_serial_bridge")

        # ---------------- SERIAL ----------------
        self.serial_port = serial.Serial(
            port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_54A7002607-if00",
            baudrate=115200,
            timeout=0.01
        )

        # ---------------- SERVICE CLIENT ----------------
        self.client = self.create_client(GetDetections, "/get_detections")

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /get_detections service...")

        self.get_logger().info("Service available")

        # ---------------- STATE ----------------
        self.stage = "RESET_TRUE"
        self.last_call_time = self.get_clock().now()

        # ---------------- TIMER ----------------
        self.timer = self.create_timer(0.1, self.state_machine)

    # ==================================================
    # STATE MACHINE
    # ==================================================
    def state_machine(self):
        now = self.get_clock().now()

        if self.stage == "RESET_TRUE":
            self.call_service(reset=True)
            self.stage = "WAIT_RESET"
            self.last_call_time = now

        elif self.stage == "WAIT_RESET":
            if (now - self.last_call_time).nanoseconds > 5e8:  # 0.5 sec
                self.stage = "RESET_FALSE"

        elif self.stage == "RESET_FALSE":
            self.call_service(reset=False)
            self.stage = "WAIT_RESPONSE"

        elif self.stage == "DONE":
            pass  # nothing more to do

    # ==================================================
    # SERVICE CALL
    # ==================================================
    def call_service(self, reset: bool):
        req = GetDetections.Request()
        req.reset = reset
        req.trigger_roi = ""

        future = self.client.call_async(req)
        future.add_done_callback(self.handle_response)

    # ==================================================
    # SERVICE RESPONSE
    # ==================================================
    def handle_response(self, future):
        if self.stage != "WAIT_RESPONSE":
            return  # ignore first reset response

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return

        if not response.success:
            self.get_logger().error(f"Detection failed: {response.status}")
            return

        self.get_logger().info(
            f"Detections: ROI1={response.roi1_class}, ROI2={response.roi2_class}"
        )

        packet = self.build_packet(response)
        self.serial_port.write(packet)

        self.get_logger().info(f"Sent packet: {[hex(b) for b in packet]}")

        self.stage = "DONE"

    # ==================================================
    # PACKET BUILDER
    # ==================================================
    def build_packet(self, response) -> bytes:
        roi1 = 0xFF if response.roi1_class == "real" else 0x00
        roi2 = 0xFF if response.roi2_class == "real" else 0x00

        payload = bytes([roi1, roi2])
        crc = self.calculate_crc(payload)

        return bytes([START_BYTE]) + payload + bytes([crc])

    # ==================================================
    # CRC (simple XOR)
    # ==================================================
    def calculate_crc(self, data: bytes) -> int:
        crc = 0
        for b in data:
            crc ^= b
        return crc


def main():
    rclpy.init()
    node = DetectionSerialBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.serial_port.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

