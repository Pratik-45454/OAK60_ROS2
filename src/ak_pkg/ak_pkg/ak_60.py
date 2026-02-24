#!/usr/bin/env python3

import time
import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile,  QoSReliabilityPolicy, QoSHistoryPolicy

from .Waveshare import WaveshareUSBCAN
from .ak_mit import AKMotorMIT


class AKMotorService(Node):

    def __init__(self):
        super().__init__('ak_60')

        self.connection_lock = threading.Lock()

        self.up_srv = self.create_service(
            SetBool,
            'ak_motor_up_execute',
            self.up_service_callback
        )

        self.down_srv = self.create_service(
            SetBool,
            'ak_motor_down_exec',
            self.down_service_callback
        )

        self.stop_srv = self.create_service(
            SetBool,
            'ak_motor_stop',
            self.stop_callback
        )

        self.device = "/dev/ttyCH341USB0"
        self.can_baud = 0x01
        self.motor_id = 1

        self.dt = 0.002
        self.kp = 12.0
        self.kd = 1.4
        self.threshold = 1.0
        self.timeout = 5.0  # seconds

        self.running = False
        self.stop_requested = False
        self.isFirst = True

        self.get_logger().info("Initializing CAN and AK motor...")

        self.can = None
        self.motor = None
        self.init_can_with_retry()

        self.get_logger().info("Running blocking startup sequence...")
        self.prepare_motor()
        success = self.move_to_blocking(2.5)
        if not success:
            self.get_logger().error("Startup failed (timeout)")
        else:
            self.get_logger().info("Startup completed successfully")

    def init_can_with_retry(self):
        while rclpy.ok():
            try:
                self.can = WaveshareUSBCAN(self.device, can_baud=self.can_baud)
                self.can.start_rx()
                self.motor = AKMotorMIT(self.can, motor_id=self.motor_id)
                time.sleep(1.0)
                return
            except Exception:
                time.sleep(2.0)

    def safe_close_can(self):
        try:
            if self.motor:
                try:
                    self.motor.exit_mit()
                except Exception:
                    pass
            if self.can:
                try:
                    self.can.close()
                except Exception:
                    pass
        except Exception:
            pass

    def reconnect(self):
        with self.connection_lock:
            self.safe_close_can()
            time.sleep(2.0)
            self.init_can_with_retry()

    def prepare_motor(self):
        try:
            self.motor.exit_mit()
            time.sleep(0.01)
            self.motor.zero_position()
            time.sleep(0.01)
            self.motor.enter_mit()
        except Exception:
            self.reconnect()

    def move_to_blocking(self, target):
        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.timeout:
                self.get_logger().error(f"Timeout while moving to {target}")
                try:
                    self.motor.exit_mit()
                except Exception:
                    pass
                return False

            if self.stop_requested:
                self.get_logger().warn("Stop requested during motion")
                try:
                    self.motor.exit_mit()
                except Exception:
                    pass
                return False

            try:
                self.motor.send(
                    p=target,
                    v=0.0,
                    kp=self.kp,
                    kd=self.kd,
                    t=0.0
                )
            except Exception:
                self.reconnect()
                return False

            try:
                fb = self.motor.read()
            except Exception:
                self.reconnect()
                return False

            if fb is not None:
                pos = fb[0]
                if abs(pos - target) < self.threshold:
                    return True

            time.sleep(self.dt)

        return False


    def up_service_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "False request. Nothing executed."
            return response

        if self.running:
            response.success = False
            response.message = "Motor already running."
            return response

        self.running = True
        self.stop_requested = False

        self.get_logger().info("Executing UP sequence (blocking)")

        try:
            if self.isFirst:
                self.prepare_motor()
                if not self.move_to_blocking(-12.5):
                    response.success = False
                    response.message = "UP first move timeout/failure"
                    return response

                self.prepare_motor()
                self.isFirst = False

            success = self.move_to_blocking(-12.5)

            response.success = success
            response.message = "UP completed successfully" if success else "UP failed or timed out"

        finally:
            self.running = False

        return response

    def down_service_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "False request. Nothing executed."
            return response

        if self.running:
            response.success = False
            response.message = "Motor already running."
            return response

        self.running = True
        self.stop_requested = False

        self.get_logger().info("Executing DOWN sequence (blocking)")

        try:
            success = self.move_to_blocking(12.5)

            response.success = success
            response.message = "DOWN completed successfully" if success else "DOWN failed or timed out"

        finally:
            self.running = False

        return response

    def stop_callback(self, request, response):
        if request.data:
            self.stop_requested = True
            try:
                self.motor.exit_mit()
            except Exception:
                pass
            response.success = True
            response.message = "Stop requested"
        else:
            response.success = False
            response.message = "Stop ignored"

        return response

    def destroy_node(self):
        self.safe_close_can()
        super().destroy_node()


def main():
    rclpy.init()
    node = AKMotorService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()