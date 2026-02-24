#!/usr/bin/env python3

import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


class SimpleSerialBridge(Node):

    def __init__(self):
        super().__init__("simple_serial_bridge")

        self.serial_port = serial.Serial(
            port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_54A7002607-if00",
            baudrate=115200,
            timeout=0.01
        )

        self.command_id = 0
        self.value_1 = 0
        self.value_2 = 0

        self.subscription = self.create_subscription(
            UInt8MultiArray,
            "tx_data",
            self.tx_data_callback,
            10
        )

        self.timer = self.create_timer(0.02, self.send_serial)
        self.get_logger().info("simple bridge started")

    def tx_data_callback(self, msg: UInt8MultiArray):
        if len(msg.data) < 3:
            self.get_logger().warn("tx_data is insufficient")
            return

        self.command_id = msg.data[0]
        self.value_1 = msg.data[1]
        self.value_2 = msg.data[2]

    def calculate_crc(self, data: bytes) -> int:
        crc = 0
        for b in data:
            crc ^= b
        return crc

    def build_packet(self) -> bytes:
        START_BYTE = 0xA5

        payload = bytes([
            self.command_id,
            self.value_1,
            self.value_2
        ])

        crc = self.calculate_crc(payload)
        return bytes([START_BYTE]) + payload + bytes([crc])

    def send_serial(self):
        packet = self.build_packet()
        self.serial_port.write(packet)

        self.get_logger().info(f"Sent: {[hex(b) for b in packet]}")


def main():
    rclpy.init()
    node = SimpleSerialBridge()

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

