import serial
import time
import threading

class WaveshareUSBCAN:
    def __init__(self, port, uart_baud=2_000_000, can_baud=0x01):
        self.ser = serial.Serial(port, uart_baud, timeout=0.01)
        time.sleep(0.2)

        self._init_adapter(can_baud)

        self._rx_thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = {}   # can_id -> data

    def _checksum(self, data):
        return sum(data[2:]) & 0xFF

    def _init_adapter(self, can_baud):
        pkt = [
            0xAA, 0x55,
            0x12,          # variable protocol
            can_baud,      # CAN baud
            0x01,          # STANDARD frame
            0x00, 0x00, 0x00, 0x00,  # filter
            0x00, 0x00, 0x00, 0x00,  # mask
            0x00,          # normal mode
            0x00,          # auto resend
            0x00, 0x00, 0x00, 0x00
        ]
        pkt.append(self._checksum(pkt))
        self.ser.write(bytes(pkt))
        time.sleep(0.1)

    # ---------- TX ----------
    def send_can(self, can_id, data):
        dlc = len(data)
        ctrl = 0xC0 | dlc

        pkt = [0xAA, ctrl]
        pkt += list(can_id.to_bytes(2, 'little'))
        pkt += data
        pkt.append(0x55)

        self.ser.write(bytes(pkt))

    # ---------- RX THREAD ----------
    def _rx_loop(self):
        while self._running:
            if self.ser.read(1) != b'\xAA':
                continue

            ctrl = self.ser.read(1)
            if not ctrl:
                continue

            ctrl = ctrl[0]
            dlc = ctrl & 0x0F

            raw = self.ser.read(2 + dlc + 1)
            if len(raw) != 2 + dlc + 1:
                continue

            if raw[-1] != 0x55:
                continue

            can_id = raw[1] << 8 | raw[0]
            data = list(raw[2:2 + dlc])

            with self._lock:
                self._latest[can_id] = data

    def start_rx(self):
        if self._rx_thread:
            return
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def stop_rx(self):
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=0.2)
            self._rx_thread = None

    def get_latest(self, can_id):
        with self._lock:
            return self._latest.get(can_id)

