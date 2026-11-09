import serial
import time
import threading


class WaveshareUSBCAN:
    """
    Driver for the Waveshare USB-CAN-A adapter using the variable-length protocol.

    Variable-length frame format (TX and RX):
      [0xAA] [ctrl] [ID: 2 or 4 bytes, little-endian] [data: 0-8 bytes] [0x55]

    ctrl byte breakdown:
      bit7-6 : always 0b11 (0xC0 base)
      bit5   : 0 = standard frame (11-bit ID, 2 bytes)
               1 = extended frame (29-bit ID, 4 bytes)
      bit4   : 0 = data frame
               1 = remote frame
      bit3-0 : DLC — number of data bytes (0-8)
    """

    def __init__(self, port, uart_baud=2_000_000, can_baud=0x01):
        """
        port      : serial port, e.g. "/dev/ttyUSB0" or "COM3"
        uart_baud : USB-UART baud rate (default 2 Mbps)
        can_baud  : CAN bus speed byte:
                      0x01=1M  0x02=800k  0x03=500k  0x04=400k  0x05=250k
                      0x06=200k  0x07=125k  0x08=100k  0x09=50k
                    CyberGear requires 1 Mbps → 0x01
        """
        self.ser = serial.Serial(port, uart_baud, timeout=0.1)
        time.sleep(0.2)
        self._init_adapter(can_baud)
        self._rx_thread = None
        self._running   = False
        self._lock      = threading.Lock()
        # _latest stores the most recent frame received for each full 29-bit (or 11-bit) CAN ID.
        # Key = integer CAN ID as received on the bus.
        # Value = {"data": list[int], "extended": bool, "remote": bool, "dlc": int}
        self._latest    = {}

    # ─────────────────────────────────────────────
    #  Adapter init
    # ─────────────────────────────────────────────

    def _checksum(self, data):
        """Sum bytes from index 2 onward, keep low byte."""
        return sum(data[2:]) & 0xFF

    def _init_adapter(self, can_baud):
        """
        Send the 20-byte configuration packet to the Waveshare adapter.
        Must be called once before any CAN traffic.
        """
        pkt = [
            0xAA, 0x55,              # fixed header
            0x12,                    # variable-length protocol mode
            can_baud,                # CAN baud rate code
            0x01,                    # frame-type hint (extended; does not restrict TX/RX)
            0x00, 0x00, 0x00, 0x00,  # filter ID — 0 = accept all
            0x00, 0x00, 0x00, 0x00,  # mask ID   — 0 = accept all
            0x00,                    # CAN mode: 0x00=normal
            0x01,                    # auto-retransmit: 0x01=enabled
            0x00, 0x00, 0x00, 0x00,  # reserved, must be 0
        ]
        pkt.append(self._checksum(pkt))
        self.ser.write(bytes(pkt))
        time.sleep(0.1)

    # ─────────────────────────────────────────────
    #  TX
    # ─────────────────────────────────────────────

    def send_can(self, can_id, data, extended=False, remote=False):
        """
        Transmit one CAN frame.

        can_id   : int — 11-bit (standard) or 29-bit (extended)
        data     : list/bytes, 0-8 bytes
        extended : True for 29-bit extended frame (required for CyberGear)
        remote   : True for remote-request frame (no payload)
        """
        dlc = len(data)
        if dlc > 8:
            raise ValueError("CAN payload must be 0-8 bytes")
        if not extended and can_id > 0x7FF:
            raise ValueError("Standard frame ID must be 11-bit (<=0x7FF)")
        if extended and can_id > 0x1FFFFFFF:
            raise ValueError("Extended frame ID must be 29-bit (<=0x1FFFFFFF)")

        ctrl  = 0xC0
        ctrl |= 0x20 if extended else 0x00
        ctrl |= 0x10 if remote   else 0x00
        ctrl |= dlc & 0x0F

        pkt  = bytes([0xAA, ctrl])
        pkt += can_id.to_bytes(4 if extended else 2, "little")
        pkt += bytes(data)
        pkt += bytes([0x55])
        self.ser.write(pkt)

    # ─────────────────────────────────────────────
    #  RX — buffered accumulator
    # ─────────────────────────────────────────────

    def _rx_loop(self):
        """
        Background RX thread using a bytearray accumulator buffer.

        Uses a chunk-based accumulator: locates 0xAA, verifies the complete
        frame is buffered before consuming it, and skips one byte cleanly on a
        framing error so the scan restarts without losing subsequent frames.

        NOTE for CyberGear: feedback frames arrive with a different 29-bit ID
        than the command ID (comm_type changes, master_id appears in bits 23-8).
        We store every frame by its full received CAN ID so the Motor layer can
        scan _latest with its own matching logic.
        """
        buf = bytearray()
        while self._running:
            chunk = self.ser.read(32)
            if not chunk:
                continue
            buf.extend(chunk)

            while True:
                # Locate frame start byte
                start = buf.find(0xAA)
                if start == -1:
                    buf.clear()
                    break
                if start > 0:
                    del buf[:start]      # discard bytes before 0xAA

                if len(buf) < 2:
                    break                # wait for ctrl byte

                ctrl        = buf[1]
                is_extended = bool(ctrl & 0x20)
                dlc         = ctrl & 0x0F
                id_len      = 4 if is_extended else 2
                frame_len   = 2 + id_len + dlc + 1  # AA + ctrl + ID + data + 0x55

                if len(buf) < frame_len:
                    break                # incomplete frame — wait for more bytes

                if buf[frame_len - 1] != 0x55:
                    del buf[0]          # bad end byte — skip this 0xAA and retry
                    continue

                frame     = bytes(buf[:frame_len])
                del buf[:frame_len]

                is_remote = bool(ctrl & 0x10)
                can_id    = int.from_bytes(frame[2:2 + id_len], "little")
                can_data  = list(frame[2 + id_len: 2 + id_len + dlc])

                if is_extended:
                    motor_id  = (can_id >> 8) & 0xFF   # bits 15:8 — motor ID in CyberGear replies
                    comm_type = (can_id >> 24) & 0x1F  # bits 28:24
                    with self._lock:
                        self._latest[motor_id] = {
                            "data":      can_data,
                            "extended":  is_extended,
                            "remote":    is_remote,
                            "dlc":       dlc,
                            "comm_type": comm_type,
                            "raw_id":    can_id,
                        }
                else:
                    with self._lock:
                        self._latest[can_id] = {
                            "data":     can_data,
                            "extended": is_extended,
                            "remote":   is_remote,
                            "dlc":      dlc,
                        }

    def start_rx(self):
        """Start background RX thread. Safe to call multiple times."""
        if self._rx_thread:
            return
        self._running   = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def stop_rx(self):
        """Stop background RX thread."""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=0.5)
            self._rx_thread = None

    def get_latest(self, can_id):
        """
        Return the most recent frame received for the given ID, or None.

        For standard frames: key is the 11-bit CAN ID.
        For extended frames (CyberGear): key is motor_id (bits 7:0 of the 29-bit ID).
        Use raw_id from the result dict if you need the full 29-bit ID.
        """
        with self._lock:
            return self._latest.get(can_id)

    def get_latest_snapshot(self):
        """Return a shallow copy of the full _latest dict (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    # ─────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────

    def close(self):
        self.stop_rx()
        self.ser.close()

    def __enter__(self):
        self.start_rx()
        return self

    def __exit__(self, *_):
        self.close()