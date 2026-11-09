import struct
from .Waveshare import WaveshareUSBCAN
import math


# ─────────────────────────────────────────────────────────────────────────────
#  Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

def float_to_uint(x, xmin, xmax, bits):
    x = max(min(x, xmax), xmin)
    return int((x - xmin) * ((1 << bits) - 1) / (xmax - xmin))


def uint_to_float(x, xmin, xmax, bits):
    return (x / ((1 << bits) - 1)) * (xmax - xmin) + xmin


# ─────────────────────────────────────────────────────────────────────────────
#  MotorMIT
# ─────────────────────────────────────────────────────────────────────────────

class MotorMIT:

    # AK / DM physical limits
    AK_P_MIN,  AK_P_MAX  = -12.5, 12.5
    AK_V_MIN,  AK_V_MAX  = -45.0, 45.0
    AK_T_MIN,  AK_T_MAX  = -15.0, 15.0
    AK_KP_MIN, AK_KP_MAX =   0.0, 500.0
    AK_KD_MIN, AK_KD_MAX =   0.0,   5.0

    # CyberGear physical limits
    CG_P_MIN,  CG_P_MAX  = -12.5, 12.5
    CG_V_MIN,  CG_V_MAX  = -30.0, 30.0
    CG_T_MIN,  CG_T_MAX  = -12.0, 12.0
    CG_KP_MIN, CG_KP_MAX =   0.0, 500.0
    CG_KD_MIN, CG_KD_MAX =   0.0,   5.0

    # CyberGear communication types
    CG_CMD_MOTOR_CONTROL  = 0x01
    CG_CMD_MOTOR_FEEDBACK = 0x02
    CG_CMD_MOTOR_ENABLE   = 0x03
    CG_CMD_MOTOR_STOP     = 0x04
    CG_CMD_SET_MECH_ZERO  = 0x06

    def __init__(self, can: WaveshareUSBCAN, motor_id: int = 1,
                 motor_type: str = "ak", master_id: int = 0x00):
        self.can       = can
        self.id        = motor_id
        self.type      = motor_type.lower()
        self.master_id = master_id

    def _make_can_id(self, comm_type: int, data_field: int = 0x0000) -> int:
        return (
            ((comm_type  & 0x1F)   << 24) |
            ((data_field & 0xFFFF) <<  8) |
            ( self.id    & 0xFF)
        )

    # ─────────────────────────────────────────────
    #  enter_mit  (enable / arm)
    # ─────────────────────────────────────────────

    def enter_mit(self):
        if self.type == "ak" or self.type == "dm":
            self.can.send_can(self.id, [0xFF] * 7 + [0xFC])

        elif self.type == "cg":
            can_id = self._make_can_id(self.CG_CMD_MOTOR_ENABLE,
                                       data_field=self.master_id)
            self.can.send_can(can_id, bytes(8), extended=True)

        else:
            raise ValueError(f"Unknown motor_type '{self.type}'.")

    # ─────────────────────────────────────────────
    #  exit_mit  (stop / disarm)
    # ─────────────────────────────────────────────

    def exit_mit(self):
        if self.type == "ak" or self.type == "dm":
            self.can.send_can(self.id, [0xFF] * 7 + [0xFD])

        elif self.type == "cg":
            can_id  = self._make_can_id(self.CG_CMD_MOTOR_STOP,
                                        data_field=self.master_id)
            self.can.send_can(can_id, bytes(8), extended=True)

        else:
            raise ValueError(f"Unknown motor_type '{self.type}'.")

    # ─────────────────────────────────────────────
    #  zero_position
    # ─────────────────────────────────────────────

    def zero_position(self):
        if self.type == "ak" or self.type == "dm":
            self.can.send_can(self.id, [0xFF] * 7 + [0xFE])

        elif self.type == "cg":
            can_id = self._make_can_id(self.CG_CMD_SET_MECH_ZERO,
                                       data_field=self.master_id)
            self.can.send_can(can_id, bytes([1, 0, 0, 0, 0, 0, 0, 0]),
                              extended=True)

        else:
            raise ValueError(f"Unknown motor_type '{self.type}'.")

    # ─────────────────────────────────────────────
    #  send
    # ─────────────────────────────────────────────

    def send(self, p: float, v: float, kp: float, kd: float, t: float):
        if self.type == "ak" or self.type == "dm":
            p_i  = float_to_uint(p,  self.AK_P_MIN,  self.AK_P_MAX,  16)
            v_i  = float_to_uint(v,  self.AK_V_MIN,  self.AK_V_MAX,  12)
            kp_i = float_to_uint(kp, self.AK_KP_MIN, self.AK_KP_MAX, 12)
            kd_i = float_to_uint(kd, self.AK_KD_MIN, self.AK_KD_MAX, 12)
            t_i  = float_to_uint(t,  self.AK_T_MIN,  self.AK_T_MAX,  12)

            data = [
                p_i >> 8,
                p_i & 0xFF,
                v_i >> 4,
                ((v_i  & 0xF) << 4) | (kp_i >> 8),
                kp_i & 0xFF,
                kd_i >> 4,
                ((kd_i & 0xF) << 4) | (t_i  >> 8),
                t_i & 0xFF,
            ]
            self.can.send_can(self.id, data)

        elif self.type == "cg":
            t_u  = float_to_uint(t,  self.CG_T_MIN,  self.CG_T_MAX,  16)
            p_u  = float_to_uint(p,  self.CG_P_MIN,  self.CG_P_MAX,  16)
            v_u  = float_to_uint(v,  self.CG_V_MIN,  self.CG_V_MAX,  16)
            kp_u = float_to_uint(kp, self.CG_KP_MIN, self.CG_KP_MAX, 16)
            kd_u = float_to_uint(kd, self.CG_KD_MIN, self.CG_KD_MAX, 16)

            can_id = self._make_can_id(self.CG_CMD_MOTOR_CONTROL, data_field=t_u)
            data   = struct.pack('>HHHH', p_u, v_u, kp_u, kd_u)
            self.can.send_can(can_id, data, extended=True)

        else:
            raise ValueError(f"Unknown motor_type '{self.type}'.")

    # ─────────────────────────────────────────────
    #  read
    # ─────────────────────────────────────────────

    def read(self) -> dict | None:
        if self.type == "ak":
            frame = self.can.get_latest(self.id)
            if frame is None:
                return None
            d = frame["data"]
            if len(d) < 8:
                return None

            pos_raw = (d[1] << 8) | d[2]
            vel_raw = (d[3] << 4) | (d[4] >> 4)
            cur_raw = ((d[4] & 0x0F) << 8) | d[5]
            temp    = d[6] - 40
            err     = d[7]

            return {
                "pos":    uint_to_float(pos_raw, self.AK_P_MIN, self.AK_P_MAX, 16),
                "vel":    uint_to_float(vel_raw, self.AK_V_MIN, self.AK_V_MAX, 12),
                "torque": uint_to_float(cur_raw, self.AK_T_MIN, self.AK_T_MAX, 12),
                "temp":   temp,
                "error":  err,
            }

        elif self.type == "dm":
            frame = self.can.get_latest(self.id)
            if frame is None:
                return None
            d = frame["data"]
            if len(d) < 8:
                return None

            motor_id   = d[0] & 0x0F
            err        = d[0] >> 4
            pos_raw    = (d[1] << 8) | d[2]
            vel_raw    = (d[3] << 4) | (d[4] >> 4)
            tor_raw    = ((d[4] & 0x0F) << 8) | d[5]
            mos_temp   = d[6]
            rotor_temp = d[7]

            return {
                "id":         motor_id,
                "error":      err,
                "pos":        uint_to_float(pos_raw, self.AK_P_MIN, self.AK_P_MAX, 16),
                "vel":        uint_to_float(vel_raw, self.AK_V_MIN, self.AK_V_MAX, 12),
                "torque":     uint_to_float(tor_raw, self.AK_T_MIN, self.AK_T_MAX, 12),
                "mos_temp":   mos_temp,
                "rotor_temp": rotor_temp,
            }

        elif self.type == "cg":
            frame = self.can.get_latest(self.id)
            if frame is None:
                return None
            d = frame["data"]
            if len(d) < 8:
                return None

            can_id = frame["raw_id"]
            fault  = (can_id >> 16) & 0x3F
            mode   = (can_id >> 22) & 0x03

            raw_pos    = (d[0] << 8) | d[1]
            raw_vel    = (d[2] << 8) | d[3]
            raw_torque = (d[4] << 8) | d[5]
            raw_temp   = (d[6] << 8) | d[7]

            return {
                "pos":    uint_to_float(raw_pos,    self.CG_P_MIN, self.CG_P_MAX, 16),
                "vel":    uint_to_float(raw_vel,    self.CG_V_MIN, self.CG_V_MAX, 16),
                "torque": uint_to_float(raw_torque, self.CG_T_MIN, self.CG_T_MAX, 16),
                "temp":   raw_temp / 10.0,
                "fault":  fault,
                "mode":   mode,
            }

        else:
            raise ValueError(f"Unknown motor_type '{self.type}'.")
def set_can_id(self, new_id: int) -> None:
    """
    Communication type 7 — change motor CAN ID.
    Takes effect immediately. Reply is a type-0 broadcast frame.

    Bit layout of the 29-bit CAN ID:
      bits 28~24 : comm type = 7
      bits 23~16 : new motor CAN ID
      bits 15~8  : host CAN ID (master_id)
      bits  7~0  : current motor CAN ID
    """
    can_id = (
        (0x07             << 24) |   # comm type
        ((new_id  & 0xFF) << 16) |   # new motor CAN ID  (bits 23~16)
        ((self.master_id & 0xFF) <<  8) |   # host CAN ID       (bits 15~8)
        ( self.id & 0xFF)            # current motor CAN ID (bits 7~0)
    )
    data = bytes([1, 0, 0, 0, 0, 0, 0, 0])
    self.can.send_can(can_id, data, extended=True)