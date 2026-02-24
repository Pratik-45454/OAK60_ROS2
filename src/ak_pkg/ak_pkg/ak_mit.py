from .Waveshare import WaveshareUSBCAN

def float_to_uint(x, xmin, xmax, bits):
    x = max(min(x, xmax), xmin)
    return int((x - xmin) * ((1 << bits) - 1) / (xmax - xmin))

def uint_to_float(x, x_min, x_max, bits):
    span = x_max - x_min
    return (x / ((1 << bits) - 1)) * span + x_min


class AKMotorMIT:
    P_MIN, P_MAX = -12.5, 12.5
    V_MIN, V_MAX = -45.0, 45.0
    T_MIN, T_MAX = -15.0, 15.0
    KP_MIN, KP_MAX = 0.0, 500.0
    KD_MIN, KD_MAX = 0.0, 5.0

    def __init__(self, can: WaveshareUSBCAN, motor_id=1):
        self.can = can
        self.id = motor_id

    # ----- MIT commands -----
    def enter_mit(self):
        self.can.send_can(self.id, [0xFF]*7 + [0xFC])

    def exit_mit(self):
        self.can.send_can(self.id, [0xFF]*7 + [0xFD])

    def zero_position(self):
        self.can.send_can(self.id, [0xFF]*7 + [0xFE])

    # ----- Control -----
    def send(self, p, v, kp, kd, t):
        p_i  = float_to_uint(p,  self.P_MIN, self.P_MAX, 16)
        v_i  = float_to_uint(v,  self.V_MIN, self.V_MAX, 12)
        kp_i = float_to_uint(kp, self.KP_MIN, self.KP_MAX, 12)
        kd_i = float_to_uint(kd, self.KD_MIN, self.KD_MAX, 12)
        t_i  = float_to_uint(t,  self.T_MIN, self.T_MAX, 12)

        data = [
            p_i >> 8,
            p_i & 0xFF,
            v_i >> 4,
            ((v_i & 0xF) << 4) | (kp_i >> 8),
            kp_i & 0xFF,
            kd_i >> 4,
            ((kd_i & 0xF) << 4) | (t_i >> 8),
            t_i & 0xFF
        ]

        self.can.send_can(self.id, data)

    def read(self):
        d = self.can.get_latest(self.id)
        if not d or len(d) < 8:
            return None
        pos_raw = (d[1] << 8) | d[2]
        vel_raw = (d[3] << 4) | (d[4] >> 4)
        cur_raw = ((d[4] & 0x0F) << 8) | d[5]
        temp = d[6] - 40
        err  = d[7]

        p = uint_to_float(pos_raw, self.P_MIN, self.P_MAX, 16)
        v = uint_to_float(vel_raw, self.V_MIN, self.V_MAX, 12)
        i = uint_to_float(cur_raw, self.T_MIN, self.T_MAX, 12)

        return p, v, i, temp, err


