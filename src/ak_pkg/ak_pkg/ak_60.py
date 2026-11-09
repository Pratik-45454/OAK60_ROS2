#!/usr/bin/env python3

import time
import threading
import os
import math
# from urllib import response
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from motor_interfaces.srv import MotorState
import functools
import copy


from .Waveshare import WaveshareUSBCAN
from .Motor import MotorMIT


class AKMotorService(Node):

    def __init__(self):
        super().__init__('ak_motor_service')

        self.connection_lock = threading.Lock()

        self.devices = [
            "/dev/ttyCH341USB0",
            "/dev/ttyCH341USB1",
            "/dev/ttyCH341USB2"
        ]

        self.device = None        
        self.can_baud = 0x01
        self.ramp_time = 0.5      
        self.ramp_k = 1.0         
        self._s0 = self._sigmoid(-self.ramp_k / 2)
        self._s1 = self._sigmoid(self.ramp_k / 2)
        self.ramp_dt = 0.05 
        self.dt = 0.05
        self.threshold = 0.9
        self.timeout = 5.0
        self.services_enabled = True
        self.temp_limit = 45.0   # °C, block services above this

        self.motors = {
            "climb": {
                "id": 0,
                "handle": None,
                "running": False,
                "stop_requested": False,
                "is_first": True,
                "initialized": False,
                "gains": {
                    "up": {"kp": 10, "kd": 1.3, 'ramp_time': 1.0},
                    "down": {"kp": 8, "kd": 0.8, 'ramp_time': 1.5}
                },
                "last": "d",
                "type": "ak",
                "enter_delay": 0.01,
                "ramp_k": 1.0,
                "slight_moved": False,
            },

            "roll": {
                "id": 4,
                "handle": None,
                "running": False,
                "stop_requested": False,
                "initialized": False,
                "gains": {
                    "up": {"kp": 23, "kd": 1.4, 'ramp_time': 2.0},
                    "down": {"kp": 13, "kd": 1.4, 'ramp_time': 1.5}
                },
                "last": "d",
                "type": "cg",
                "enter_delay": 0.05,
                "current_state": 0,
                "ramp_k": 0.5,

            },

            "elev_vert": {
                "id": 2,
                "handle": None,
                "running": False,
                "stop_requested": False,
                "gains": {'up':{
                    "kp": 15, 
                    "kd": 1.3,
                    'ramp_time': 0.7}
                },
                'initialized': False,
                "type": "dm",
                "enter_delay": 0.05,
                "current_state": 0,
                'current_abs': 0,
                'zero_index': 0,
                'zeros': [12.5,25.0],
                "ramp_k": 1.0
            },

            "gripper_catch": {
                "id": 3,
                "handle": None,
                "running": False,
                "stop_requested": False,
                'initialized': False,
                "gains": {'up':{
                    "kp": 20, 
                    "kd": 1.4,
                    'ramp_time': 0.7}
                },
                "type": "dm",
                "enter_delay": 0.05,
                "current_state": 0,
                "ramp_k": 1.0,
            },

            "gripper_pitch": {
                "id": 0x7F,
                "handle": None,
                "running": False,
                "stop_requested": False,
                'initialized': False,
                "gains": {'up':{
                    "kp": 42, 
                    "kd": 1.4,
                    'ramp_time': 1.3}
                },
                "current_state": 0,
                "type": "cg",
                "enter_delay": 0.05,
                "ramp_k": 0.7
            },

            "spearhead_catch": {
                "id": 9,
                "handle": None,
                "running": False,
                "stop_requested": False,
                'initialized': False,
                "states": {-1: 1.5, 0: 2.5, 1: 3.5},
                "type": "ak",
                "enter_delay": 0.05,
            }

        }

        self.can = None

        # Snapshot of pristine motor state (excluding 'handle') for emergency reset
        self._motors_initial_state = copy.deepcopy(self.motors)

        # SERVICES
        self.up_srv = self.create_service(
            SetBool,
            'ak_motor_up_execute',
            self.climb_up_callback
        )

        self.down_srv = self.create_service(
            SetBool,
            'ak_motor_down_exec',
            self.climb_down_callback
        )

        self.roll_up_srv = self.create_service(
            MotorState,
            'ak_motor_roll_up',
            self.roll_up_callback
        )


        self.elev_vert_srv = self.create_service(
            MotorState,
            'elev_vert_move',
            self.elev_vert_callback
        )

        self.gripper_catch_srv = self.create_service(
            MotorState,
            'gripper_catch_move',
            self.gripper_catch_callback
        )

        self.gripper_pitch_srv = self.create_service(
            MotorState,
            'gripper_pitch_set',
            self.gripper_pitch_callback
        )

        self.spearhead_catch_srv = self.create_service(
            MotorState,
            'spearhead_catch_set',
            self.spearhead_catch_callback
        )

        self.stop_climb_srv = self.create_service(
            SetBool,
            'ak_motor_stop_climb',
            self.stop_climb_callback
        )

        self.stop_roll_srv = self.create_service(
            SetBool,
            'cg_motor_stop_roll',
            self.stop_roll_callback
        )

        self.move_climb_slight_srv = self.create_service(
            SetBool,
            'move_climb_slight',
            self.move_climb_slight_callback
        )

        self.init_climb_srv = self.create_service(
            SetBool, 'init_climb', self.init_climb_callback
        )
        self.init_roll_srv = self.create_service(
            SetBool, 'init_roll', self.init_roll_callback
        )
        self.init_elev_vert_srv = self.create_service(
            SetBool, 'init_elev_vert', self.init_elev_vert_callback
        )
        self.init_gripper_catch_srv = self.create_service(
            SetBool, 'init_gripper_catch', self.init_gripper_catch_callback
        )

        self.init_gripper_pitch_srv = self.create_service(
            SetBool, 'init_gripper_pitch', self.init_gripper_pitch_callback
        )

        self.init_spearhead_catch_srv = self.create_service(
            SetBool, 'init_spearhead_catch', self.init_spearhead_catch_callback
        )

        self.stop_elev_vert_srv = self.create_service(
            SetBool, 'dm_motor_stop_elev_vert', 
            self.stop_elev_vert_callback
        )

        self.stop_gripper_catch_srv = self.create_service(
            SetBool, 'dm_motor_stop_gripper_catch', 
            self.stop_gripper_catch_callback
        )

        self.stop_gripper_pitch_srv = self.create_service(
            SetBool, 'cg_motor_stop_gripper_pitch', 
            self.stop_gripper_pitch_callback
        )

        self.stop_spearhead_catch_srv = self.create_service(
            SetBool,
            'stop_spearhead_catch',
            self.stop_spearhead_catch_callback
        )  

        self.test_all_motors_srv = self.create_service(
            SetBool,
            'test_all_motors',
            self.test_all_callback
        )

        self.emergency_stop_srv = self.create_service(
           SetBool,
          'emergency_stop',
            self.emergency_stop_callback
        )

        self.get_logger().info("Initializing CAN...")
        self.init_can_with_retry()

        

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def map_range(self, x, x1, x2, y1, y2):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    def find_can_device(self):
        for dev in self.devices:

            if os.path.exists(dev):

                self.get_logger().info(
                    f"Found CAN device: {dev}"
                )
                return dev
        return None

    def require_enabled(func):
        @functools.wraps(func)
        def wrapper(self, request, response):
            if not self.services_enabled:
                response.success = False
                response.message = "Services disabled by emergency_stop"
                self.get_logger().warn("Services disabled by emergency_stop")
                return response
            return func(self, request, response)
        return wrapper

    def require_temp_ok(motor_name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, request, response):
                motor_obj = self.motors[motor_name]
                if not self.check_motor_temp(motor_obj, motor_name):
                    response.success = False
                    response.message = (
                        f"{motor_name} motor over temperature "
                        f"(>{self.temp_limit}°C), service blocked"
                    )
                    return response
                return func(self, request, response)
            return wrapper
        return decorator
    
    def emergency_stop_callback(self, request, response):
        self.services_enabled = not(request.data)

        if request.data:
            self.reset_motor_states()
            self.get_logger().warn("Emergency stop triggered: all motor states reset to initial values")

        self.get_logger().info(f"Services {'ENABLED' if self.services_enabled else 'DISABLED'}")
        response.success = True
        response.message = f"Services {'enabled' if self.services_enabled else 'disabled'}"
        return response

    def init_can_with_retry(self):
        while rclpy.ok():

            try:

                self.device = self.find_can_device()

                if self.device is None:
                
                    self.get_logger().error(
                        "No CAN device found "
                        "(/dev/ttyCH341USB0 or USB1)"
                    )

                    time.sleep(1.0)
                    continue
                
                self.get_logger().info(
                    f"Opening CAN port: {self.device}"
                )

                self.can = WaveshareUSBCAN(
                    self.device,
                    can_baud=self.can_baud
                )

                self.can.start_rx()

                self.get_logger().info(
                    "CAN initialization successful"
                )

                break

            except KeyboardInterrupt:

                self.get_logger().warn(
                    "CAN init interrupted"
                )

                return

            except Exception as e:

                self.get_logger().error(
                    f"CAN init failed: {e}"
                )

                try:
                    if self.can is not None:
                        self.can.close()
                except Exception:
                    pass

                self.can = None

                time.sleep(1.0)

        if self.can is None:
            return

        for name, m in self.motors.items():

            try:

                m["handle"] = MotorMIT(
                    self.can,
                    motor_id=m["id"],
                    motor_type=m['type']
                )

                self.get_logger().info(
                    f"{name} motor initialized"
                )

            except Exception as e:

                m["handle"] = None

                self.get_logger().warn(
                    f"{name} motor failed: {e}"
                )

    def prepare_motor(self, motor_obj):

        motor = motor_obj["handle"]

        if motor is None:

            self.get_logger().error(
                "prepare_motor(): motor handle is None"
            )

            return False

        try:
            time.sleep(0.001)
            motor_obj['handle'].send(p = 0.0 , v = 0.0, kp = 0.0, kd = 0.0, t= 0.0)
            # self.get_logger().info("Preparing motor...")
            time.sleep(0.01)
            motor.exit_mit()
            time.sleep(0.01)
            motor.zero_position()
            time.sleep(0.01)
            motor.enter_mit()
            time.sleep(motor_obj["enter_delay"])
            # fb = motor.read()
            # time.sleep(0.01)
            # self.get_logger().info(
            #     f"Initial feedback: {fb}"
            # )
            self.get_logger().info(
                "Motor prepare successful"
            )

            return True

        except Exception as e:

            self.get_logger().error(
                f"prepare_motor() failed: "
                f"{type(e).__name__}: {e}"
            )

            return False

    def check_motor_connection(self, motor_obj, name, retries=3):
            """Send a zero command and verify feedback is received."""
            motor = motor_obj["handle"]
            try:
                motor.send(p=0.0, v=0.0, kp=0.0, kd=0.0, t=0.0)
                for attempt in range(retries):
                    time.sleep(0.05)
                    fb = motor.read()
                    if fb is not None and 'pos' in fb:
                        self.get_logger().info(f"{name}: connection confirmed, pos={fb['pos']:.3f}")
                        return True
                    self.get_logger().warn(f"{name}: no feedback on attempt {attempt + 1}/{retries}")
                self.get_logger().error(f"{name}: no feedback after {retries} attempts")
                return False
            except Exception as e:
                self.get_logger().error(f"{name}: connection check failed: {e}")
                return False

    def check_motor_temp(self, motor_obj, name):
        """Return True if motor temp is within safe limits (or unknown/unreadable)."""
        motor = motor_obj.get("handle")
        if motor is None:
            return True  # let existing "motor unavailable" checks handle this

        try:
            fb = motor.read()
            if fb is not None and 'temp' in fb:
                temp = fb['temp']
                if temp > self.temp_limit:
                    self.get_logger().error(
                        f"{name}: OVER TEMPERATURE ({temp:.1f}°C > {self.temp_limit}°C), service blocked"
                    )
                    return False
            return True
        except Exception as e:
            self.get_logger().warn(f"{name}: temp check failed ({e}), allowing by default")
            return True

    def move_motor(self, motor_obj, target, kp, kd, ramp_time, start):

        motor = motor_obj["handle"]

        if motor is None:
            self.get_logger().error("move_motor(): motor handle is None")
            return False

        ramp_k = motor_obj.get("ramp_k", self.ramp_k)
        s0 = self._sigmoid(-ramp_k / 2)
        s1 = self._sigmoid( ramp_k / 2)

        start_pos = start

        self.get_logger().info(
            f"Ramping {start_pos:.3f} → {target:.3f} "
            f"over {ramp_time}s, kp={kp}, kd={kd}"
        )

        t_start = time.monotonic()
        loop_count = 0
        hold_start = None          # when we first crossed threshold

        while rclpy.ok():

            loop_count += 1

            if motor_obj["stop_requested"]:
                self.get_logger().warn("Motion stopped by request")
                try:
                    time.sleep(0.01)
                    motor.exit_mit()
                except Exception:
                    pass
                return False

            elapsed = time.monotonic() - t_start
            alpha = min(elapsed / ramp_time, 1.0)
            s = self._sigmoid(ramp_k * (alpha - 0.5))
            factor = (s - s0) / (s1 - s0)
            cmd = start_pos + (target - start_pos) * factor if alpha < 1.0 else target

            try:
                motor.send(p=cmd, v=0.0, kp=kp, kd=kd, t=0.0)
            except Exception as e:
                self.get_logger().error(f"motor.send() failed: {e}")
                return False

            time.sleep(self.ramp_dt)

            try:
                fb = motor.read()
            except Exception as e:
                self.get_logger().error(f"motor.read() failed: {e}")
                return False

            if fb is not None and 'pos' in fb:
                current_pos = fb['pos']
                error = abs(current_pos - target)

                if loop_count % 50 == 0:
                    self.get_logger().info(
                        f"cmd={cmd:.3f}, pos={current_pos:.3f}, "
                        f"err={error:.3f}, alpha={alpha:.2f}"
                    )

                # Start 1-second hold timer on first threshold crossing
                if error < self.threshold and hold_start is None:
                    hold_start = time.monotonic()
                    self.get_logger().info(
                        f"Threshold crossed at pos={current_pos:.3f}, holding 1s..."
                    )

                # Exit only after holding for 1 full second
                if hold_start is not None and (time.monotonic() - hold_start) >= 0.7:
                    self.get_logger().info(
                        f"Hold complete, final pos={current_pos:.3f}"
                    )
                    return True

        self.get_logger().warn("move_motor(): exited because rclpy is no longer ok")
        return False

    def move_to_state(self, name, requested_state):
        motor_obj = self.motors[name]
        states = motor_obj["states"]
        if requested_state not in states:
            self.get_logger().error(
                f"Invalid state {requested_state}"
            )
            return False, "Invalid state"

        current_state = motor_obj["current_state"]
        target = states[requested_state]
        ok = self.execute_motion(
            name,
            'up',
            target,
            st = current_state
        )
        if ok:
            motor_obj["current_state"] = requested_state
            return True, f"Moved to state {requested_state}"
        return False, "Motion failed"
    

    def execute_motion(self, name, direction, target, st):

        motor_obj = self.motors[name]

        if motor_obj["running"]:
            self.get_logger().warn(f"{name} motor already running")
            return False

        gains = motor_obj["gains"][direction]
        kp = gains["kp"]
        kd = gains["kd"]
        ramp_time = gains.get("ramp_time", motor_obj.get("ramp_time", self.ramp_time))

        motor_obj["running"] = True
        motor_obj["stop_requested"] = False

        try:
            motor = motor_obj["handle"]

            if motor is None:
                self.get_logger().error(f"{name} motor unavailable")
                return False

            self.get_logger().info(
                f"execute_motion(name={name}, direction={direction}, target={target})"
            )

            if name == 'climb' and motor_obj['is_first']:
                st = 0
                self.move_motor(motor_obj, 10, kp, kd, ramp_time, start=-2.5)
                self.prepare_motor(motor_obj)
                motor_obj['is_first'] = False

            success = self.move_motor(motor_obj, target, kp, kd, ramp_time, start=st)

            if success:
                if name in ["climb", "roll"]:
                    motor_obj["last"] = "u" if direction == "up" else "d"
                time.sleep(0.01)
                self.get_logger().info(f"{name} reached target")
            else:
                self.get_logger().warn(f"{name} motion failed")

            return success

        finally:
            motor_obj["running"] = False

    def stop_motor(self, name):
        motor_obj = self.motors[name]
        motor_obj["stop_requested"] = True
        motor_obj["is_first"] = True

        self.shutdown_motor(name)

    def reset_motor_states(self):
        """Reset every motor's state dict to its init-time values, keeping 'handle' untouched."""
        for name, motor_obj in self.motors.items():
            live_handle = motor_obj.get("handle")
            pristine = copy.deepcopy(self._motors_initial_state[name])
            pristine["handle"] = live_handle
            self.motors[name] = pristine
            self.get_logger().info(f"{name} state reset to initial values")

    @require_enabled
    @require_temp_ok("climb")
    def climb_up_callback(self, request, response):

        if not request.data:
            response.success = False
            response.message = "Request data false"

            return response

        if not self.motors['climb']['initialized']:

            response.success = False
            response.message = "Climb motor not initialized"

            return response

        if self.motors['climb']['last'] != 'd' or self.motors['climb']['slight_moved']:
            response.success = False
            response.message = "Last climb motion not DOWN or slight movement detected"

            return response

        ok = self.execute_motion(
            "climb",
            "up",
            12.5,
            st = -12.5
        )

        response.success = ok

        response.message = (
            "Climb UP completed successfully"
            if ok else
            "Climb UP failed"
        )

        return response

    @require_enabled
    @require_temp_ok("climb")
    def climb_down_callback(self, request, response):

        if not request.data:

            response.success = False
            response.message = "Request data false"

            return response

        if not self.motors['climb']['initialized']:

            response.success = False
            response.message = "Climb motor not initialized"

            return response

        if self.motors['climb']['last'] != 'u':

            response.success = False
            response.message = "Last climb motion not UP"

            return response

        ok = self.execute_motion(
            "climb",
            "down",
            -12.5,
            st = 12.5
        )

        response.success = ok

        response.message = (
            "Climb DOWN completed successfully"
            if ok else
            "Climb DOWN failed"
        )

        return response

    @require_enabled
    @require_temp_ok("roll")
    def roll_up_callback(self, request, response):

        if not request.data or not self.motors['roll']['initialized']:
            response.success = False
            response.message = "Request data false or roll motor not initialized"
            return response

        if abs(request.state) > 12.5:
            response.success = False
            response.message = "Angle cannot exceed 12.5 radians"
            return response
        self.get_logger().info(f"roll state={request.state:.4f}")

        ok = self.execute_motion(
            "roll",
            "up",
            request.state,
            st=0
        )

        if ok:
            try:
                self.prepare_motor(self.motors['roll'])
                self.get_logger().info(
                    "Roll prepare_motor applied after reaching target"
                )
                motor = self.motors['roll']['handle']

                motor.send(
                    p=0.0,
                    v=0.0,
                    kp=20.0,
                    kd=0.0,
                    t=0.0
                )
                
            except Exception as e:
                self.get_logger().warn(
                    f"prepare_motor failed: {e}"
                )

        response.success = ok
        response.message = (
            "Roll UP completed successfully"
            if ok else
            "Roll UP failed"
        )

        return response

    @require_enabled
    @require_temp_ok("elev_vert")
    def elev_vert_callback(self, request, response):

        if not request.data or not self.motors['elev_vert']['initialized']:

            response.success = False
            response.message = "Request data false or elev motor not initialized"

            return response

        if request.state>36:
            response.success = False
            response.message = "Angle cannot exceed 36 radians"
            return response
            
        try:

            check = self.motors['elev_vert']['zeros'][self.motors['elev_vert']['zero_index']] - request.state
    
            if abs(check) > 12.5:
                self.execute_motion(
                    "elev_vert",
                    "up",
                    -12.5 if check<0 else 12.5,
                    st = self.motors['elev_vert']["current_state"]
                )
                self.prepare_motor(self.motors['elev_vert'])
                self.motors['elev_vert']['zero_index'] = int(not(self.motors['elev_vert']['zero_index']))
                self.motors['elev_vert']['current_abs'] = self.motors['elev_vert']['zeros'][self.motors['elev_vert']['zero_index']]
                self.motors['elev_vert']['current_state'] = 0
    
            actual = self.map_range(request.state - self.motors['elev_vert']['zeros'][self.motors['elev_vert']['zero_index']], -12.5, 12.5, 12.5, -12.5)
            ok = self.execute_motion(
                "elev_vert",
                "up",
                actual,
                st = self.motors['elev_vert']["current_state"]
            )
            if ok:
                self.motors['elev_vert']['current_state'] = actual
                self.motors['elev_vert']['current_abs'] = request.state

            response.success = ok
            response.message = (
                "elev motion completed successfully" if rclpy.ok else "elev motion failed"
            )
        except Exception as e:
            response.success = False
            response.message = f"elev_vert_callback exception: {e}"

        return response

    @require_enabled
    @require_temp_ok("climb")
    def move_climb_slight_callback(self, request, response):
        motor_obj = self.motors["climb"]

        if not motor_obj["initialized"]:
            response.success = False
            response.message = "Climb motor not initialized"
            return response

        if not motor_obj["is_first"]:
            response.success = False
            response.message = "Climb already used, slight move not allowed"
            return response

        if request.data and motor_obj["slight_moved"]:
            response.success = False
            response.message = "Already at slight position (-3.5)"
            return response

        if not request.data and not motor_obj["slight_moved"]:
            response.success = False
            response.message = "Already at init position (-2.5)"
            return response

        gains = motor_obj["gains"]["down"]
        target  = -3.5 if request.data else -2.5
        start   = -2.5 if request.data else -3.5

        success = self.move_motor(
            motor_obj, target,
            gains["kp"], gains["kd"],
            gains["ramp_time"],
            start=start
        )

        if success:
            motor_obj["slight_moved"] = request.data
            response.success = True
            response.message = f"Climb slight {'extended' if request.data else 'returned'} successfully"
        else:
            response.success = False
            response.message = "Climb slight move failed"

        return response

    @require_enabled
    @require_temp_ok("gripper_catch")
    def gripper_catch_callback(self, request, response):
        if not request.data or not self.motors['gripper_catch']['initialized']:

            response.success = False
            response.message = "Request data false or gripper catch motor not initialized"

            return response

        
        if request.state > 12.5:

            response.success = False
            response.message = "Angle cannot exceed 12.5 radians"

            return response

        target = request.state

        # negative is open positive is close
        self.get_logger().info(
            f"gripper_catch state={request.state:.4f}"
        )
        
        ok = self.execute_motion(
            "gripper_catch",
            "up",
            target,
            st = self.motors['gripper_catch']["current_state"]
        )

        response.success = ok

        if ok:
            self.motors['gripper_catch']['current_state'] = target

        response.message = (
            "gripper catch motion completed successfully"
            if ok else
            "gripper catch motion failed"
        )

        return response
    
    @require_enabled
    @require_temp_ok("gripper_pitch")
    def gripper_pitch_callback(self, request, response):

        if not request.data or not self.motors['gripper_pitch']['initialized']:

            response.success = False
            response.message = "Request data false or gripper pitch motor not initialized"

            return response

        
        if request.state < 0 or request.state > 12.5:

            response.success = False
            response.message = "Angle cannot exceed 12.5 radians or cannot be positive"

            return response
        
        target = -request.state

        self.get_logger().info(
            f"gripper_pitch state={request.state:.4f}"
        )
        
        ok = self.execute_motion(
            "gripper_pitch",
            "up",
            target,
            st = self.motors['gripper_pitch']["current_state"]
        )

        response.success = ok

        if ok:
            self.motors['gripper_pitch']['current_state'] = target

        response.message = (
            "gripper pitch motion completed successfully"
            if ok else
            "gripper pitch motion failed"
        )

        return response

    @require_enabled
    @require_temp_ok("spearhead_catch")
    def spearhead_catch_callback(self, request, response):

        if not request.data or not self.motors['spearhead_catch']['initialized']:
            response.success = False
            response.message = "Request data false or spearhead catch not initialized"
            return response

        requested_state = request.state

        try:
            motor = self.motors['spearhead_catch']['handle']

            motor.send(
                p=requested_state,
                v=0.0,
                kp=0,
                kd=0.0,
                t=0.0
            )

            response.success = True
            response.message = f"Moved spearhead catch to state {requested_state}"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response


    def stop_climb_callback(self, request, response):

        if request.data:

            self.stop_motor("climb")

            response.success = True
            response.message = "Climb stopped"

        else:

            response.success = False
            response.message = "Request data false"

        return response
    
    def stop_gripper_catch_callback(self, request, response):
        if request.data:
            self.stop_motor("gripper_catch")
            response.success =  True
            response.message = 'gripper catch stopped'


        else:
            response.success = False
            response.message = 'requeset data false'

        return response
    
    def stop_elev_vert_callback(self, request, response):
        if request.data:
            self.stop_motor("elev_vert")
            response.success =  True
            response.message = 'vertical elevator stopped'


        else:
            response.success = False
            response.message = 'requeset data false'

        return response
    
    def stop_gripper_pitch_callback(self, request, response):
        if request.data:
            self.stop_motor("gripper_pitch")
            response.success =  True
            response.message = 'gripper pitch stopped'

        else:
            response.success = False
            response.message = 'requeset data false'

        return response

    def stop_roll_callback(self, request, response):

        if request.data:

            self.stop_motor("roll")

            response.success = True
            response.message = "Roll stopped"

        else:

            response.success = False
            response.message = "Request data false"

        return response


    def stop_spearhead_catch_callback(self, request, response):
        if request.data:
            motor_obj = self.motors["spearhead_catch"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Spearhead catch motor unavailable"
                return response

            motor_obj["handle"].send(
                p=0,
                v=0.0,
                kp=0.0,
                kd=0.0,
                t=0.0
            )
            
            response.success = True
            response.message = "Spearhead catch stopped"
        else:
            response.success = False
            response.message = "Request data false"
        return response

    def init_climb_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['climb']['initialized']:
            response.success = True
            response.message = "Climb already initialized"
            return response

        try:
            motor_obj = self.motors["climb"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Climb motor unavailable"
                return response

            gains = motor_obj["gains"]["down"]

            if not self.prepare_motor(motor_obj):
                response.success = False
                response.message = "Climb prepare failed"
                return response
            

            if not self.check_motor_connection(motor_obj, "climb"):  # or whichever motor
                response.success = False
                response.message = "Climb motor not responding (no feedback)"
                return response
            
            # ramp_time = motor_obj.get("ramp_time", self.ramp_time)

            success = self.move_motor(
                motor_obj, -2.5,
                gains["kp"], gains["kd"],
                0.5,
                start=0
            )
            if not success:
                response.success = False
                response.message = "Climb init move failed"
                return response

            motor_obj['initialized'] = True
            response.success = True
            response.message = "Climb initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response

    def init_roll_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['roll']['initialized']:
            response.success = True
            response.message = "Roll already initialized"
            return response

        try:
            motor_obj = self.motors["roll"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Roll motor unavailable"
                return response

            gains = motor_obj["gains"]["up"]

            if not self.prepare_motor(motor_obj):
                response.success = False
                response.message = "Roll prepare failed"
                return response
            
            if not self.check_motor_connection(motor_obj, "roll"):  # or whichever motor
                response.success = False
                response.message = "roll motor not responding (no feedback)"
                return response
            
            # motor_obj.get("ramp_time", self.ramp_time)
            # success = self.move_motor(
            #     motor_obj, request.state,
            #     gains["kp"], gains["kd"],
            #     0.8,
            #     start=0
            # )
            # if not success:
            #     response.success = False
            #     response.message = "Roll init move failed"
            #     return response

            motor_obj['initialized'] = True
            motor_obj['current_state'] = 0
            # motor_obj['last'] = 'u'
            response.success = True
            response.message = "Roll initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response

    def init_elev_vert_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['elev_vert']['initialized']:
            response.success = True
            response.message = "Vert elev already initialized"
            return response

        try:
            motor_obj = self.motors["elev_vert"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Vert elev motor unavailable"
                return response

            gains = motor_obj["gains"]["up"]
            kp, kd = gains["kp"], gains["kd"]

            if not self.prepare_motor(motor_obj):
                response.success = False
                response.message = "Vert elev prepare failed"
                return response
            
            if not self.check_motor_connection(motor_obj, "elev_vert"):  # or whichever motor
                response.success = False
                response.message = "elev vert motor not responding (no feedback)"
                return response

            success = self.move_motor(motor_obj, -12.5, kp, kd, start=0, ramp_time = 0.8)
            if not success:
                response.success = False
                response.message = "Vert elev init move 1 failed"
                return response

            self.prepare_motor(motor_obj)
            success = self.move_motor(motor_obj, 12.5, kp, kd, start=0, ramp_time = 0.8)
            if not success:
                response.success = False
                response.message = "Vert elev init move 2 failed"
                return response

            motor_obj['current_state'] = 12.5
            motor_obj['initialized'] = True
            response.success = True
            response.message = "Vert elev initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response

    def init_gripper_catch_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['gripper_catch']['initialized']:
            response.success = True
            response.message = "Gripper catch already initialized"
            return response

        try:
            motor_obj = self.motors["gripper_catch"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Gripper catch motor unavailable"
                return response

            gains = motor_obj["gains"]["up"]
            kp, kd = gains["kp"], gains["kd"]

            if not self.prepare_motor(motor_obj):
                response.success = False
                response.message = "Gripper catch prepare failed"
                return response

            if not self.check_motor_connection(motor_obj, "gripper_catch"):  # or whichever motor
                response.success = False
                response.message = "Gripper catch motor not responding (no feedback)"
                return response

            

            # success = self.move_motor(motor_obj, 12.5, kp, kd, start=0, ramp_time = 0.8)
            # if not success:
            #     response.success = False
            #     response.message = "Gripper catch init move 1 failed"
            #     return response

            # self.prepare_motor(motor_obj)

            # success = self.move_motor(motor_obj, -12.5, kp, kd, start=0, ramp_time = 0.8)
            # if not success:
            #     response.success = False
            #     response.message = "Gripper catch init move 2 failed"
            #     return response

            motor_obj['initialized'] = True
            motor_obj['current_state'] = 0
            response.success = True
            response.message = "Gripper catch initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response
    
    def init_gripper_pitch_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['gripper_pitch']['initialized']:
            response.success = True
            response.message = "Gripper pitch already initialized"
            return response

        try:
            motor_obj = self.motors["gripper_pitch"]
            if motor_obj["handle"] is None:
                response.success = False
                response.message = "Gripper pitch motor unavailable"
                return response

            gains = motor_obj["gains"]["up"]
            kp, kd = gains["kp"], gains["kd"]

            if not self.prepare_motor(motor_obj):
                response.success = False
                response.message = "Gripper pitch prepare failed"
                return response
            
            if not self.check_motor_connection(motor_obj, "gripper_pitch"):  # or whichever motor
                response.success = False
                response.message = "Gripper pitch motor not responding (no feedback)"
                return response
                
            # motor_obj["handle"].send(p=0.0, v=0.0, kp=30.0, kd=1.3, t=0.0)
            # time.sleep(0.05)
            # success = self.move_motor(motor_obj, 12.5, kp, kd, start=0, ramp_time = 0.8)
            # if not success:
            #     response.success = False
            #     response.message = "Gripper pitch init move 1 failed"
            #     return response

            # self.prepare_motor(motor_obj)

            # success = self.move_motor(motor_obj, -12.5, kp, kd, start=0, ramp_time = 0.8)
            # if not success:
            #     response.success = False
            #     response.message = "Gripper pitch init move 2 failed"
            #     return response

            motor_obj['initialized'] = True
            motor_obj['current_state'] = 0
            response.success = True
            response.message = "Gripper pitch initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response

    def init_spearhead_catch_callback(self, request, response):
        motor_obj = self.motors["spearhead_catch"]
        motor = motor_obj["handle"]

        if motor is None:
            response.success = False
            response.message = "Spearhead catch motor unavailable"
            return response

        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        if self.motors['spearhead_catch']['initialized']:
            response.success = True
            response.message = "Spearhead catch already initialized"
            return response

        try:
            motor.exit_mit()
            time.sleep(0.01)
            motor.enter_mit()
            time.sleep(motor_obj.get("enter_delay", 0.05))

            if not self.check_motor_connection(motor_obj, "spearhead_catch"):  
                response.success = False
                response.message = "Spearhead catch motor not responding (no feedback)"
                return response

            motor.send(
                p=1.5,
                v=0.0,
                kp=0.0,
                kd=0.0,
                t=0.0
            )

            motor_obj['initialized'] = True
            response.success = True
            response.message = "Spearhead catch initialization complete"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response

    def _test_single_motor(self, name, motor_obj):
        """
        Test one motor and return (bool, str) = (ok, status_line).

        Rules:
          - spearhead_catch (ak, id=9): send all-zeros only, no exit/zero/enter
          - all others:  exit → zero_position → enter → send all-zeros → read
        """
        motor = motor_obj["handle"]
        if motor is None:
            return False, f"  {name:<16} [HANDLE MISSING]"

        skip_sequence = (name == "spearhead_catch")

        try:
            if not skip_sequence:
                motor.exit_mit()
                time.sleep(0.05)
                motor.zero_position()
                time.sleep(0.05)
                motor.enter_mit()
                time.sleep(motor_obj.get("enter_delay", 0.05))

            # send all-zeros
            motor.send(p=0.0, v=0.0, kp=0.0, kd=0.0, t=0.0)

            # try up to 10 times for feedback
            feedback = None
            for attempt in range(10):
                time.sleep(0.02)
                feedback = motor.read()
                if feedback is not None:
                    break

            if feedback is not None and "pos" in feedback:
                pos = feedback["pos"]
                return True, f"  {name:<16} OK      pos={pos:.3f}"
            else:
                return False, f"  {name:<16} NO FEEDBACK"

        except Exception as e:
            return False, f"  {name:<16} ERROR   {type(e).__name__}: {e}"

    def test_all_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "Request data false"
            return response

        lines = ["", "===== MOTOR BUS TEST ====="]
        all_ok = True

        for name, motor_obj in self.motors.items():
            ok, line = self._test_single_motor(name, motor_obj)
            lines.append(line)
            if not ok:
                all_ok = False
            time.sleep(0.1)   # brief gap between motors

        lines.append("==========================")
        lines.append(f"RESULT: {'ALL OK' if all_ok else 'SOME MOTORS FAILED'}")
        lines.append("")

        response.success = all_ok
        response.message = "\n".join(lines)
        return response

    def shutdown_motor(self, name):
        """Run the full shutdown sequence for a single motor."""
        m = self.motors[name]
        motor = m["handle"]

        if motor is None:
            self.get_logger().warn(f"{name} motor handle is None, skipping shutdown")
            return

        try:
            time.sleep(0.1)
            motor.send(p=0.0, v=0.0, kp=0.0, kd=0.0, t=0.0)
            time.sleep(0.01)
            if name == 'spearhead_catch':
                return
            motor.exit_mit()
            # time.sleep(0.01)
            # motor.zero_position()
            # time.sleep(0.01)
            # motor.enter_mit()
            # time.sleep(0.05)
            self.get_logger().info(f"{name} shutdown sequence complete")
        except Exception as e:
            self.get_logger().warn(f"{name} shutdown sequence failed: {e}")

    def destroy_node(self):
        print("[INFO] Shutting down AKMotorService...")
    
        for name in self.motors:
            print(f"[INFO] Running shutdown sequence for {name}")
            self.shutdown_motor(name)
    
        if self.can:
            try:
                print("[INFO] Closing CAN device")
                self.can.close()
            except Exception as e:
                print(f"[WARN] CAN close failed: {e}")
    
        super().destroy_node()

def main():

    rclpy.init()

    node = None

    try:

        node = AKMotorService()

        if rclpy.ok():
            rclpy.spin(node)

    except KeyboardInterrupt:

        print("\n[INFO] Keyboard interrupt received")

    except Exception as e:

        print(f"[ERROR] Fatal error: {e}")

    finally:

        if node is not None:

            try:
                node.destroy_node()

            except Exception as e:

                print(
                    f"[WARN] destroy_node failed: {e}"
                )

        try:

            if rclpy.ok():
                rclpy.shutdown()

        except Exception as e:

            print(
                f"[WARN] shutdown failed: {e}"
            )


if __name__ == '__main__':
    main()