#!/usr/bin/env python3
"""
Navigation controller with A* GLOBAL PLANNER + PURE PURSUIT + Y-turn + ARC-BASED
OBSTACLE AVOIDANCE.

Global planner (A* on the Cartographer /map):
- Obstacles on the map, plus what the LiDAR sees right now, are grown by half the car
  width + margin. A* finds the cheapest path through known free space, keeping away
  from walls where it can. The path is published on /planned_path (Foxglove).
- Pure pursuit aims pp_lookahead_m ahead along that path instead of straight at the
  goal. The last stretch, reversing and all obstacle avoidance work as before.
- Replans on a new goal, every replan_period_s, after an avoid move, and when the car
  is more than 0.5 m off the path. No map / no path: drive straight at the goal.

Obstacle avoidance (forward and reverse):
- LiDAR points are moved into base_link with the same TF Cartographer uses
  (base_link <- laser, LiDAR mounted with 180° roll), so they share the frame
  pure pursuit steers in: +curvature = left = servo_left_us.
- Every control tick the arc pure pursuit wants is checked: how far can the car
  body drive along that arc before touching a LiDAR point (free length)?
- Free -> plain pure pursuit.
- Blocked -> candidate arcs from full left to full right lock are checked and the
  clear one closest to the pure pursuit arc is driven, keeping the same avoid side
  while blocked. Arcs are judged along the curve the car is actually driving, so
  this also works while turning: already at full left lock, the car knows there
  is no "more left" and goes around on the right, or backs up.
- Every arc blocked -> back up a little, steering so the nose swings to open space.
- Reversing works the same way on the rear, including the nose swinging out on a
  tight reverse turn; every reverse arc blocked -> pull forward a little. Y-turn
  backups stop early when something is behind.
- Goal right next to an obstacle: that obstacle is left out of the avoidance (all
  others still count) and the car stops in front of it as close as is safe,
  instead of detouring around it or giving up.
- Pure pursuit aims at most pp_lookahead_m along the line to the goal: the car turns
  toward the goal, then drives straight, instead of sweeping a wide circle. Goal more
  than 90° to the side: full lock. Y-turn only when the goal is inside the min
  turning circle.

Features:
- Reverse Pure Pursuit: Full steering control when reversing
- Hysteresis rear zone: Enter at 130°, exit at 45°, only for a goal closer than
  reverse_max_distance_m. A goal farther behind: turn around driving forward at
  full lock, with short obstacle-checked backups (a K-turn) when the turn is blocked
- Deadzone detection for both forward and reverse
- Adaptive Y-turn as fallback
"""

import heapq
import math
import threading
import time
from enum import Enum

import numpy as np
import rclpy
import serial
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


# Rear zone thresholds with HYSTERESIS
REAR_ZONE_ENTER_RAD = 2.2689  # 130 degrees - enter reverse
REAR_ZONE_EXIT_RAD = 0.7854   # 45 degrees - exit reverse (lowered to prevent premature exit)
REVERSE_EXIT_MARGIN_M = 0.3   # leave reverse once the goal is this much beyond reverse_max_distance_m

# Obstacle avoidance
OBSTACLE_DISTANCE_M = 0.60      # Free length needed along the arc ahead
AVOID_BACKUP_TIMEOUT_S = 8.0    # Give up an avoid backup that makes no progress
GOAL_STOP_FREE_M = 0.05         # Driving in to a goal next to an obstacle: stop below this free length
TURN_SIDE_BY_ROOM_RAD = math.radians(150)  # goal nearly straight behind: turn around on the roomier side

# Global planner (A* on /map)
MAP_FREE = 25                   # /map value <= this: free; >= MAP_OCCUPIED: obstacle; between: barely seen
MAP_OCCUPIED = 65
PLAN_SOFT_CLEARANCE_M = 0.40    # paths prefer to stay this far from obstacles...
PLAN_WALL_COST = 3.0            # ...a cell right at the lethal distance costs this much extra
PLAN_UNCERTAIN_COST = 2.0       # barely-seen cells cost this many times more
PLAN_SNAP_M = 0.5               # start / goal inside an obstacle: use the nearest free cell this close
PLAN_MAX_EXPANSIONS = 60000     # give up A* after this many cells
REPLAN_DEVIATION_M = 0.5        # replan when the car is this far off the path
SQRT2 = math.sqrt(2.0)
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))


class ManeuverMode(Enum):
    """Maneuver state machine"""
    NORMAL = 1          # Pure Pursuit navigation
    YTURN_BACKUP = 2    # Y-turn: backing up
    YTURN_TURN = 3      # Y-turn: turning
    YTURN_FORWARD = 4   # Y-turn: moving forward
    AVOID_BACKUP = 5    # Obstacle avoidance: every arc blocked, moving a little the other way


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(rotation):
    return math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )


def tick_delta(new_value, old_value):
    return (new_value - old_value + 2**31) % 2**32 - 2**31


class GridPlanner:
    """
    A* on an occupancy grid. The car is a point: obstacles are grown by lethal_radius
    (half the car width + margin), and cells closer than soft_radius to an obstacle
    cost more, so paths keep to the middle of free space. Unknown cells are not used.
    """

    def __init__(self, lethal_radius, soft_radius):
        self.lethal_radius = lethal_radius
        self.soft_radius = soft_radius
        self.grid = None

    def set_map(self, data, width, height, resolution, origin_x, origin_y):
        self.grid = np.frombuffer(data, dtype=np.int8).reshape(height, width).astype(np.int16)
        self.width, self.height = width, height
        self.res = resolution
        self.origin_x, self.origin_y = origin_x, origin_y

    def to_cell(self, x, y):
        return int(math.floor((x - self.origin_x) / self.res)), int(math.floor((y - self.origin_y) / self.res))

    def to_world(self, i, j):
        return self.origin_x + (i + 0.5) * self.res, self.origin_y + (j + 0.5) * self.res

    def build_costs(self, extra_obstacles):
        """Flat lists: step cost per cell (0 = cannot enter) and clearance in cells."""
        grid = self.grid
        occupied = grid >= MAP_OCCUPIED
        for x, y in extra_obstacles:
            i, j = self.to_cell(x, y)
            if 0 <= i < self.width and 0 <= j < self.height:
                occupied[j, i] = True

        # Clearance by growing the obstacles one cell at a time, alternating 4- and
        # 8-neighbour steps so the grown shape stays close to a circle
        soft = int(math.ceil(self.soft_radius / self.res))
        clear = np.full(grid.shape, soft + 1, dtype=np.float32)
        clear[occupied] = 0.0
        reached = occupied.copy()
        for r in range(1, soft + 1):
            grown = reached.copy()
            grown[1:, :] |= reached[:-1, :]
            grown[:-1, :] |= reached[1:, :]
            grown[:, 1:] |= reached[:, :-1]
            grown[:, :-1] |= reached[:, 1:]
            if r % 2 == 0:
                grown[1:, 1:] |= reached[:-1, :-1]
                grown[1:, :-1] |= reached[:-1, 1:]
                grown[:-1, 1:] |= reached[1:, :-1]
                grown[:-1, :-1] |= reached[1:, 1:]
            clear[grown & ~reached] = r
            reached = grown

        clear_m = clear * self.res
        span = self.soft_radius - self.lethal_radius
        cost = 1.0 + PLAN_WALL_COST * np.clip((self.soft_radius - clear_m) / span, 0.0, 1.0) ** 2
        cost[(grid > MAP_FREE) & (grid < MAP_OCCUPIED)] *= PLAN_UNCERTAIN_COST
        cost[(clear_m <= self.lethal_radius) | (grid < 0)] = 0.0
        return cost.ravel().tolist(), clear.ravel().tolist()

    def nearest_free(self, cost, i, j, reach):
        """Enterable cell closest to (i, j) within `reach` cells, or None."""
        w, h = self.width, self.height
        best, best_d2 = None, None
        for dj in range(-reach, reach + 1):
            for di in range(-reach, reach + 1):
                ci, cj = i + di, j + dj
                if 0 <= ci < w and 0 <= cj < h and cost[cj * w + ci] > 0.0:
                    d2 = di * di + dj * dj
                    if best is None or d2 < best_d2:
                        best, best_d2 = (ci, cj), d2
        return best

    def astar(self, cost, start, goal):
        """Cheapest 8-connected cell path start -> goal as flat indices, or None."""
        w, h = self.width, self.height
        n = w * h
        inf = float("inf")
        g = [inf] * n
        parent = [-1] * n
        closed = bytearray(n)
        s = start[1] * w + start[0]
        t = goal[1] * w + goal[0]
        gi, gj = goal
        g[s] = 0.0
        heap = [(0.0, s)]
        expansions = 0
        while heap:
            _, u = heapq.heappop(heap)
            if u == t:
                break
            if closed[u]:
                continue
            closed[u] = 1
            expansions += 1
            if expansions > PLAN_MAX_EXPANSIONS:
                return None
            ui, uj = u % w, u // w
            gu = g[u]
            for di, dj, step in NEIGHBOURS:
                vi, vj = ui + di, uj + dj
                if vi < 0 or vj < 0 or vi >= w or vj >= h:
                    continue
                v = vj * w + vi
                c = cost[v]
                if c <= 0.0 or closed[v]:
                    continue
                if di and dj and (cost[uj * w + vi] <= 0.0 or cost[vj * w + ui] <= 0.0):
                    continue  # do not cut a blocked corner
                ng = gu + step * c
                if ng < g[v]:
                    g[v] = ng
                    parent[v] = u
                    hi, hj = abs(vi - gi), abs(vj - gj)
                    heapq.heappush(heap, (ng + hi + hj + (SQRT2 - 2.0) * min(hi, hj), v))
        if g[t] == inf:
            return None
        cells = []
        while t != -1:
            cells.append(t)
            t = parent[t]
        return cells[::-1]

    def line_ok(self, a, b, cost, clear, min_clear):
        """Straight line a -> b only crosses enterable cells with at least min_clear."""
        w = self.width
        ai, aj, bi, bj = a % w, a // w, b % w, b // w
        steps = 2 * max(abs(bi - ai), abs(bj - aj)) + 1
        for k in range(1, steps):
            f = k / steps
            c = int(round(aj + (bj - aj) * f)) * w + int(round(ai + (bi - ai) * f))
            if cost[c] <= 0.0 or clear[c] < min_clear:
                return False
        return True

    def smooth(self, cells, cost, clear):
        """Drop waypoints while a straight line keeps the clearance A* had along there."""
        out = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            best = anchor + 1
            min_clear = min(clear[cells[anchor]], clear[cells[best]])
            for k in range(anchor + 2, len(cells)):
                min_clear = min(min_clear, clear[cells[k]])
                if not self.line_ok(cells[anchor], cells[k], cost, clear, min_clear):
                    break
                best = k
            out.append(cells[best])
            anchor = best
        return out

    def plan(self, start_xy, goal_xy, extra_obstacles=()):
        """Waypoints [(x, y), ...] in the map frame from start to goal, or None."""
        if self.grid is None:
            return None
        cost, clear = self.build_costs(extra_obstacles)
        reach = int(math.ceil(PLAN_SNAP_M / self.res))
        start = self.nearest_free(cost, *self.to_cell(*start_xy), reach)
        goal = self.nearest_free(cost, *self.to_cell(*goal_xy), reach)
        if start is None or goal is None:
            return None
        cells = self.astar(cost, start, goal)
        if cells is None:
            return None
        waypoints = [self.to_world(c % self.width, c // self.width) for c in self.smooth(cells, cost, clear)]
        waypoints[0] = tuple(start_xy)
        if goal == self.to_cell(*goal_xy):
            waypoints[-1] = tuple(goal_xy)
        return waypoints


class GoalPoseDrive(Node):
    def __init__(self):
        super().__init__("goal_pose_drive")

        # Hardware parameters
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("ticks_per_revolution", 329.8)
        self.declare_parameter("wheel_diameter_m", 0.065)
        self.declare_parameter("wheelbase_m", 0.20)
        self.declare_parameter("max_steering_angle_deg", 40.0)
        self.declare_parameter("servo_center_us", 2125)
        self.declare_parameter("servo_left_us", 1650)
        self.declare_parameter("servo_right_us", 2600)
        self.declare_parameter("encoder_forward_sign", -1)
        self.declare_parameter("slow_pwm", 40)
        self.declare_parameter("arrival_tolerance_m", 0.18)
        self.declare_parameter("pp_lookahead_m", 0.6)  # pure pursuit aims at most this far along the line to the goal
        self.declare_parameter("reverse_max_distance_m", 0.6)  # goal behind and closer than this: reverse; farther: turn around
        self.declare_parameter("encoder_timeout_s", 0.5)
        
        # Y-turn parameters
        self.declare_parameter("enable_yturn", True)
        self.declare_parameter("theta_max_deg", 90.0)
        self.declare_parameter("free_front_m", 2.0)
        self.declare_parameter("yturn_backup_pwm", 35)
        self.declare_parameter("yturn_turn_pwm", 35)
        self.declare_parameter("yturn_turn_duration_s", 2.0)
        self.declare_parameter("yturn_backup_distance_min_m", 0.15)
        self.declare_parameter("yturn_backup_distance_max_m", 0.6)
        self.declare_parameter("enable_adaptive_backup", True)
        self.declare_parameter("yturn_speed_estimate_mps", 0.07)
        self.declare_parameter("yturn_min_distance_m", 0.5)
        self.declare_parameter("yturn_cooldown_s", 3.0)
        self.declare_parameter("yturn_max_per_goal", 4)

        # Obstacle avoidance parameters
        self.declare_parameter("enable_obstacle_avoid", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("scan_timeout_s", 1.0)
        # Extra LiDAR yaw on top of the TF: the car front is 5° off the TF x-axis
        # (same value as front_angle_deg in simple_obstacle_turn_2.py)
        self.declare_parameter("lidar_yaw_trim_deg", 5.0)
        # Car body, measured from the rear axle (the point the bicycle model turns about)
        self.declare_parameter("rear_axle_x_m", -0.10)   # rear axle x in base_link: LiDAR sits 10 cm ahead of it
        self.declare_parameter("car_width_m", 0.20)
        self.declare_parameter("car_front_m", 0.23)      # rear axle -> front bumper
        self.declare_parameter("car_rear_m", 0.04)       # rear axle -> rear bumper
        self.declare_parameter("avoid_margin_m", 0.05)   # extra clearance around the body
        self.declare_parameter("obstacle_distance_m", OBSTACLE_DISTANCE_M)
        self.declare_parameter("avoid_min_free_m", 0.15)  # an arc shorter than this is unusable
        self.declare_parameter("avoid_num_arcs", 21)
        self.declare_parameter("avoid_hold_s", 1.0)       # keep the avoid side this long after clear
        self.declare_parameter("avoid_backup_m", 0.20)

        # Global planner parameters
        self.declare_parameter("enable_global_planner", True)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("replan_period_s", 3.0)

        # Get parameters
        self.ticks_per_revolution = float(self.get_parameter("ticks_per_revolution").value)
        self.wheel_diameter_m = float(self.get_parameter("wheel_diameter_m").value)
        self.wheelbase_m = float(self.get_parameter("wheelbase_m").value)
        self.max_steering_angle = math.radians(float(self.get_parameter("max_steering_angle_deg").value))
        self.servo_center_us = int(self.get_parameter("servo_center_us").value)
        self.servo_left_us = int(self.get_parameter("servo_left_us").value)
        self.servo_right_us = int(self.get_parameter("servo_right_us").value)
        self.encoder_forward_sign = int(self.get_parameter("encoder_forward_sign").value)
        self.slow_pwm = int(self.get_parameter("slow_pwm").value)
        self.arrival_tolerance_m = float(self.get_parameter("arrival_tolerance_m").value)
        self.pp_lookahead_m = float(self.get_parameter("pp_lookahead_m").value)
        self.reverse_max_distance = float(self.get_parameter("reverse_max_distance_m").value)
        self.encoder_timeout_s = float(self.get_parameter("encoder_timeout_s").value)
        
        # Y-turn parameters
        self.enable_yturn = bool(self.get_parameter("enable_yturn").value)
        self.theta_max = math.radians(float(self.get_parameter("theta_max_deg").value))
        self.free_front_m = float(self.get_parameter("free_front_m").value)
        self.yturn_backup_pwm = int(self.get_parameter("yturn_backup_pwm").value)
        self.yturn_turn_pwm = int(self.get_parameter("yturn_turn_pwm").value)
        self.yturn_turn_duration = float(self.get_parameter("yturn_turn_duration_s").value)
        self.yturn_min_distance = float(self.get_parameter("yturn_min_distance_m").value)
        self.yturn_cooldown = float(self.get_parameter("yturn_cooldown_s").value)
        self.yturn_max_per_goal = int(self.get_parameter("yturn_max_per_goal").value)
        self.yturn_backup_distance_min = float(self.get_parameter("yturn_backup_distance_min_m").value)
        self.yturn_backup_distance_max = float(self.get_parameter("yturn_backup_distance_max_m").value)
        self.enable_adaptive_backup = bool(self.get_parameter("enable_adaptive_backup").value)
        self.yturn_speed_estimate = float(self.get_parameter("yturn_speed_estimate_mps").value)

        # Obstacle avoidance parameters
        self.enable_obstacle_avoid = bool(self.get_parameter("enable_obstacle_avoid").value)
        self.scan_timeout = float(self.get_parameter("scan_timeout_s").value)
        self.lidar_yaw_trim = math.radians(float(self.get_parameter("lidar_yaw_trim_deg").value))
        self.rear_axle_x = float(self.get_parameter("rear_axle_x_m").value)
        self.car_width = float(self.get_parameter("car_width_m").value)
        self.car_front = float(self.get_parameter("car_front_m").value)
        self.car_rear = float(self.get_parameter("car_rear_m").value)
        self.avoid_margin = float(self.get_parameter("avoid_margin_m").value)
        self.obstacle_distance = float(self.get_parameter("obstacle_distance_m").value)
        self.avoid_min_free = float(self.get_parameter("avoid_min_free_m").value)
        self.avoid_hold_s = float(self.get_parameter("avoid_hold_s").value)
        self.avoid_backup_m = float(self.get_parameter("avoid_backup_m").value)
        self.enable_global_planner = bool(self.get_parameter("enable_global_planner").value)
        self.replan_period = float(self.get_parameter("replan_period_s").value)
        
        # Calculate min turning radius
        self.R_min = self.wheelbase_m / math.tan(self.max_steering_angle)
        self.kappa_max = 1.0 / self.R_min

        # Obstacle avoidance geometry: inflated car body and candidate arcs
        self.body_half_width = 0.5 * self.car_width + self.avoid_margin
        self.avoid_front = self.car_front + self.avoid_margin
        self.avoid_rear = self.car_rear + self.avoid_margin
        self.scan_max_range = self.obstacle_distance + 1.0
        num_arcs = max(3, int(self.get_parameter("avoid_num_arcs").value))
        if num_arcs % 2 == 0:
            num_arcs += 1  # odd, so driving straight is one of the candidates
        self.avoid_arcs = [self.kappa_max * (2.0 * i / (num_arcs - 1) - 1.0) for i in range(num_arcs)]
        self.avoid_arc_step = 2.0 * self.kappa_max / (num_arcs - 1)
        # LiDAR points closer than this to the goal leave no room for the car to stand on it
        lidar_to_end = max(self.car_front + self.rear_axle_x, self.car_rear - self.rear_axle_x)
        self.goal_clear_radius = math.hypot(lidar_to_end, 0.5 * self.car_width) + self.avoid_margin
        self.planner = GridPlanner(self.body_half_width, PLAN_SOFT_CLEARANCE_M)

        # Serial connection
        serial_port = self.get_parameter("serial_port").value
        baudrate = int(self.get_parameter("baudrate").value)
        self.uart = serial.Serial(serial_port, baudrate, timeout=0.1, write_timeout=0.2)
        self.uart_lock = threading.Lock()
        self.encoder_lock = threading.Lock()
        self.encoder_ticks = None
        self.encoder_servo_us = self.servo_center_us
        self.last_encoder_time = None
        self.serial_error = None

        # State variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = -2.87
        self.previous_ticks = None
        self.previous_odom_time = None
        self.goal = None
        self.last_command = None
        self.last_drive_sent_at = 0.0
        
        # Y-turn state
        self.maneuver_mode = ManeuverMode.NORMAL
        self.yturn_start_pos = None
        self.yturn_start_time = None
        self.yturn_direction = 1
        self.last_yturn_time = 0.0
        self.yturn_count = 0
        self.yturn_consecutive = 0  # Y-turns done for the current goal
        
        # Reverse mode state
        self.currently_reversing = False

        # Obstacle avoidance state
        self.last_scan_time = None
        self.scan_points = []          # LiDAR points (x forward, y left) in the rear-axle frame
        self.laser_transform = None    # (m00, m01, m10, m11, ox, oy): laser -> rear-axle frame
        self.scan_rays = []            # unit vector per scan index, cached
        self.scan_rays_key = None
        self.avoid_side = 0            # +1 / -1: going around on the left / right of the PP arc
        self.avoid_last_blocked = 0.0
        self.avoid_backup_count = 0    # avoid backups for the current goal
        self.avoid_backup_start = None
        self.avoid_backup_started_at = 0.0
        self.avoid_backup_distance = 0.0
        self.avoid_backup_kappa = 0.0
        self.avoid_backup_reverse = True  # direction of that move (True = backing up)
        self.avoid_reverse = False        # avoid_side refers to reversing
        self.turnaround_side = 0          # +1 / -1 while turning around (left / right) toward a goal behind

        # Global path state
        self.path = None                  # A* waypoints [(x, y), ...] in map; None: steer straight at the goal
        self.path_index = 0               # path segment the car is on
        self.need_plan = False
        self.last_plan_time = 0.0

        # ROS2 setup
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.goal_callback, 10)
        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value,
            self.scan_callback, qos_profile_sensor_data,
        )
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, self.get_parameter("map_topic").value, self.map_callback, latched)
        self.path_pub = self.create_publisher(Path, "/planned_path", latched)
        self.create_timer(0.02, self.update_odometry)
        self.create_timer(0.1, self.control_goal)

        self.send_command("S", force=True)
        self.encoder_thread = threading.Thread(target=self.read_encoder_telemetry, daemon=True)
        self.encoder_thread.start()

        self.get_logger().info(
            f"🚗 Reverse Pure Pursuit Controller Started\n"
            f"   Serial: {serial_port} @ {baudrate}\n"
            f"   Y-turn: {'enabled' if self.enable_yturn else 'disabled'}\n"
            f"   Global planner: {'A* on /map' if self.enable_global_planner else 'disabled'}\n"
            f"   Obstacle avoid: {'enabled' if self.enable_obstacle_avoid else 'disabled'} "
            f"({len(self.avoid_arcs)} arcs, look {self.obstacle_distance:.2f}m, "
            f"body {self.car_width:.2f}m wide +{self.avoid_margin:.2f}m margin)\n"
            f"   R_min: {self.R_min:.3f}m, θ_max: {math.degrees(self.theta_max):.0f}°\n"
            f"   Rear zone: Enter={math.degrees(REAR_ZONE_ENTER_RAD):.0f}°, "
            f"Exit={math.degrees(REAR_ZONE_EXIT_RAD):.0f}°, reverse only within {self.reverse_max_distance:.2f}m"
        )

    def send_command(self, command, force=False):
        now = time.monotonic()
        if not force and command == self.last_command:
            if command == "S" or now - self.last_drive_sent_at < 0.1:
                return
        try:
            with self.uart_lock:
                self.uart.write((command + "\n").encode("ascii"))
                self.uart.flush()
        except (serial.SerialException, OSError) as error:
            self.serial_error = str(error)
            self.goal = None
            self.get_logger().error(f"UART command failed: {error}")
            return
        self.last_command = command
        self.last_drive_sent_at = now

    def stop(self):
        self.goal = None
        self.maneuver_mode = ManeuverMode.NORMAL
        self.currently_reversing = False
        self.avoid_side = 0
        self.turnaround_side = 0
        if self.path is not None:
            self.path = None
            self.publish_path()
        self.send_command("S", force=True)

    def read_encoder_telemetry(self):
        while rclpy.ok():
            try:
                line = self.uart.readline().decode("ascii", errors="ignore").strip()
            except (serial.SerialException, OSError) as error:
                self.serial_error = str(error)
                return

            fields = line.split(",")
            if len(fields) != 4 or fields[0] != "E":
                continue
            try:
                ticks = (int(fields[1]), int(fields[2]))
                servo_us = int(fields[3])
            except ValueError:
                continue
            with self.encoder_lock:
                self.encoder_ticks = ticks
                self.encoder_servo_us = servo_us
                self.last_encoder_time = time.monotonic()

    def scan_callback(self, scan):
        """Convert the scan to points (x forward, y left) in the rear-axle frame."""
        if self.laser_transform is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "base_link", scan.header.frame_id, rclpy.time.Time()
                )
            except TransformException as error:
                self.get_logger().warning(
                    f"No TF base_link <- {scan.header.frame_id} yet: {error}", throttle_duration_sec=2.0
                )
                return
            q = transform.transform.rotation
            t = transform.transform.translation
            # 2D part of the 3D rotation; keeps the y flip of the 180°-rolled LiDAR
            r00 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            r01 = 2.0 * (q.x * q.y - q.z * q.w)
            r10 = 2.0 * (q.x * q.y + q.z * q.w)
            r11 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
            # Then turn by the yaw trim (real car front) and move the origin to the rear axle
            c = math.cos(self.lidar_yaw_trim)
            s = math.sin(self.lidar_yaw_trim)
            self.laser_transform = (
                c * r00 - s * r10, c * r01 - s * r11,
                s * r00 + c * r10, s * r01 + c * r11,
                c * t.x - s * t.y - self.rear_axle_x, s * t.x + c * t.y,
            )
            self.scan_rays_key = None
            self.get_logger().info(
                f"LiDAR TF ready: {'y flipped (180° roll)' if r00 * r11 - r01 * r10 < 0 else 'no flip'}, "
                f"yaw trim {math.degrees(self.lidar_yaw_trim):+.1f}°"
            )

        m00, m01, m10, m11, ox, oy = self.laser_transform
        key = (scan.angle_min, scan.angle_increment, len(scan.ranges))
        if key != self.scan_rays_key:
            self.scan_rays = []
            for index in range(len(scan.ranges)):
                angle = scan.angle_min + index * scan.angle_increment
                ca, sa = math.cos(angle), math.sin(angle)
                self.scan_rays.append((m00 * ca + m01 * sa, m10 * ca + m11 * sa))
            self.scan_rays_key = key

        max_range = min(scan.range_max, self.scan_max_range)
        half_width = 0.5 * self.car_width
        points = []
        for (ux, uy), distance in zip(self.scan_rays, scan.ranges):
            if not scan.range_min <= distance <= max_range:
                continue  # also drops inf / nan
            x = ox + distance * ux
            y = oy + distance * uy
            if -self.car_rear <= x <= self.car_front and -half_width <= y <= half_width:
                continue  # inside the car itself
            points.append((x, y))
        self.scan_points = points
        self.last_scan_time = time.monotonic()

    def map_callback(self, msg):
        info = msg.info
        first = self.planner.grid is None
        self.planner.set_map(msg.data, info.width, info.height, info.resolution,
                             info.origin.position.x, info.origin.position.y)
        if first:
            self.get_logger().info(
                f"🗺️  /map received: {info.width}x{info.height} cells of {info.resolution:.2f} m"
            )

    def goal_callback(self, message):
        if message.header.frame_id.strip("/") != "map":
            self.get_logger().error(f"Goal frame must be 'map', got '{message.header.frame_id}'. Stopping.")
            self.stop()
            return
        self.goal = (message.pose.position.x, message.pose.position.y)
        self.maneuver_mode = ManeuverMode.NORMAL
        self.currently_reversing = False  # Reset on new goal
        self.avoid_side = 0
        self.avoid_backup_count = 0
        self.turnaround_side = 0
        self.path = None
        self.need_plan = True
        self.yturn_consecutive = 0
        self.get_logger().info(f"🎯 Goal received: x={self.goal[0]:.2f}, y={self.goal[1]:.2f}")

    def update_odometry(self):
        with self.encoder_lock:
            ticks = self.encoder_ticks
            servo_us = self.encoder_servo_us
            encoder_time = self.last_encoder_time

        if (ticks is None or encoder_time is None or 
            time.monotonic() - encoder_time > self.encoder_timeout_s):
            return

        now = self.get_clock().now()
        linear_velocity = 0.0
        angular_velocity = 0.0
        
        if self.previous_ticks is not None and self.previous_odom_time is not None:
            delta_left = tick_delta(ticks[0], self.previous_ticks[0])
            delta_right = tick_delta(ticks[1], self.previous_ticks[1])
            dt = (now - self.previous_odom_time).nanoseconds / 1e9
            
            if dt > 0.0:
                meters_per_tick = math.pi * self.wheel_diameter_m / self.ticks_per_revolution
                distance = ((delta_left + delta_right) * 0.5 * meters_per_tick * self.encoder_forward_sign)
                
                pulse_delta = servo_us - self.servo_center_us
                if pulse_delta < 0:
                    fraction = pulse_delta / (self.servo_center_us - self.servo_left_us)
                else:
                    fraction = pulse_delta / (self.servo_right_us - self.servo_center_us)
                steering_angle = max(-1.0, min(1.0, -fraction)) * self.max_steering_angle
                
                yaw_delta = distance * math.tan(steering_angle) / self.wheelbase_m
                midpoint_yaw = self.yaw + yaw_delta * 0.5
                self.x += distance * math.cos(midpoint_yaw)
                self.y += distance * math.sin(midpoint_yaw)
                self.yaw = normalize_angle(self.yaw + yaw_delta)
                linear_velocity = distance / dt
                angular_velocity = yaw_delta / dt

        self.previous_ticks = ticks
        self.previous_odom_time = now
        
        qz = math.sin(self.yaw * 0.5)
        qw = math.cos(self.yaw * 0.5)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = angular_velocity
        self.odom_pub.publish(odom)

    def one_arc_reachable(self, xb, yb):
        """Check if goal (xb, yb) in base_link frame is reachable by one arc."""
        d2 = xb * xb + yb * yb
        
        if d2 < 1e-9:
            return True, 0.0, 0.0, 'AT_GOAL'
        
        if abs(yb) < 1e-9:
            if 0 < xb <= self.free_front_m:
                return True, 0.0, xb, 'STRAIGHT'
            else:
                return False, None, None, 'OUT_OF_RANGE'
        
        c = d2 / (2 * abs(yb))
        
        if c < self.R_min:
            return False, None, None, 'DEAD_ZONE'
        
        theta = 2 * math.atan2(abs(yb), xb)
        if theta < 0:
            theta += 2 * math.pi
        
        s = c * theta
        
        if theta > self.theta_max:
            return False, theta, s, 'ANGLE_TOO_LARGE'
        
        if s > self.free_front_m:
            return False, theta, s, 'ARC_TOO_LONG'
        
        return True, theta, s, 'REACHABLE'

    def should_reverse(self, vehicle_angle_rad, distance_to_goal):
        """
        Determine if should use reverse mode with hysteresis + distance check.
        
        Args:
            vehicle_angle_rad: Angle from vehicle X-axis to goal (radians)
            distance_to_goal: Distance to goal (meters)
            
        Returns:
            bool: True if should reverse
        """
        abs_angle = abs(vehicle_angle_rad)
        
        if self.currently_reversing:
            # Exit reverse when the goal is clearly in front (angle < 45 deg).
            # The old distance condition kept the car in reverse when the goal was close.
            if abs_angle < REAR_ZONE_EXIT_RAD:
                self.currently_reversing = False
                self.get_logger().info(
                    f"🔄 Exit REVERSE: angle={math.degrees(vehicle_angle_rad):+.1f}° < 45°"
                )
                return False
            if distance_to_goal > self.reverse_max_distance + REVERSE_EXIT_MARGIN_M:
                self.currently_reversing = False
                self.get_logger().info(
                    f"🔄 Exit REVERSE: goal {distance_to_goal:.2f}m away, turning around forward instead"
                )
                return False
            return True
        else:
            # Enter reverse only for a goal close behind; farther away, turn around driving forward
            if abs_angle > REAR_ZONE_ENTER_RAD and distance_to_goal < self.reverse_max_distance:
                self.currently_reversing = True
                self.get_logger().info(
                    f"🔄 Enter REVERSE: angle={math.degrees(vehicle_angle_rad):+.1f}° > 130°, "
                    f"goal {distance_to_goal:.2f}m < {self.reverse_max_distance:.2f}m"
                )
                return True
            return False

    def reverse_pure_pursuit_control(self, dx, dy, distance, yaw, xb, yb):
        """
        Reverse Pure Pursuit controller.
        Returns: (curvature, success)
        """
        # Check if reachable by reversing (flip xb for deadzone check)
        xb_flip = -xb
        reachable, theta, s, classification = self.one_arc_reachable(xb_flip, yb)
        
        if not reachable:
            self.get_logger().warning(
                f"⚠️  Rear goal NOT reachable: {classification} "
                f"(xb={xb:.2f}, yb={yb:.2f})",
                throttle_duration_sec=2.0
            )
            return None, False
        
        # Calculate heading error
        heading_error = normalize_angle(math.atan2(dy, dx) - yaw)
        
        # Calculate curvature (FLIP for reverse!)
        curvature_forward = 2.0 * math.sin(heading_error) / max(distance, 0.25)
        # Cung tron nhu khi tien: duoi xe di ve phia tam cung, KHONG lat dau.
        curvature_reverse = curvature_forward
        
        kappa = max(-self.kappa_max, min(self.kappa_max, curvature_reverse))
        steering_angle = math.atan(self.wheelbase_m * kappa)
        
        self.get_logger().info(
            f"🔙 Reverse PP: v_ang={math.degrees(math.atan2(yb, xb)):+.1f}°, "
            f"d={distance:.2f}m, steer={math.degrees(steering_angle):+.1f}°",
            throttle_duration_sec=0.5
        )
        
        return kappa, True

    def curvature_to_servo(self, kappa):
        """Servo pulse for a path curvature (+ = left = servo_left_us), same mapping as pure pursuit."""
        steering_angle = math.atan(self.wheelbase_m * kappa)
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
        angle_fraction = steering_angle / self.max_steering_angle
        if angle_fraction >= 0.0:
            return round(self.servo_center_us - angle_fraction * (self.servo_center_us - self.servo_left_us))
        return round(self.servo_center_us - angle_fraction * (self.servo_right_us - self.servo_center_us))

    def base_to_body(self, x, y):
        """base_link point -> rear-axle frame of scan_points (yaw trim + rear axle offset)."""
        c = math.cos(self.lidar_yaw_trim)
        s = math.sin(self.lidar_yaw_trim)
        return c * x - s * y - self.rear_axle_x, s * x + c * y

    def arc_free_length(self, kappa, points, limit, reverse=False):
        """
        How far the rear axle can travel along an arc of curvature kappa (forward, or
        backward when reverse=True) before the inflated body touches one of `points`:
        avoid_front ahead of the rear axle, avoid_rear behind it, body_half_width to
        each side. Capped at `limit`. Covers the ends swinging out on a tight turn
        (the nose when reversing). A point in the margin right ahead of the bumper (in
        the direction of travel) blocks it: free length 0. Other points already inside
        the inflated body, like a wall right beside it, do not count for the part of
        the body they are in.
        """
        w = self.body_half_width
        front, rear = self.avoid_front, self.avoid_rear
        # Already touching the margin ahead: ignoring it would let the car drive on into it
        real_edge, edge = (self.car_rear, rear) if reverse else (self.car_front, front)
        for x, y in points:
            if real_edge < (-x if reverse else x) <= edge and -w <= y <= w:
                return 0.0
        best = limit
        if abs(kappa) < 1e-3:
            for x, y in points:
                if -w <= y <= w:
                    gap = -rear - x if reverse else x - front
                    if 0.0 <= gap < best:
                        best = gap
            return best

        # Mirror right turns so the turn center is always at (0, +radius). Angles are
        # measured around it from the rear axle, positive in the forward direction.
        radius = 1.0 / abs(kappa)
        side = 1.0 if kappa > 0.0 else -1.0
        r_in2 = (radius - w) ** 2
        r_out2 = (radius + w) ** 2
        r_max2 = max(front, rear) ** 2 + r_out2  # farthest body corner
        for x, y in points:
            yc = radius - side * y
            d2 = x * x + yc * yc
            if d2 < r_in2 or d2 > r_max2:
                continue  # the body never reaches this radius
            phi = math.atan2(x, yc)
            # At this radius the body covers |x| in [a, b], clipped to [-rear, front]:
            # one piece, or a nose piece and a tail piece outside the side line
            a = math.sqrt(max(0.0, d2 - r_out2))
            b = math.sqrt(d2 - r_in2)
            pieces = ((-b, b),) if a == 0.0 else ((a, b), (-b, -a))
            for lo, hi in pieces:
                lo, hi = max(lo, -rear), min(hi, front)
                if lo > hi:
                    continue
                th_lo = math.atan2(lo, math.sqrt(d2 - lo * lo))
                th_hi = math.atan2(hi, math.sqrt(d2 - hi * hi))
                if reverse:
                    turn = th_lo - phi
                    inside = phi <= th_hi
                else:
                    turn = phi - th_hi
                    inside = phi >= th_lo
                if turn < 0.0:
                    if inside:
                        continue  # the point is inside this piece already
                    turn += 2.0 * math.pi
                if turn * radius < best:
                    best = turn * radius
        return best

    def plan_curvature(self, kappa_pp, distance, points, reverse=False):
        """
        Arc to drive (forward, or backward when reverse=True): kappa_pp if its arc is
        free of `points`, otherwise the clear candidate arc closest to it on the
        current avoid side. None if every arc is blocked.
        """
        if reverse != self.avoid_reverse:
            self.avoid_side = 0  # the avoid side belongs to one driving direction
            self.avoid_reverse = reverse
        look = self.obstacle_distance
        # Obstacles past the goal do not block the way to it
        need = min(look, max(0.0, distance - self.arrival_tolerance_m))
        free_pp = self.arc_free_length(kappa_pp, points, look, reverse)
        now = time.monotonic()

        if free_pp >= need:
            if self.avoid_side != 0 and now - self.avoid_last_blocked > self.avoid_hold_s:
                self.avoid_side = 0
                self.get_logger().info("✅ Path clear, back to pure pursuit")
            return kappa_pp

        self.avoid_last_blocked = now
        frees = [(k, self.arc_free_length(k, points, look, reverse)) for k in self.avoid_arcs]
        left = [k for k, free in frees if free >= look and k > kappa_pp]
        right = [k for k, free in frees if free >= look and k < kappa_pp]
        best_left = min(left) if left else None     # clear arc closest to kappa_pp on its left
        best_right = max(right) if right else None  # ... and on its right

        side = self.avoid_side
        if side == 0 and best_left is not None and best_right is not None:
            dev_left = best_left - kappa_pp
            dev_right = kappa_pp - best_right
            if abs(dev_left - dev_right) < 0.5 * self.avoid_arc_step:
                # Same detour either way: take the side with more room on average
                room_left = [free for k, free in frees if k > kappa_pp]
                room_right = [free for k, free in frees if k < kappa_pp]
                side = 1 if sum(room_left) / len(room_left) >= sum(room_right) / len(room_right) else -1
            else:
                side = 1 if dev_left < dev_right else -1
        elif side >= 0 and best_left is None and best_right is not None:
            side = -1
        elif side <= 0 and best_right is None and best_left is not None:
            side = 1

        if side > 0 and best_left is not None:
            kappa = best_left
        elif side < 0 and best_right is not None:
            kappa = best_right
        else:
            # No arc is clear for the whole look-ahead: creep along the most open one
            kappa, free = max(frees, key=lambda item: (item[1], -abs(item[0] - kappa_pp)))
            if free < self.avoid_min_free:
                return None
            side = 1 if kappa > kappa_pp else -1

        if side != self.avoid_side:
            self.get_logger().warning(
                f"🚧 Path {'behind' if reverse else 'ahead'} blocked ({free_pp:.2f}m free < {need:.2f}m needed): "
                f"going around on the {'LEFT' if side > 0 else 'RIGHT'}"
            )
            self.avoid_side = side
        return kappa

    def goal_blocked(self, xb, yb):
        """True if LiDAR points sit within goal_clear_radius of the goal: the car cannot stand on it."""
        gx, gy = self.base_to_body(xb, yb)
        r2 = self.goal_clear_radius ** 2
        hits = 0
        for x, y in self.scan_points:
            if (x - gx) ** 2 + (y - gy) ** 2 < r2:
                hits += 1
                if hits >= 3:  # several points, so one noisy return does not count
                    return True
        return False

    def points_away_from_goal(self, xb, yb):
        """scan_points without the ones around the goal (the obstacle the goal sits against)."""
        gx, gy = self.base_to_body(xb, yb)
        r2 = self.goal_clear_radius ** 2
        return [(x, y) for x, y in self.scan_points if (x - gx) ** 2 + (y - gy) ** 2 >= r2]

    def closest_approach_arc(self, xb, yb, distance, reverse):
        """
        Goal against an obstacle: the free arc along which base_link gets closest to the
        goal (at least 1 cm closer than now), or None when no arc helps any more.
        """
        gx, gy = self.base_to_body(xb, yb)   # goal in the rear-axle frame
        lead = -self.rear_axle_x             # base_link sits this far ahead of the rear axle
        best_kappa, best_d = None, distance - 0.01
        for kappa in self.avoid_arcs:
            free = self.arc_free_length(kappa, self.scan_points, self.obstacle_distance, reverse) - GOAL_STOP_FREE_M
            travelled = 0.0
            while travelled < free:
                travelled = min(travelled + 0.02, free)
                s = -travelled if reverse else travelled
                th = s * kappa
                if abs(kappa) < 1e-6:
                    x, y = s, 0.0
                else:
                    x, y = math.sin(th) / kappa, (1.0 - math.cos(th)) / kappa
                d = math.hypot(gx - x - lead * math.cos(th), gy - y - lead * math.sin(th))
                if d < best_d:
                    best_kappa, best_d = kappa, d
        return best_kappa

    def start_avoid_backup(self, position, kappa_pp, reverse=False, nose_left=None):
        """
        Every arc in the driving direction is blocked: move a little the other way
        (back up, or pull forward if the car was reversing) with opposite lock, so the
        blocked end swings toward open space, like the Y-turn backup. nose_left forces
        the side (turn-around K-turn) instead of picking the roomier one.
        """
        if nose_left is None:
            frees = [(k, self.arc_free_length(k, self.scan_points, self.obstacle_distance, reverse))
                     for k in self.avoid_arcs]
            best_kappa = max(frees, key=lambda item: (item[1], -abs(item[0] - kappa_pp)))[0]
            turn_left = best_kappa > 0.0 or (best_kappa == 0.0 and kappa_pp >= 0.0)
        else:
            turn_left = nose_left

        move_reverse = not reverse
        move_kappa = -self.kappa_max if turn_left else self.kappa_max
        room = self.arc_free_length(move_kappa, self.scan_points, self.avoid_backup_m, move_reverse)
        if room < 0.05:
            move_kappa = 0.0
            room = self.arc_free_length(move_kappa, self.scan_points, self.avoid_backup_m, move_reverse)
        # Blocking check disabled - always attempt maneuver even if room < 0.05
        # Original logic: if room < 0.05: stop and return
        if room < 0.05:
            self.get_logger().warning(f"⚠️ Limited room ({room:.2f}m), attempting maneuver anyway")

        self.maneuver_mode = ManeuverMode.AVOID_BACKUP
        self.avoid_backup_kappa = move_kappa
        self.avoid_backup_reverse = move_reverse
        self.avoid_backup_distance = room
        self.avoid_backup_start = (position.x, position.y)
        self.avoid_backup_started_at = time.monotonic()
        self.avoid_backup_count += 1
        self.avoid_side = 1 if turn_left else -1
        self.avoid_reverse = reverse
        self.avoid_last_blocked = time.monotonic()
        if nose_left is not None:
            self.get_logger().info(
                f"🔄 Turning around: backing up {room:.2f}m (#{self.avoid_backup_count}) "
                f"to swing the nose {'LEFT' if turn_left else 'RIGHT'}"
            )
        elif reverse:
            self.get_logger().warning(
                f"⛔ No free arc behind: pulling forward {room:.2f}m (#{self.avoid_backup_count}), "
                f"tail toward {'LEFT' if turn_left else 'RIGHT'}"
            )
        else:
            self.get_logger().warning(
                f"⛔ No free arc ahead: backing up {room:.2f}m (#{self.avoid_backup_count}), "
                f"nose toward {'LEFT' if turn_left else 'RIGHT'}"
            )

    def run_avoid_backup(self, position):
        """AVOID_BACKUP: move until the planned distance, an obstacle in the way, or timeout."""
        moved = math.hypot(position.x - self.avoid_backup_start[0], position.y - self.avoid_backup_start[1])
        room = self.arc_free_length(
            self.avoid_backup_kappa, self.scan_points, self.avoid_backup_m, self.avoid_backup_reverse
        )
        timed_out = time.monotonic() - self.avoid_backup_started_at > AVOID_BACKUP_TIMEOUT_S

        if moved >= self.avoid_backup_distance or room < 0.03 or timed_out:
            self.get_logger().info(
                f"↩️  Avoid {'backup' if self.avoid_backup_reverse else 'pull-forward'} done "
                f"({moved:.2f}m), re-planning"
            )
            self.maneuver_mode = ManeuverMode.NORMAL
            self.need_plan = True  # the obstacle may be new to the map: route around it
            self.send_command("S", force=True)
            return
        servo_us = self.curvature_to_servo(self.avoid_backup_kappa)
        command = "B" if self.avoid_backup_reverse else "V"
        self.send_command(f"{command},{servo_us},{self.yturn_backup_pwm}")

    def yturn_allowed(self, distance):
        """Y-turn enabled, goal far enough, and cooldown elapsed."""
        if not self.enable_yturn:
            return False
        if distance <= self.yturn_min_distance:
            return False
        return time.time() - self.last_yturn_time > self.yturn_cooldown

    def calculate_adaptive_backup_distance(self, heading_error, distance_to_goal):
        """Calculate optimal backup distance based on heading error."""
        target_angle = abs(heading_error)
        forward_distance = self.yturn_speed_estimate * self.yturn_turn_duration
        theta_forward = (forward_distance * math.tan(self.max_steering_angle)) / self.wheelbase_m
        theta_backup_needed = target_angle - theta_forward
        
        if theta_backup_needed > 0:
            backup_distance = (theta_backup_needed * self.wheelbase_m) / math.tan(self.max_steering_angle)
        else:
            backup_distance = self.yturn_backup_distance_min
        
        backup_distance = max(self.yturn_backup_distance_min, 
                             min(self.yturn_backup_distance_max, backup_distance))
        
        if distance_to_goal < 1.0:
            scale_factor = 0.7
        elif distance_to_goal > 2.0:
            scale_factor = 1.0
        else:
            scale_factor = 0.7 + (distance_to_goal - 1.0) * 0.3
        
        backup_distance *= scale_factor
        backup_distance = max(self.yturn_backup_distance_min, 
                             min(self.yturn_backup_distance_max, backup_distance))
        
        return round(backup_distance, 2)

    def start_yturn(self, position, yaw, yb, heading_error, distance):
        """Start Y-turn maneuver."""
        if self.yturn_consecutive >= self.yturn_max_per_goal:
            self.get_logger().error(
                f"Y-turn limit reached ({self.yturn_max_per_goal}) for this goal, stopping."
            )
            self.stop()
            return
        self.maneuver_mode = ManeuverMode.YTURN_BACKUP
        self.yturn_start_pos = (position.x, position.y)
        self.yturn_start_time = time.time()
        self.yturn_direction = 1 if yb > 0 else -1
        
        if self.enable_adaptive_backup:
            self.yturn_backup_distance_dynamic = self.calculate_adaptive_backup_distance(heading_error, distance)
            backup_method = "adaptive"
        else:
            self.yturn_backup_distance_dynamic = self.yturn_backup_distance_min
            backup_method = "fixed"
        
        self.get_logger().info(
            f"🔄 Y-turn #{self.yturn_count + 1}: "
            f"dir={'LEFT' if self.yturn_direction > 0 else 'RIGHT'}, "
            f"backup={self.yturn_backup_distance_dynamic:.2f}m ({backup_method}), "
            f"h_err={math.degrees(heading_error):.0f}°"
        )

    def execute_yturn(self, position, yaw):
        """Execute Y-turn state machine."""
        if self.maneuver_mode == ManeuverMode.YTURN_BACKUP:
            if self.yturn_start_pos is None:
                self.maneuver_mode = ManeuverMode.NORMAL
                return
            
            dx = position.x - self.yturn_start_pos[0]
            dy = position.y - self.yturn_start_pos[1]
            backup_distance = math.hypot(dx, dy)
            
            backup_kappa = -self.kappa_max if self.yturn_direction > 0 else self.kappa_max
            rear_blocked = self.enable_obstacle_avoid and self.arc_free_length(
                backup_kappa, self.scan_points, 0.05, reverse=True
            ) < 0.05
            if rear_blocked:
                self.get_logger().warning("🚧 Y-turn: obstacle behind, ending the backup early")

            if backup_distance >= self.yturn_backup_distance_dynamic or rear_blocked:
                self.get_logger().info("🔄 Y-turn: backup done, turning forward")
                self.maneuver_mode = ManeuverMode.YTURN_TURN
                self.yturn_start_time = time.time()
                self.send_command("S", force=True)
                # no sleep: the next 0.1 s control tick already gives the stop gap
            else:
                servo_us = self.servo_right_us if self.yturn_direction > 0 else self.servo_left_us
                self.send_command(f"B,{servo_us},{self.yturn_backup_pwm}")
        
        elif self.maneuver_mode == ManeuverMode.YTURN_TURN:
            elapsed = time.time() - self.yturn_start_time
            turn_kappa = self.kappa_max if self.yturn_direction > 0 else -self.kappa_max
            turn_blocked = self.enable_obstacle_avoid and self.arc_free_length(
                turn_kappa, self.scan_points, self.avoid_min_free
            ) < self.avoid_min_free
            if turn_blocked:
                self.get_logger().warning("🚧 Y-turn: obstacle on the turn arc, ending the turn early")
            
            if elapsed >= self.yturn_turn_duration or turn_blocked:
                self.get_logger().info("🔄 Y-turn complete, resuming")
                self.last_yturn_time = time.time()
                self.yturn_count += 1
                self.yturn_consecutive += 1
                self.maneuver_mode = ManeuverMode.NORMAL
                self.send_command("S", force=True)
                # no sleep: the next 0.1 s control tick already gives the stop gap
            else:
                servo_us = self.servo_left_us if self.yturn_direction > 0 else self.servo_right_us
                self.send_command(f"V,{servo_us},{self.yturn_turn_pwm}")

    def follow_path(self, position, yaw):
        """
        Point pp_lookahead_m ahead along the A* path, or None to steer straight at the
        goal (no map, no path, or already on the last stretch).
        """
        if not self.enable_global_planner:
            return None
        px, py = position.x, position.y
        replan = self.need_plan or time.monotonic() - self.last_plan_time > self.replan_period
        if self.path and not replan and self.closest_on_path(px, py)[1] > REPLAN_DEVIATION_M:
            self.get_logger().info("🗺️  Off the path, replanning")
            replan = True
        if replan:
            self.plan_global_path(px, py, yaw)
        if not self.path or len(self.path) < 2:
            return None

        # Walk pp_lookahead_m along the path from the closest point
        k, _, f = self.closest_on_path(px, py)
        self.path_index = k
        (ax, ay), (bx, by) = self.path[k], self.path[k + 1]
        x, y = ax + f * (bx - ax), ay + f * (by - ay)
        left = math.hypot(bx - x, by - y)
        remaining = self.pp_lookahead_m
        while remaining > left:
            remaining -= left
            k += 1
            if k >= len(self.path) - 1:
                return None  # last stretch: the goal logic takes over
            x, y = self.path[k]
            bx, by = self.path[k + 1]
            left = math.hypot(bx - x, by - y)
        f = remaining / left if left > 1e-9 else 0.0
        return x + f * (bx - x), y + f * (by - y)

    def closest_on_path(self, px, py):
        """(segment, distance, fraction along it) of the path point closest to (px, py)."""
        best = (self.path_index, float("inf"), 0.0)
        for k in range(max(0, self.path_index - 2), min(len(self.path) - 1, self.path_index + 10)):
            (ax, ay), (bx, by) = self.path[k], self.path[k + 1]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            f = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            d = math.hypot(ax + f * dx - px, ay + f * dy - py)
            if d < best[1]:
                best = (k, d, f)
        return best

    def plan_global_path(self, px, py, yaw):
        self.need_plan = False
        self.last_plan_time = time.monotonic()
        if self.planner.grid is None:
            self.get_logger().warning("🗺️  No /map yet: driving straight at the goal", throttle_duration_sec=5.0)
            return
        started = time.monotonic()
        path = self.planner.plan((px, py), self.goal, self.scan_points_in_map(px, py, yaw))
        ms = (time.monotonic() - started) * 1000.0
        if path is None:
            if self.path:
                self.get_logger().warning(f"🗺️  Replan failed ({ms:.0f} ms), keeping the old path")
            else:
                self.get_logger().warning(f"🗺️  No A* path to the goal ({ms:.0f} ms): driving straight at it")
            return
        self.path = path
        self.path_index = 0
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        self.get_logger().info(f"🗺️  A* path: {len(path)} waypoints, {length:.2f} m ({ms:.0f} ms)")
        self.publish_path()

    def scan_points_in_map(self, px, py, yaw):
        """Current LiDAR points in the map frame, so obstacles the map lacks count for A*."""
        c_t, s_t = math.cos(self.lidar_yaw_trim), math.sin(self.lidar_yaw_trim)
        c_y, s_y = math.cos(yaw), math.sin(yaw)
        out = []
        for x, y in self.scan_points:
            x += self.rear_axle_x                                 # rear-axle frame -> trimmed base_link
            xb, yb = c_t * x + s_t * y, -s_t * x + c_t * y        # undo the yaw trim
            out.append((px + c_y * xb - s_y * yb, py + s_y * xb + c_y * yb))
        return out

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path or []:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def turn_around(self, heading_error, distance, position):
        """
        Goal behind the side, too far to reverse to: turn toward it driving forward at
        full lock. When that arc is blocked, back up with opposite lock (a K-turn) so
        the nose keeps swinging the same way, instead of driving away from the goal.
        """
        if self.turnaround_side == 0:
            side = 1 if heading_error > 0.0 else -1
            if abs(heading_error) > TURN_SIDE_BY_ROOM_RAD:
                # Nearly straight behind: either way is as short, take the roomier one
                free_left = self.arc_free_length(self.kappa_max, self.scan_points, self.obstacle_distance)
                free_right = self.arc_free_length(-self.kappa_max, self.scan_points, self.obstacle_distance)
                if free_left != free_right:
                    side = 1 if free_left > free_right else -1
            self.turnaround_side = side
            self.get_logger().info(
                f"🔄 Target behind ({math.degrees(heading_error):+.0f}°, {distance:.2f}m): "
                f"turning around to the {'LEFT' if side > 0 else 'RIGHT'}"
            )

        kappa = self.turnaround_side * self.kappa_max
        if self.arc_free_length(kappa, self.scan_points, self.avoid_min_free) >= self.avoid_min_free:
            self.send_command(f"V,{self.curvature_to_servo(kappa)},{self.slow_pwm}")
            return
        self.start_avoid_backup(position, kappa, nose_left=self.turnaround_side > 0)

    def drive_with_avoidance(self, kappa_pp, distance, xb, yb, position, reverse):
        """Drive the pure pursuit arc kappa_pp (forward or in reverse), going around obstacles."""
        kappa = kappa_pp
        if self.enable_obstacle_avoid:
            points = self.scan_points
            goal_occupied = self.goal_blocked(xb, yb)
            if goal_occupied:
                # Goal sits against an obstacle: do not dodge that one (others still count),
                # just get as close to the goal as it allows (below)
                points = self.points_away_from_goal(xb, yb)
            kappa = self.plan_curvature(kappa_pp, distance, points, reverse)
            if kappa is None:
                self.start_avoid_backup(position, kappa_pp, reverse)
                return
            if goal_occupied and self.arc_free_length(
                kappa, self.scan_points, self.obstacle_distance, reverse
            ) < GOAL_STOP_FREE_M:
                # That arc ends at the obstacle: take whichever free arc gets closest to the goal
                kappa = self.closest_approach_arc(xb, yb, distance, reverse)
            if kappa is None:
                self.get_logger().info(
                    f"🎯 Goal is right next to an obstacle: stopped as close as possible, "
                    f"{distance:.2f}m from it"
                )
                self.stop()
                return

        servo_us = self.curvature_to_servo(kappa)
        if kappa != kappa_pp:
            self.get_logger().info(
                f"↪️  Avoiding {'LEFT' if self.avoid_side > 0 else 'RIGHT'}{' (reversing)' if reverse else ''}: "
                f"κ={kappa:+.2f} (pure pursuit {kappa_pp:+.2f}), servo={servo_us}",
                throttle_duration_sec=0.5,
            )
        self.send_command(f"{'B' if reverse else 'V'},{servo_us},{self.slow_pwm}")

    def control_goal(self):
        if self.serial_error or self.goal is None:
            self.send_command("S")
            return

        with self.encoder_lock:
            last_encoder_time = self.last_encoder_time
        
        if last_encoder_time is None or time.monotonic() - last_encoder_time > self.encoder_timeout_s:
            self.get_logger().error("Encoder timeout", throttle_duration_sec=2.0)
            self.stop()
            return

        if self.enable_obstacle_avoid and (
            self.last_scan_time is None
            or time.monotonic() - self.last_scan_time > self.scan_timeout
        ):
            self.get_logger().error("No fresh /scan, stopping", throttle_duration_sec=2.0)
            self.stop()
            return

        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except TransformException as error:
            self.get_logger().warning(f"TF unavailable: {error}", throttle_duration_sec=2.0)
            self.stop()
            return

        position = transform.transform.translation
        yaw = quaternion_yaw(transform.transform.rotation)
        dx = self.goal[0] - position.x
        dy = self.goal[1] - position.y
        distance = math.hypot(dx, dy)
        
        if distance <= self.arrival_tolerance_m:
            self.get_logger().info("🎯 Goal reached!")
            self.stop()
            return

        # Transform to base_link
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        xb = cos_yaw * dx + sin_yaw * dy
        yb = -sin_yaw * dx + cos_yaw * dy
        
        heading_error = normalize_angle(math.atan2(dy, dx) - yaw)
        vehicle_angle_rad = math.atan2(yb, xb)
        
        # Avoid backup keeps priority until it finishes
        if self.maneuver_mode == ManeuverMode.AVOID_BACKUP:
            self.run_avoid_backup(position)
            return

        # Y-turn has priority
        if self.maneuver_mode != ManeuverMode.NORMAL:
            self.execute_yturn(position, yaw)
            return
        
        # ============================================================
        # REVERSE MODE CHECK (with hysteresis + distance)
        # ============================================================
        in_reverse_mode = self.should_reverse(vehicle_angle_rad, distance)
        
        if in_reverse_mode:
            # === REVERSE PURE PURSUIT ===
            kappa_rev, success = self.reverse_pure_pursuit_control(dx, dy, distance, yaw, xb, yb)

            if success:
                self.drive_with_avoidance(kappa_rev, distance, xb, yb, position, reverse=True)
                return

            # Goal not reachable by one reverse arc
            if self.yturn_allowed(distance):
                self.get_logger().info("🔄 Y-turn from reverse (not reachable)")
                self.start_yturn(position, yaw, yb, heading_error, distance)
                return

            if xb < 0.0:
                # Goal is behind the car: simple reverse toward it
                self.get_logger().warning("Deadzone, using simple reverse", throttle_duration_sec=2.0)
                curvature = 2.0 * math.sin(heading_error) / max(distance, 0.25)
                kappa_rev = max(-self.kappa_max, min(self.kappa_max, curvature))
                self.drive_with_avoidance(kappa_rev, distance, xb, yb, position, reverse=True)
                return

            # Goal is in FRONT (hysteresis band 45-90 deg): never reverse away from it
            if self.enable_yturn and distance > self.yturn_min_distance:
                # Y-turn is in cooldown: stand still until it is allowed
                self.send_command("S")
                return

            # Very close to a goal in front (or Y-turn disabled): leave reverse,
            # finish with forward pure pursuit below
            self.currently_reversing = False
        
        # ============================================================
        # GLOBAL PATH - steer at a point on the A* path instead of the goal
        # ============================================================
        steer_heading, steer_distance = heading_error, distance
        target = self.follow_path(position, yaw)
        if target is not None:
            steer_heading = normalize_angle(math.atan2(target[1] - position.y, target[0] - position.x) - yaw)
            steer_distance = math.hypot(target[0] - position.x, target[1] - position.y)

        # ============================================================
        # FORWARD MODE - Y-turn only when the goal is inside the min turning
        # circle, where forward driving can never reach it (not while going
        # around an obstacle). Larger angles: full lock below.
        # ============================================================
        if self.enable_yturn and self.avoid_side == 0 and target is None:
            reachable, theta, arc_length, classification = self.one_arc_reachable(xb, yb)

            if classification == 'DEAD_ZONE' and self.yturn_allowed(distance):
                self.get_logger().info(f"🔄 Y-turn from forward: {classification}")
                self.start_yturn(position, yaw, yb, heading_error, distance)
                return
        
        # ============================================================
        # FORWARD MODE - goal behind the side: turn around (K-turn if needed)
        # ============================================================
        if self.enable_obstacle_avoid and abs(steer_heading) > math.pi / 2:
            self.turn_around(steer_heading, steer_distance, position)
            return
        if self.turnaround_side != 0:
            self.turnaround_side = 0
            self.get_logger().info("✅ Turned around, heading to the goal")

        # ============================================================
        # FORWARD PURE PURSUIT + ARC-BASED OBSTACLE AVOIDANCE
        # ============================================================
        # Aim at most pp_lookahead_m along the line to the goal: turn toward it, then go straight
        lookahead = min(max(steer_distance, 0.25), self.pp_lookahead_m)
        curvature = 2.0 * math.sin(steer_heading) / lookahead
        if abs(steer_heading) > math.pi / 2:
            # Target behind the side: the pure pursuit arc would loop far out, turn in tight instead
            curvature = math.copysign(self.kappa_max, steer_heading)
        kappa_pp = max(-self.kappa_max, min(self.kappa_max, curvature))
        self.drive_with_avoidance(kappa_pp, distance, xb, yb, position, reverse=False)

    def destroy_node(self):
        self.send_command("S", force=True)
        if self.uart.is_open:
            self.uart.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = GoalPoseDrive()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()