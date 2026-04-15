import os
import time
import json
import placo
import numpy as np

class MotionEngine:
    _dt = 0.01
    _refine = 10

    _time_since_last_right_contact = 0.0
    _time_since_last_left_contact = 0.0

    _initial_delay = -1.0
    _t = -1.0
    _last_replan = 0

    _start_time = None

    _is_ignore_feet_contacts = False

    def __init__(self, robot_folder_path: str = "", gait_parameters: dict = {}) -> None:
        # Load limits
        self._limits_file_path = os.path.join(robot_folder_path, "limits.json")

        with open(self._limits_file_path, "r") as f:
            self._limits = json.load(f)

        # Load robot and parameters
        self._urdf_file_path = os.path.join(robot_folder_path, "bdx.urdf")

        self.robot = placo.HumanoidRobot(self._urdf_file_path)

        self.robot.set_velocity_limits(12.0)

        self.robot.set_joint_limits("left_knee", self._limits["left_knee_lower"], self._limits["left_knee_upper"])
        self.robot.set_joint_limits("right_knee", self._limits["right_knee_lower"], self._limits["right_knee_upper"])

        self._robot_parameters = placo.HumanoidParameters()
        self._load_parameters(gait_parameters)

        # Load collision pairs
        #self._collisions_file_path = os.path.join(robot_folder_path, "collisions.json")
        #self.robot.load_collision_pairs(self._collisions_file_path)

        # Create kinematics solver
        self._solver = placo.KinematicsSolver(self.robot)

        self._solver.enable_velocity_limits(True)

        self._solver.enable_joint_limits(False)

        self._solver.dt = self._dt / self._refine

        # Create walk QP task
        self._task = placo.WalkTasks()

        self._task.trunk_mode = self._robot_parameters.trunk_mode

        self._task.com_x = 0.0

        self._task.initialize_tasks(self._solver, self.robot)

        self._task.left_foot_task.orientation().mask.set_axises("yz", "local")
        self._task.right_foot_task.orientation().mask.set_axises("yz", "local")

        # Create joints task
        self._joints = self._robot_parameters.joints

        joint_angles = self._robot_parameters.joint_angles
        masked_joint_angles = {joint: np.deg2rad(degree) for joint, degree in joint_angles.items()}

        self._joints_task = self._solver.add_joints_task()

        self._joints_task.set_joints(masked_joint_angles)

        self._joints_task.configure("joints", "soft", 1.0)

        # Place robot in initial pose
        self._task.reach_initial_pose(
            np.eye(4),
            self._robot_parameters.feet_spacing,
            self._robot_parameters.walk_com_height,
            self._robot_parameters.walk_trunk_pitch,
        )

        # Create Footsteps planner
        self._d_x = 0.0
        self._d_y = 0.0
        self._d_theta = 0.0
        self._number_of_steps = 5

        self._footsteps_planner = placo.FootstepsPlannerRepetitive(self._robot_parameters)
        
        self._footsteps_planner.configure(self._d_x, self._d_y, self._d_theta, self._number_of_steps)

        # Plan footsteps
        self._T_world_left = placo.flatten_on_floor(self.robot.get_T_world_left())
        self._T_world_right = placo.flatten_on_floor(self.robot.get_T_world_right())

        self._footsteps = self._footsteps_planner.plan(
            placo.HumanoidRobot_Side.left, self._T_world_left, self._T_world_right
        )

        self._supports = placo.FootstepsPlanner.make_supports(
            self._footsteps, 0.0, True, self._robot_parameters.has_double_support(), True
        )

        # Create pattern generator
        self._pattern_generator = placo.WalkPatternGenerator(self.robot, self._robot_parameters)

        # Nominal CoM for LIPM: same horizontal placement as reach_initial_pose (mid-feet at walk height).
        # With trunk_mode, IK tracks trunk position, not true CoM — robot.com_world() can sit outside the
        # support polygon and the LIPM QP then fails with "Failed to plan CoM trajectory".
        Tl = np.asarray(self._T_world_left, dtype=float)
        Tr = np.asarray(self._T_world_right, dtype=float)
        p_com_init = np.array(
            [
                0.5 * (Tl[0, 3] + Tr[0, 3]),
                0.5 * (Tl[1, 3] + Tr[1, 3]),
                self._robot_parameters.walk_com_height,
            ]
        )
        self._trajectory = self._pattern_generator.plan(self._supports, p_com_init, 0.0)

        # Period
        self._period = 2 * self._robot_parameters.single_support_duration + 2 * self._robot_parameters.double_support_duration()

    
    @property
    def robot_parameters(self):
        return self._robot_parameters

    @property
    def t(self) -> float:
        return self._t

    @property
    def period(self):
        return self._period


    def _load_parameters(self, gait_parameters: dict) -> None:
        self._robot_parameters.trunk_mode = gait_parameters.get("trunk_mode", False)
        self._robot_parameters.double_support_ratio = gait_parameters.get("double_support_ratio", self._robot_parameters.double_support_ratio)
        self._robot_parameters.startend_double_support_ratio = gait_parameters.get("startend_double_support_ratio", self._robot_parameters.startend_double_support_ratio)
        self._robot_parameters.planned_timesteps = gait_parameters.get("planned_timesteps", self._robot_parameters.planned_timesteps)
        
        #self._robot_parameters.replan_timesteps = gait_parameters.get("replan_timesteps", self._robot_parameters.replan_timesteps)
        self._replan_timesteps = gait_parameters.get("replan_timesteps", 10)
        
        self._robot_parameters.walk_com_height = gait_parameters.get("walk_com_height", self._robot_parameters.walk_com_height)
        self._robot_parameters.walk_foot_height = gait_parameters.get("walk_foot_height", self._robot_parameters.walk_foot_height)
        self._robot_parameters.walk_trunk_pitch = gait_parameters.get("walk_trunk_pitch", self._robot_parameters.walk_trunk_pitch)
        self._robot_parameters.walk_foot_rise_ratio = gait_parameters.get("walk_foot_rise_ratio", self._robot_parameters.walk_foot_rise_ratio)
        self._robot_parameters.single_support_duration = gait_parameters.get("single_support_duration", self._robot_parameters.single_support_duration)
        self._robot_parameters.single_support_timesteps = gait_parameters.get("single_support_timesteps", self._robot_parameters.single_support_timesteps)
        self._robot_parameters.foot_length = gait_parameters.get("foot_length", self._robot_parameters.foot_length)
        self._robot_parameters.feet_spacing = gait_parameters.get("feet_spacing", self._robot_parameters.feet_spacing)
        self._robot_parameters.zmp_margin = gait_parameters.get("zmp_margin", self._robot_parameters.zmp_margin)
        self._robot_parameters.foot_zmp_target_x = gait_parameters.get("foot_zmp_target_x", self._robot_parameters.foot_zmp_target_x)
        self._robot_parameters.foot_zmp_target_y = gait_parameters.get("foot_zmp_target_y", self._robot_parameters.foot_zmp_target_y)
        self._robot_parameters.walk_max_dtheta = gait_parameters.get("walk_max_dtheta", self._robot_parameters.walk_max_dtheta)
        self._robot_parameters.walk_max_dy = gait_parameters.get("walk_max_dy", self._robot_parameters.walk_max_dy)
        self._robot_parameters.walk_max_dx_forward = gait_parameters.get("walk_max_dx_forward", self._robot_parameters.walk_max_dx_forward)
        self._robot_parameters.walk_max_dx_backward = gait_parameters.get("walk_max_dx_backward", self._robot_parameters.walk_max_dx_backward)
        self._robot_parameters.joints = gait_parameters.get("joints", [])
        self._robot_parameters.joint_angles = gait_parameters.get("joint_angles", {})


    def get_angles(self) -> dict:
        return {joint: self.robot.get_joint(joint) for joint in self._joints}

    def get_supports(self):
        return self._trajectory.get_supports()

    def get_current_support_phase(self):
        if self._trajectory.support_is_both(self._t):
            return [1, 1]
        elif str(self._trajectory.support_side(self._t)) == "left":
            return [1, 0]
        elif str(self._trajectory.support_side(self._t)) == "right":
            return [0, 1]
        else:
            raise ValueError(f"Invalid support phase at time {self._t}")

    def set_trajectory(self, dx: float, dy: float, dtheta: float) -> None:
        self._d_x = dx
        self._d_y = dy
        self._d_theta = dtheta

        self._footsteps_planner.configure(self._d_x, self._d_y, self._d_theta, self._number_of_steps)

    def tick(self, dt: float) -> None:
        if self._start_time is None:
            self._start_time = time.time()

        if not self._is_ignore_feet_contacts:
            self._time_since_last_left_contact = 0.0
            self._time_since_last_right_contact = 0.0

        falling = not self._is_ignore_feet_contacts and (
            self._time_since_last_left_contact > self._robot_parameters.single_support_duration or
            self._time_since_last_right_contact > self._robot_parameters.single_support_duration
        )

        for k in range(self._refine):
            if not falling:
                self._task.update_tasks_from_trajectory(self._trajectory, self._t - dt + k * self._dt / self._refine)

            self.robot.update_kinematics()
            _ = self._solver.solve(True)

        if (self._t - self._last_replan > self._replan_timesteps * self._robot_parameters.dt() and 
            self._pattern_generator.can_replan_supports(self._trajectory, self._t)):
            
            self._supports = self._pattern_generator.replan_supports(self._footsteps_planner, self._trajectory, self._t, self._last_replan)

            self._trajectory = self._pattern_generator.replan(self._supports, self._trajectory, self._t)

            self._last_replan = self._t

        self._time_since_last_left_contact += dt
        self._time_since_last_right_contact += dt
        self._t += dt
