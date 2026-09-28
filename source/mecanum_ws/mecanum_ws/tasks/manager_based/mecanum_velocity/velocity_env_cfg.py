# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base-velocity tracking task for a mecanum-wheeled robot on flat ground."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from mecanum_ws.robots import mecanum as robot

from . import mdp

##
# Scene definition
##


@configclass
class MecanumSceneCfg(InteractiveSceneCfg):
    """Flat ground with a mecanum robot."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
            # carries the compliant roller contact (only effective on the ground side)
            physics_material=robot.ground_material(),
        ),
    )

    robot: ArticulationCfg = robot.MECANUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.05,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.5, 1.5)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP: the policy directly commands the four wheel velocities."""

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=robot.WHEEL_JOINT_NAMES,
        scale=robot.MAX_WHEEL_SPEED,
        use_default_offset=False,
        preserve_order=True,
        clip={".*": (-robot.MAX_WHEEL_SPEED, robot.MAX_WHEEL_SPEED)},
    )


@configclass
class IkActionsCfg:
    """Action specifications for the MDP: the policy commands a base twist, mapped to wheels by mecanum IK."""

    base_vel = mdp.MecanumBaseVelocityActionCfg(
        asset_name="robot",
        wheel_joint_names=robot.WHEEL_JOINT_NAMES,
        wheel_joint_signs=robot.WHEEL_JOINT_SIGNS,
        wheel_radius=robot.WHEEL_RADIUS,
        wheel_base_half_length=robot.WHEEL_BASE_HALF_LENGTH,
        track_half_width=robot.TRACK_HALF_WIDTH,
        scale=(1.5, 1.5, 2.0),
        max_wheel_speed=robot.MAX_WHEEL_SPEED,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=robot.WHEEL_JOINT_NAMES, preserve_order=True)},
            scale=1.0 / robot.MAX_WHEEL_SPEED,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.1),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=robot.WHEEL_JOINT_NAMES),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task: track the commanded base velocity
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # regularization
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


##
# Environment configuration
##


@configclass
class MecanumVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity tracking; the policy outputs the four wheel velocities."""

    scene: MecanumSceneCfg = MecanumSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # 360 Hz physics: needed by the roller wheels (see scripts/tools/validate_mecanum_wheels.py), 51 Hz policy
        self.decimation = 7
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 360.0
        self.sim.render_interval = self.decimation
        # default material = material of the robot colliders (the URDF spawner binds none)
        self.sim.physics_material = robot.ROBOT_MATERIAL
        # viewer settings
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0


@configclass
class MecanumVelocityFlatEnvCfg_PLAY(MecanumVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None


@configclass
class MecanumVelocityFlatIkEnvCfg(MecanumVelocityFlatEnvCfg):
    """Velocity tracking; the policy outputs a base twist that is mapped to the wheels by mecanum IK."""

    actions: IkActionsCfg = IkActionsCfg()


@configclass
class MecanumVelocityFlatIkEnvCfg_PLAY(MecanumVelocityFlatIkEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
