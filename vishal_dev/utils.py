import dexmimicgen  

import h5py 
import imageio 
import numpy as np
import time 
import os
import json
import pose_utils_vishal_from_mimicgen as PoseUtils
import robosuite.utils.transform_utils as T

from pose_utils_vishal_from_mimicgen import transform_source_data_segment_using_object_pose
from pose_utils_vishal_from_mimicgen import interpolate_poses
from pose_utils_vishal_from_mimicgen import make_pose
from pose_utils_vishal_from_mimicgen import unmake_pose

def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta

def just_reset(env, state):
    # The following resets the state
    env.reset()
    # env.sim.set_state_from_flattened(state)
    # env.sim.forward()

    # This part is just to visualize the state after resetting
    env.render()
    obfull = env.step(np.zeros(env.action_dim))
    env.render()
    return obfull

def reset_to_state(env, state):
    # The following resets the state
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    # This part is just to visualize the state after resetting
    env.render()
    obfull = env.step(np.zeros(env.action_dim))
    env.render()
    return obfull

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None

def eef_to_action(eef_pos, eef_rot_mats, original_action, prev_eef_pos, prev_eef_quat, max_dpos=0.05, max_drot=0.5, cur_rot_from_env=False):
    """
    Convert end-effector pose target actions to simulation-executable action space.

    args:
        eef_pos (np.ndarray): target end-effector position (2, 3)
        eef_rot_mats (np.ndarray): target end-effector orientation as rotation matrices (2, 3, 3)
        original_action (np.ndarray): original action to copy other dimensions from (15,)
        prev_eef_pos (np.ndarray): previous end-effector position (2, 3)
        prev_eef_quat (np.ndarray): previous end-effector orientation as quaternion (2, 4)
        max_dpos (float): maximum position delta
        max_drot (float): maximum rotation delta in axis-angle representation
        cur_rot_from_env (bool): rotation matrices from environment are slightly different from that ones from the dataset.
            If true, these rotation matrices must be adjusted before computing the delta rotation. Otherwise, the delta rotation
            will be incorrect.
    """
    max_dpos = 0.05
    max_drot = 0.5

    action = np.copy(original_action)

    # Set robot action in position space
    action[0:3] = eef_pos[0] - prev_eef_pos[0]
    action[0:3] = np.clip(action[0:3] / max_dpos, -1., 1.)
    action[12:15] = eef_pos[1] - prev_eef_pos[1]
    action[12:15] = np.clip(action[12:15] / max_dpos, -1., 1.)

    # Set robot action in rotation space
    curr_rot = T.quat2mat(prev_eef_quat[0])
    target_rot = eef_rot_mats[0] # T.quat2mat(prev_eef_quat[0])
    if cur_rot_from_env:
        curr_rot[:, [1, 2]] = curr_rot[:, [2, 1]]
        curr_rot = -1 * curr_rot
    delta_rot_mat = target_rot.dot(curr_rot.T)
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
    action[3:6] = delta_rotation  

    curr_rot = T.quat2mat(prev_eef_quat[1])
    target_rot = eef_rot_mats[1] # T.quat2mat(prev_eef_quat[1])
    if cur_rot_from_env:
        curr_rot[:, [1, 2]] = curr_rot[:, [2, 1]]
        curr_rot[:, 1] = -1 * curr_rot[:, 1]
    delta_rot_mat = target_rot.dot(curr_rot.T)
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
    action[15:18] = delta_rotation
    return action

def eef_to_action_single_arm(eef_pos, eef_rot_mats, original_action, prev_eef_pos, prev_eef_quat, max_dpos=0.05, max_drot=0.5, cur_rot_from_env=False):
    """
    Convert end-effector pose target actions to simulation-executable action space.

    args:
        eef_pos (np.ndarray): target end-effector position (2, 3)
        eef_rot_mats (np.ndarray): target end-effector orientation as rotation matrices (2, 3, 3)
        original_action (np.ndarray): original action to copy other dimensions from (15,)
        prev_eef_pos (np.ndarray): previous end-effector position (2, 3)
        prev_eef_quat (np.ndarray): previous end-effector orientation as quaternion (2, 4)
        max_dpos (float): maximum position delta
        max_drot (float): maximum rotation delta in axis-angle representation
        cur_rot_from_env (bool): rotation matrices from environment are slightly different from that ones from the dataset.
            If true, these rotation matrices must be adjusted before computing the delta rotation. Otherwise, the delta rotation
            will be incorrect.
    """
    max_dpos = 0.05
    max_drot = 0.5

    action = np.copy(original_action)

    # Set robot action in position space
    action[0:3] = eef_pos[0] - prev_eef_pos[0]
    action[0:3] = np.clip(action[0:3] / max_dpos, -1., 1.)

    # Set robot action in rotation space
    curr_rot = T.quat2mat(prev_eef_quat[0])
    target_rot = eef_rot_mats[0] # T.quat2mat(prev_eef_quat[0])
    if cur_rot_from_env:
        curr_rot[:, [1, 2]] = curr_rot[:, [2, 1]]
        curr_rot = -1 * curr_rot
    delta_rot_mat = target_rot.dot(curr_rot.T)
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
    action[3:6] = delta_rotation  
    return action

def segment_trajectory(data, demo_id):
    subtask_signals = data['data'][demo_id]['datagen_info']['subtask_term_signals']['lid_off_ground']
    print(subtask_signals)  # Example: [0, 0, 0, 1, 0, 0, 1]
    segments = []
    current_segment = []
    # for i, signal in enumerate(subtask_signals):
    flip_flag = False
    for i in range(len(data['data'][demo_id]['actions'])):
        signal = subtask_signals[i]
        if signal == 1 and current_segment and not flip_flag:
            segments.append(current_segment)
            current_segment = []
            current_segment.append(i)
            flip_flag = True
        elif signal == 1 and current_segment:
            current_segment.append(i)
        elif signal == 0:
            current_segment.append(i)
    if current_segment:
        segments.append(current_segment)
    return segments