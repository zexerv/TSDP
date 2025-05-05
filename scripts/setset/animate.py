#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Robot Trajectory Animator from .npy File with Static Frames

Loads a C-space trajectory from a .npy file, interpolates it,
loads environment/interface transforms from config files,
and creates an animated 3D plot showing the robot's motion,
its TCP coordinate frame, and the static Device/Interface frames.

Usage:
    python animate.py <path_to_trajectory.npy> [config_dir] [num_interpolated_points]

Arguments:
    path_to_trajectory.npy: Path to the input .npy file (Nx6 C-space trajectory, radians).
    config_dir (optional): Path to the directory containing config.py and YAML files.
                           Defaults to './config'.
    num_interpolated_points (optional): Total animation frames (default: 200). Min: 2.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D # For legend proxies
import math
import argparse
from pathlib import Path
import yaml
import importlib.util # To load config.py dynamically
import traceback
from typing import Optional
# --- SciPy Check (Required for Robust Transformations) ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
    print("INFO: Using scipy.spatial.transform for rotations.")

    def quat_wxyz_to_matrix(quat_wxyz):
        """Converts quaternion [w, x, y, z] to 3x3 rotation matrix using scipy."""
        try:
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]] # Convert to scipy format [x, y, z, w]
            return R.from_quat(quat_xyzw).as_matrix()
        except Exception as e:
            print(f"ERROR in quat_wxyz_to_matrix: {e}")
            return np.identity(3) # Return identity on error

    def matrix_from_pose_dict(pose_dict):
        """ Creates 4x4 matrix from pose dict {position: [x,y,z], quaternion: [w,x,y,z]}. """
        pos = np.array(pose_dict['position'])
        quat_raw = np.array(pose_dict['quaternion'])
        if len(pos) != 3: raise ValueError("Position must have 3 elements")
        if len(quat_raw) != 4: raise ValueError("Quaternion must have 4 elements")
        quat_wxyz = quat_raw # Assuming wxyz format from YAML
        norm = np.linalg.norm(quat_wxyz)
        if norm < 1e-6: quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0]) # Handle zero quaternion
        else: quat_wxyz = quat_wxyz / norm
        rot_matrix = quat_wxyz_to_matrix(quat_wxyz)
        matrix = np.identity(4); matrix[:3, :3] = rot_matrix; matrix[:3, 3] = pos
        return matrix

except ImportError:
    print("CRITICAL ERROR: scipy not found. Install with 'pip install scipy'.")
    SCIPY_AVAILABLE = False
    # Define dummy functions to avoid immediate crash, but they will fail later
    def quat_wxyz_to_matrix(quat_wxyz): raise NotImplementedError("Scipy required")
    def matrix_from_pose_dict(pose_dict): raise NotImplementedError("Scipy required")

# --- Robot Parameters and Forward Kinematics ---
# Assuming UR5e parameters. Modify if using a different robot.

# Denavit-Hartenberg parameters for UR5e
DH_PARAMS_UR5E = [
    {'a': 0.0,    'alpha': math.pi/2,  'd': 0.1625, 'theta_offset': 0.0},
    {'a': -0.425, 'alpha': 0.0,        'd': 0.0,    'theta_offset': 0.0},
    {'a': -0.3922,'alpha': 0.0,        'd': 0.0,    'theta_offset': 0.0},
    {'a': 0.0,    'alpha': math.pi/2,  'd': 0.1333, 'theta_offset': 0.0},
    {'a': 0.0,    'alpha': -math.pi/2, 'd': 0.0997, 'theta_offset': 0.0},
    {'a': 0.0,    'alpha': 0.0,        'd': 0.0996, 'theta_offset': 0.0}
]

# *** CORRECT TCP OFFSET ***
TCP_Z_OFFSET = 0.1565 # Set to the correct value
H_FLANGE_TCP = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, TCP_Z_OFFSET],
    [0, 0, 0, 1]
], dtype=float)

def dh_matrix(a, alpha, d, theta):
    """Calculates the DH transformation matrix."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_t, -sin_t*cos_a,  sin_t*sin_a, a*cos_t],
        [sin_t,  cos_t*cos_a, -cos_t*sin_a, a*sin_t],
        [    0,        sin_a,        cos_a,       d],
        [    0,            0,            0,       1]
    ])

def forward_kinematics(joint_angles_rad, dh_params=DH_PARAMS_UR5E, H_flange_tcp=H_FLANGE_TCP):
    """
    Calculates Forward Kinematics for the given joint angles.
    Returns: (joint_positions, T0_TCP) or (None, None) on failure.
             joint_positions: List of 3D coordinates [base, j1..j6, TCP].
             T0_TCP: 4x4 pose matrix of the TCP relative to the base.
    """
    if len(joint_angles_rad) != len(dh_params):
        # print(f"Error: FK expected {len(dh_params)} angles, got {len(joint_angles_rad)}") # Less verbose
        return None, None
    try:
        transforms = [np.identity(4)]; T_prev = transforms[0]
        for i in range(len(dh_params)):
            p = dh_params[i]
            T_i_minus_1_to_i = dh_matrix(p['a'], p['alpha'], p['d'], joint_angles_rad[i] + p['theta_offset'])
            T_curr = T_prev @ T_i_minus_1_to_i
            transforms.append(T_curr); T_prev = T_curr
        T0_flange = transforms[-1]; T0_TCP = T0_flange @ H_flange_tcp
        joint_positions = [T[:3, 3] for T in transforms]; joint_positions.append(T0_TCP[:3, 3])
        return joint_positions, T0_TCP
    except Exception as e:
        # print(f"Error during Forward Kinematics calculation: {e}") # Less verbose
        return None, None

# --- Interpolation Function ---
def interpolate_trajectory(trajectory_rad, num_points_out):
    """Linearly interpolates a C-space trajectory. Handles angle wrapping."""
    num_points_in, num_joints = trajectory_rad.shape
    if num_points_in < 2: return trajectory_rad # Cannot interpolate
    diffs = np.diff(trajectory_rad, axis=0); diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    distances = np.linalg.norm(diffs, axis=1); cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    if cumulative_dist[-1] < 1e-6: return np.tile(trajectory_rad[0], (num_points_out, 1))
    interp_times = np.linspace(0, cumulative_dist[-1], num_points_out)
    interpolated_traj = np.zeros((num_points_out, num_joints))
    unwrapped_traj = np.unwrap(trajectory_rad, axis=0) # Interpolate unwrapped angles
    for j in range(num_joints): interpolated_traj[:, j] = np.interp(interp_times, cumulative_dist, unwrapped_traj[:, j])
    return interpolated_traj # Return unwrapped for smooth animation steps

# --- YAML Loading Helpers ---
def load_yaml_file(filepath: Path) -> Optional[dict]:
    """Loads a YAML file safely."""
    if not filepath.is_file(): print(f"ERROR: YAML file not found: {filepath}"); return None
    try:
        with open(filepath, 'r') as f: data = yaml.safe_load(f)
        return data if data is not None else {}
    except Exception as e: print(f"ERROR reading/parsing YAML {filepath}: {e}"); return None

def load_interface_transforms(filepath: Path) -> Optional[dict[str, np.ndarray]]:
    """Loads interface transforms (H_D_I) from YAML."""
    config_data = load_yaml_file(filepath);
    if config_data is None: return None
    transforms = {}; loaded_count = 0
    for iface_id, data in config_data.items():
        try:
            if isinstance(data, dict) and 'T_aruco_interface' in data:
                H_D_I = np.array(data['T_aruco_interface'], dtype=float)
                if H_D_I.shape == (4, 4): transforms[iface_id] = H_D_I; loaded_count += 1
        except Exception: pass # Ignore errors for specific interfaces
    print(f"Loaded {loaded_count} interface transforms from {filepath.name}.")
    return transforms if transforms else None

def load_environment_config(filepath: Path) -> Optional[np.ndarray]:
    """Loads environment config and extracts the device pose matrix (H_B_D)."""
    if not SCIPY_AVAILABLE: return None
    config_data = load_yaml_file(filepath);
    if config_data is None: return None
    try:
        dev_data = config_data.get('aruco_device', config_data.get('aruco_device_pose'))
        if dev_data is None: raise KeyError("Cannot find device key")
        pose_data = dev_data.get('pose');
        if pose_data is None: raise KeyError("Missing 'pose' key")
        H_B_D = matrix_from_pose_dict(pose_data) # Uses scipy internally
        print(f"Loaded device pose (H_B_D) from {filepath.name}.")
        return H_B_D
    except Exception as e: print(f"ERROR extracting device pose from {filepath.name}: {e}"); return None

# --- Visualization Functions ---

# Store plotted elements globally to update them
robot_lines_anim = []
tcp_frame_quivers_anim = []
tcp_frame_labels_anim = []

def plot_coordinate_frame_static(ax, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], color=None, linewidth=1, text_offset=1.15):
    """ Plots a static 3D coordinate frame on the given axes. """
    if pose_matrix is None: return
    try:
        origin = pose_matrix[:3, 3]; R_mat = pose_matrix[:3, :3]; colors = color if color else ['r', 'g', 'b']
        for i in range(3):
            axis_vector = R_mat[:, i]; current_color = colors[i] if isinstance(colors, list) else colors
            ax.quiver(origin[0], origin[1], origin[2], axis_vector[0], axis_vector[1], axis_vector[2], length=length, color=current_color, arrow_length_ratio=0.3, linewidth=linewidth, normalize=False)
            if labels and len(labels) == 3: ax.text(origin[0] + axis_vector[0] * length * text_offset, origin[1] + axis_vector[1] * length * text_offset, origin[2] + axis_vector[2] * length * text_offset, labels[i], color=current_color, fontsize=9, ha='center', va='center', zorder=10)
    except Exception as e: print(f"Warning: Failed to plot static coordinate frame: {e}")

def update_robot_plot_anim(ax, q_rad, frame_index, total_frames):
    """ Clears previous robot plot and draws the robot for the given configuration. """
    global robot_lines_anim, tcp_frame_quivers_anim, tcp_frame_labels_anim

    # Remove previous robot elements efficiently
    for artist_list in [robot_lines_anim, tcp_frame_quivers_anim, tcp_frame_labels_anim]:
        for artist in artist_list:
            artist.remove()
        artist_list.clear()

    # Calculate FK for the current configuration
    joint_positions, T_tcp = forward_kinematics(q_rad) # Use internal FK

    if joint_positions is not None and T_tcp is not None:
        # Plot robot links
        jp = np.array(joint_positions)
        line, = ax.plot(jp[:,0], jp[:,1], jp[:,2], marker='o', ms=4, ls='-', lw=2, color='blue', alpha=0.8)
        robot_lines_anim.append(line)

        # Plot TCP Frame
        tcp_origin = T_tcp[:3, 3]; tcp_R_mat = T_tcp[:3, :3]
        tcp_colors = ['r', 'g', 'b']; tcp_labels = ['Xt', 'Yt', 'Zt']
        tcp_frame_length = 0.1; text_offset = 1.15
        for i in range(3):
            axis = tcp_R_mat[:, i]
            qv = ax.quiver(tcp_origin[0],tcp_origin[1],tcp_origin[2], axis[0],axis[1],axis[2], length=tcp_frame_length, color=tcp_colors[i], arrow_length_ratio=0.3, lw=1.5, normalize=False)
            tcp_frame_quivers_anim.append(qv)
            txt = ax.text(tcp_origin[0]+axis[0]*tcp_frame_length*text_offset, tcp_origin[1]+axis[1]*tcp_frame_length*text_offset, tcp_origin[2]+axis[2]*tcp_frame_length*text_offset, tcp_labels[i], color=tcp_colors[i], fontsize=9, ha='center', va='center', zorder=10)
            tcp_frame_labels_anim.append(txt)

    # Update plot title with frame index
    ax.set_title(f"Robot Trajectory Animation (Frame {frame_index+1}/{total_frames})")
    plt.draw() # Redraw the current figure


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Dependency Check ---
    if not SCIPY_AVAILABLE:
        print("FATAL ERROR: scipy library is required for transformations. Install with 'pip install scipy'.")
        sys.exit(1)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Animate robot trajectory from .npy file with static frames.")
    parser.add_argument("trajectory_filepath", type=str, help="Path to the input .npy file (Nx6 C-space trajectory in radians).")
    parser.add_argument("config_dir", type=str, nargs='?', default="./config", help="Path to config directory (default: ./config).")
    parser.add_argument("num_interpolated_points", type=int, nargs='?', default=200, help="Total animation frames (default: 200). Min: 2.")

    try:
        args = parser.parse_args()
        traj_path = Path(args.trajectory_filepath)
        config_dir = Path(args.config_dir)
        num_interp_points = args.num_interpolated_points
        if num_interp_points < 2: num_interp_points = 2
    except SystemExit: sys.exit(1)
    except Exception as e: print(f"Error parsing arguments: {e}"); sys.exit(1)

    # --- Load config.py ---
    config_py_path = config_dir / "config.py"
    if not config_py_path.is_file(): print(f"ERROR: config.py not found in '{config_dir.resolve()}'"); sys.exit(1)
    try:
        spec = importlib.util.spec_from_file_location("config", config_py_path)
        config = importlib.util.module_from_spec(spec); sys.modules['config'] = config
        spec.loader.exec_module(config); print("Successfully loaded config.py")
    except Exception as e: print(f"ERROR loading config.py: {e}"); sys.exit(1)

    # --- Load Transforms needed for static frames ---
    print("\nLoading transforms for static frames...")
    H_B_D = None; H_B_I = None
    try:
        iface_tf_path = (config_dir / Path(config.INTERFACE_TRANSFORMS_YAML_PATH).name).resolve()
        env_cfg_path = (config_dir / Path(config.ENVIRONMENT_YAML_PATH).name).resolve()
        interface_id = config.INTERFACE_ID

        if not iface_tf_path.is_file(): raise FileNotFoundError(f"Interface transforms file not found: {iface_tf_path}")
        if not env_cfg_path.is_file(): raise FileNotFoundError(f"Environment config file not found: {env_cfg_path}")

        interface_transforms = load_interface_transforms(iface_tf_path)
        H_B_D = load_environment_config(env_cfg_path)

        if interface_transforms is not None and H_B_D is not None:
            H_D_I = interface_transforms.get(interface_id)
            if H_D_I is None: raise ValueError(f"Interface ID '{interface_id}' not found in transforms file.")
            H_B_I = H_B_D @ H_D_I # Calculate Base to Interface
        else:
             raise ValueError("Failed to load interface transforms or environment config.")

    except Exception as e:
        print(f"ERROR loading transforms needed for visualization: {e}")
        # Decide if you want to continue without plotting static frames or exit
        # sys.exit(1)
        print("Proceeding without plotting Device/Interface frames.")
        H_B_D = None # Ensure transforms are None if loading failed
        H_B_I = None

    # --- Load Trajectory Data ---
    if not traj_path.is_file(): print(f"ERROR: Trajectory file not found: '{traj_path}'"); sys.exit(1)
    try:
        print(f"Loading trajectory from: {traj_path}")
        original_trajectory_rad = np.load(traj_path)
        print(f"Loaded original trajectory shape: {original_trajectory_rad.shape}")
        if original_trajectory_rad.ndim != 2 or original_trajectory_rad.shape[1] != 6: raise ValueError("Trajectory must be Nx6")
        if len(original_trajectory_rad) < 2: raise ValueError("Trajectory needs at least 2 waypoints")
    except Exception as e: print(f"ERROR loading trajectory file '{traj_path}': {e}"); sys.exit(1)

    # --- Interpolate Trajectory ---
    print(f"Interpolating trajectory to {num_interp_points} points...")
    try: anim_trajectory_rad = interpolate_trajectory(original_trajectory_rad, num_interp_points)
    except Exception as e: print(f"ERROR interpolating trajectory: {e}"); sys.exit(1)
    print(f"Animation trajectory shape: {anim_trajectory_rad.shape}")

    # --- Setup Animation Plot ---
    print("Setting up animation plot...")
    fig_anim = plt.figure("Robot Animation", figsize=(10, 8))
    ax_anim = fig_anim.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
    ax_anim.set_xlabel("X base (m)"); ax_anim.set_ylabel("Y base (m)"); ax_anim.set_zlabel("Z base (m)")
    ax_anim.set_xlim([-1, 1]); ax_anim.set_ylim([-1, 1]); ax_anim.set_zlim([-0.2, 1.5]) # Use config limits?
    ax_anim.set_aspect('equal', adjustable='box'); ax_anim.view_init(elev=30., azim=-60)

    # --- Plot Static Frames ---
    legend_handles_static = []
    # Base Frame
    plot_coordinate_frame_static(ax_anim, np.identity(4), length=0.15, labels=['Xb', 'Yb', 'Zb'], linewidth=2)
    legend_handles_static.append(Line2D([0], [0], color='black', lw=2, label='Base Frame')) # Proxy for RGB
    # Device Frame
    if H_B_D is not None:
        plot_coordinate_frame_static(ax_anim, H_B_D, length=0.12, labels=['Xd', 'Yd', 'Zd'], color='cyan', linewidth=2)
        legend_handles_static.append(Line2D([0], [0], color='cyan', lw=2, label='Device Frame'))
    # Interface Frame
    if H_B_I is not None:
        plot_coordinate_frame_static(ax_anim, H_B_I, length=0.10, labels=['Xi', 'Yi', 'Zi'], color='magenta', linewidth=2)
        legend_handles_static.append(Line2D([0], [0], color='magenta', lw=2, label='Interface Frame'))

    # Add legend for static frames
    if legend_handles_static:
        ax_anim.legend(handles=legend_handles_static, loc='upper left', fontsize='small')

    # --- Add Slider ---
    ax_slider_anim = fig_anim.add_axes([0.15, 0.05, 0.7, 0.03])
    total_frames_anim = len(anim_trajectory_rad)
    slider_anim = Slider(ax=ax_slider_anim, label='Frame', valmin=0, valmax=total_frames_anim - 1, valinit=0, valstep=1, valfmt='%d')

    # --- Slider Update Function ---
    def update_anim(val):
        frame_index = int(slider_anim.val); frame_index = max(0, min(frame_index, total_frames_anim - 1))
        q_current = anim_trajectory_rad[frame_index]
        update_robot_plot_anim(ax_anim, q_current, frame_index, total_frames_anim)

    slider_anim.on_changed(update_anim)

    # --- Initial Plot ---
    print("Plotting initial animation frame...")
    update_anim(0)

    # --- Show Plot ---
    print("\nDisplaying animation window... Use slider to navigate.")
    plt.show()

    print("\nAnimation script finished.")