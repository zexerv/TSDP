#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Reference Frame Visualizer

Loads device and interface transforms from YAML configuration files
and plots the Base, Device, and Interface coordinate frames in 3D.

Usage:
    python visualization_rfs.py [config_dir]

Arguments:
    config_dir (optional): Path to the directory containing config.py,
                           interface_transforms.yaml, and environment.yaml.
                           Defaults to './config'.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D # For legend proxies
import math
import yaml
from pathlib import Path
import argparse
import importlib.util # To load config.py dynamically
from typing import Optional
from scipy.spatial.transform import Rotation as R

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
    print("               Transformation functions will fail.")
    SCIPY_AVAILABLE = False
    # Define dummy functions to avoid immediate crash, but they will fail later
    def quat_wxyz_to_matrix(quat_wxyz): raise NotImplementedError("Scipy required")
    def matrix_from_pose_dict(pose_dict): raise NotImplementedError("Scipy required")

# --- YAML Loading Helpers ---

def load_yaml_file(filepath: Path) -> Optional[dict]:
    """Loads a YAML file safely."""
    if not filepath.is_file():
        print(f"ERROR: YAML file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f: data = yaml.safe_load(f)
        return data if data is not None else {}
    except Exception as e:
        print(f"ERROR reading/parsing YAML file {filepath}: {e}")
        return None

def load_interface_transforms(filepath: Path) -> Optional[dict[str, np.ndarray]]:
    """Loads interface transforms (H_D_I) from YAML."""
    config_data = load_yaml_file(filepath)
    if config_data is None: return None
    transforms = {}
    for interface_id_key, data in config_data.items():
        try:
            if isinstance(data, dict) and 'T_aruco_interface' in data:
                matrix_list = data['T_aruco_interface']
                H_D_I = np.array(matrix_list, dtype=float)
                if H_D_I.shape == (4, 4): transforms[interface_id_key] = H_D_I
                else: print(f"Warn: Invalid matrix shape for {interface_id_key}")
        except Exception as e: print(f"Error processing interface {interface_id_key}: {e}")
    print(f"Loaded {len(transforms)} interface transforms from {filepath.name}.")
    return transforms if transforms else None

def load_environment_config(filepath: Path) -> Optional[np.ndarray]:
    """Loads environment config and extracts the device pose matrix (H_B_D)."""
    if not SCIPY_AVAILABLE: return None # Cannot proceed without scipy
    config_data = load_yaml_file(filepath)
    if config_data is None: return None
    try:
        device_data = config_data.get('aruco_device', config_data.get('aruco_device_pose'))
        if device_data is None: raise KeyError("Cannot find device key")
        pose_data = device_data.get('pose')
        if pose_data is None: raise KeyError("Missing 'pose' key")
        H_B_D = matrix_from_pose_dict(pose_data) # Uses scipy internally
        print(f"Loaded device pose (H_B_D) from {filepath.name}.")
        return H_B_D
    except Exception as e: print(f"ERROR extracting device pose from {filepath.name}: {e}"); return None

# --- Visualization Function ---

def plot_coordinate_frame(ax, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], color=None, linewidth=1, text_offset=1.15):
    """ Plots a 3D coordinate frame on the given axes. """
    if pose_matrix is None: return
    try:
        origin = pose_matrix[:3, 3]
        R_mat = pose_matrix[:3, :3]
        colors = color if color else ['r', 'g', 'b'] # Default to RGB

        for i in range(3):
            axis_vector = R_mat[:, i]
            current_color = colors[i] if isinstance(colors, list) else colors
            # Plot axis line (quiver)
            ax.quiver(origin[0], origin[1], origin[2],
                      axis_vector[0], axis_vector[1], axis_vector[2],
                      length=length, color=current_color, arrow_length_ratio=0.3,
                      linewidth=linewidth, normalize=False) # Use normalize=False
            # Add text label near the arrow tip
            if labels and len(labels) == 3:
                 ax.text(origin[0] + axis_vector[0] * length * text_offset,
                         origin[1] + axis_vector[1] * length * text_offset,
                         origin[2] + axis_vector[2] * length * text_offset,
                         labels[i], color=current_color, fontsize=10,
                         ha='center', va='center', zorder=10) # Ensure text is on top
    except Exception as e:
        print(f"Warning: Failed to plot coordinate frame: {e}")

# --- Main Script ---
if __name__ == "__main__":
    if not SCIPY_AVAILABLE:
        sys.exit("Scipy library is required for transformations. Please install it.")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Visualize Base, Device, and Interface reference frames.")
    parser.add_argument("config_dir", type=str, nargs='?', default="./config",
                        help="Path to the directory containing configuration files (default: ./config).")
    args = parser.parse_args()
    config_dir = Path(args.config_dir)

    print(f"Using configuration directory: {config_dir.resolve()}")

    # --- Load config.py to get file paths and interface ID ---
    config_py_path = config_dir / "config.py"
    if not config_py_path.is_file():
        print(f"ERROR: config.py not found in '{config_dir}'")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("config", config_py_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print("Successfully loaded config.py")
    except Exception as e:
        print(f"ERROR: Failed to load or execute config.py: {e}")
        sys.exit(1)

    # Get required paths and ID from the loaded config module
    try:
        iface_tf_path = config_dir / Path(config.INTERFACE_TRANSFORMS_YAML_PATH).name
        env_cfg_path = config_dir / Path(config.ENVIRONMENT_YAML_PATH).name
        interface_id = config.INTERFACE_ID
        print(f"Interface Transforms Path: {iface_tf_path}")
        print(f"Environment Config Path: {env_cfg_path}")
        print(f"Target Interface ID: {interface_id}")
    except AttributeError as e:
        print(f"ERROR: Missing required variable ({e}) in config.py.")
        sys.exit(1)

    # --- Load Transforms ---
    print("\nLoading transforms...")
    interface_transforms = load_interface_transforms(iface_tf_path)
    H_B_D = load_environment_config(env_cfg_path)

    if interface_transforms is None or H_B_D is None:
        print("ERROR: Failed to load necessary transforms. Exiting.")
        sys.exit(1)

    H_D_I = interface_transforms.get(interface_id)
    if H_D_I is None:
        print(f"ERROR: Interface ID '{interface_id}' not found in '{iface_tf_path.name}'.")
        print(f"Available IDs: {list(interface_transforms.keys())}")
        sys.exit(1)

    # --- Calculate Transforms ---
    try:
        H_B_I = H_B_D @ H_D_I
        print("Calculated H_B_I (Base to Interface)")
    except Exception as e:
        print(f"ERROR calculating H_B_I: {e}")
        sys.exit(1)

    # --- Setup Plot ---
    print("\nSetting up 3D plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Reference Frame Visualization")
    ax.set_xlabel("X base (m)"); ax.set_ylabel("Y base (m)"); ax.set_zlabel("Z base (m)")

    # --- Plot Frames ---
    print("Plotting frames...")
    # Base Frame (at origin)
    plot_coordinate_frame(ax, np.identity(4), length=0.15, labels=['Xb', 'Yb', 'Zb'], linewidth=2)
    # Device Frame (relative to Base)
    plot_coordinate_frame(ax, H_B_D, length=0.12, labels=['Xd', 'Yd', 'Zd'], color='cyan', linewidth=2)
    # Interface Frame (relative to Base)
    plot_coordinate_frame(ax, H_B_I, length=0.10, labels=['Xi', 'Yi', 'Zi'], color='magenta', linewidth=2)

    # --- Adjust Plot Limits ---
    # Collect all origin points to determine plot range
    origins = [np.zeros(3), H_B_D[:3, 3], H_B_I[:3, 3]]
    all_points = np.array(origins)
    min_coords = np.min(all_points, axis=0) - 0.2 # Add padding
    max_coords = np.max(all_points, axis=0) + 0.2 # Add padding
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])
    ax.set_aspect('equal', adjustable='box') # Try to make axes scales equal
    ax.view_init(elev=25., azim=-70) # Adjust view

    # --- Add Legend ---
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Base Frame (Origin)'), # Use black proxy for RGB
        Line2D([0], [0], color='cyan', lw=2, label='Device Frame'),
        Line2D([0], [0], color='magenta', lw=2, label='Interface Frame')
    ]
    ax.legend(handles=legend_elements, loc='best')

    # --- Show Plot ---
    print("\nDisplaying plot... Close window to exit.")
    plt.tight_layout()
    plt.show()

    print("Visualization finished.")
