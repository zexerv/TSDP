#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Robot Trajectory Visualizer from CSV

Reads joint angles from a CSV file and plots snapshots of the robot's
configuration in a 3D space using Matplotlib.

Usage:
    python visualization_main.py <path_to_csv_file.csv> [snapshot_interval]

Arguments:
    path_to_csv_file.csv: Path to the input CSV file.
    snapshot_interval (optional): Plot every Nth point (default: 20).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import argparse # For handling command-line arguments
from pathlib import Path
# --- Robot Parameters and Forward Kinematics (Copied from ik_solver context) ---
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

# TCP offset from flange (example, adjust if needed)
TCP_Z_OFFSET = 0.0 # Set to 0 if CSV angles already account for TCP
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

    Args:
        joint_angles_rad (list or np.array): List/array of 6 joint angles in radians.
        dh_params (list): DH parameters for the robot.
        H_flange_tcp (np.ndarray): Transformation from flange to TCP.

    Returns:
        tuple: (joint_positions, T0_TCP, T0_flange) or (None, None, None) on failure.
               joint_positions: List of 3D coordinates [base, j1, j2, j3, j4, j5, j6, TCP].
               T0_TCP: 4x4 pose matrix of the TCP relative to the base.
               T0_flange: 4x4 pose matrix of the flange relative to the base.
    """
    if len(joint_angles_rad) != len(dh_params):
        print(f"Error: FK expected {len(dh_params)} angles, got {len(joint_angles_rad)}")
        return None, None, None

    try:
        transforms = [np.identity(4)] # Start with base frame identity matrix T_0^0
        T_prev = transforms[0]

        # Calculate transformations from base to each joint frame
        for i in range(len(dh_params)):
            p = dh_params[i]
            # T_i-1^i matrix using current joint angle
            T_i_minus_1_to_i = dh_matrix(p['a'], p['alpha'], p['d'], joint_angles_rad[i] + p['theta_offset'])
            # T_0^i = T_0^{i-1} * T_{i-1}^i
            T_curr = T_prev @ T_i_minus_1_to_i
            transforms.append(T_curr)
            T_prev = T_curr

        # Final transform is T_0^6 (base to flange)
        T0_flange = transforms[-1]

        # Calculate TCP pose: T_0^TCP = T_0^6 * T_6^TCP
        T0_TCP = T0_flange @ H_flange_tcp

        # Extract 3D positions of each frame's origin for plotting links
        # Order: Base(0), J1, J2, J3, J4, J5, J6(Flange), TCP
        joint_positions = [T[:3, 3] for T in transforms] # Positions of Base and Joints 1-6 (Flange)
        joint_positions.append(T0_TCP[:3, 3]) # Add TCP position

        return joint_positions, T0_TCP, T0_flange
    except Exception as e:
        print(f"Error during Forward Kinematics calculation: {e}")
        return None, None, None

# --- Visualization Class ---
class TrajectoryVisualizer3D:
    """ Handles the 3D visualization of robot trajectory snapshots. """
    def __init__(self, title="Robot Trajectory Visualization"):
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Set axis labels as requested
        self.ax.set_xlabel("X base (m)", labelpad=10)
        self.ax.set_ylabel("Y base (m)", labelpad=10)
        self.ax.set_zlabel("Z base (m)", labelpad=10)
        self.ax.set_title(title, fontsize=14, pad=20)
        # Set reasonable default plot limits (adjust if needed)
        self.ax.set_xlim([-1, 1]); self.ax.set_ylim([-1, 1]); self.ax.set_zlim([-0.2, 1.5])
        self.ax.set_aspect('auto') # Use 'auto' for better general viewing
        self.ax.view_init(elev=30., azim=-60) # Adjust viewing angle
        plt.tight_layout()
        self._legend_handles = {}
        # Plot base coordinate frame
        self._plot_coordinate_frame(np.identity(4), length=0.15, labels=['X₀', 'Y₀', 'Z₀'])

    def _plot_coordinate_frame(self, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], color=None, linewidth=1):
        """ Plots a 3D coordinate frame at the given pose. """
        if pose_matrix is None: return
        try:
            origin = pose_matrix[:3, 3]; R_mat = pose_matrix[:3, :3]; colors = color if color else ['r', 'g', 'b']
            for i in range(3):
                axis = R_mat[:, i]; current_color = colors[i] if isinstance(colors, list) else colors
                self.ax.quiver(origin[0], origin[1], origin[2], axis[0], axis[1], axis[2],
                               length=length, color=current_color, arrow_length_ratio=0.3,
                               linewidth=linewidth)
                if labels and isinstance(colors, list):
                    self.ax.text(origin[0] + axis[0]*length*1.1, origin[1] + axis[1]*length*1.1, origin[2] + axis[2]*length*1.1,
                                 labels[i], color=colors[i])
        except Exception as e:
            print(f"Warning: Failed to plot coordinate frame: {e}")

    def plot_robot_snapshot(self, joint_positions, color='k', linewidth=2, style='-', alpha=1.0, label=None, plot_tcp_frame=False, T_tcp=None):
        """ Plots the robot links and optionally the TCP frame for a single snapshot. """
        if not joint_positions or len(joint_positions) < 2: return
        try:
            jp = np.array(joint_positions)
            # Plot links as lines, joints as markers
            handle = self.ax.plot(jp[:,0], jp[:,1], jp[:,2], marker='o', markersize=3, linestyle=style,
                                  linewidth=linewidth, color=color, alpha=alpha)
            line_handle = handle[0] # Get the line object for the legend

            # Store handle for legend only if label is provided and not already stored
            if line_handle and label and label not in self._legend_handles:
                self._legend_handles[label] = line_handle

            # Plot TCP frame if requested and pose is provided
            if plot_tcp_frame and T_tcp is not None:
                self._plot_coordinate_frame(T_tcp, length=0.08, labels=None, color=color, linewidth=linewidth*0.75)

        except Exception as e:
            print(f"Warning: Failed to plot robot snapshot: {e}")

    def add_legend(self, **kwargs):
        """ Adds a legend to the plot using stored handles. """
        if not self._legend_handles: return
        valid_handles = {label: handle for label, handle in self._legend_handles.items() if handle is not None}
        if valid_handles:
            try: self.ax.legend(valid_handles.values(), valid_handles.keys(), **kwargs)
            except Exception as e: print(f"Warning: Failed to add legend: {e}")

    def show(self):
        """ Displays the plot. """
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Visualize robot trajectory from CSV.")
    parser.add_argument("csv_filepath", type=str, help="Path to the input CSV file.")
    parser.add_argument("snapshot_interval", type=int, nargs='?', default=20,
                        help="Plot every Nth point (default: 20).")

    # Handle potential parsing errors
    try:
        args = parser.parse_args()
        csv_path = args.csv_filepath
        snapshot_interval = args.snapshot_interval
        if snapshot_interval <= 0:
            print("Warning: Snapshot interval must be positive. Setting to 1.")
            snapshot_interval = 1
    except SystemExit:
        sys.exit(1) # Exit if argument parsing fails (e.g., missing file path)
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    print(f"Loading trajectory from: {csv_path}")
    print(f"Plotting every {snapshot_interval} points.")

    # --- Load CSV Data ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} waypoints.")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load CSV file '{csv_path}': {e}")
        sys.exit(1)

    # --- Extract Joint Angles ---
    # Define expected column names (case-sensitive)
    joint_columns = [f'joint{i+1}pose' for i in range(6)] # Assumes 6 joints

    # Check if all required columns exist
    missing_cols = [col for col in joint_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required joint columns in CSV: {', '.join(missing_cols)}")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)

    try:
        # Select only the required columns and convert to numpy array
        # Data is already in RADIANS
        joint_trajectory_rad = df[joint_columns].values
        print(f"Extracted joint data (radians) with shape: {joint_trajectory_rad.shape}")
        if joint_trajectory_rad.shape[1] != 6:
             print(f"ERROR: Expected 6 joint columns, found {joint_trajectory_rad.shape[1]}.")
             sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to extract or convert joint data: {e}")
        sys.exit(1)

    # --- Visualization ---
    if len(joint_trajectory_rad) == 0:
        print("No data points found in the trajectory. Exiting.")
        sys.exit(0)

    visualizer = TrajectoryVisualizer3D(title=f"Robot Trajectory from {Path(csv_path).name}")

    num_waypoints = len(joint_trajectory_rad)
    indices_to_plot = range(0, num_waypoints, snapshot_interval)

    print(f"Plotting {len(indices_to_plot)} snapshots...")

    fk_success_count = 0
    for i, idx in enumerate(indices_to_plot):
        q_rad = joint_trajectory_rad[idx]
        joint_positions, T_tcp, _ = forward_kinematics(q_rad)

        if joint_positions is not None:
            fk_success_count += 1
            # Make snapshots fade slightly or change color over time? Example: alpha
            alpha_val = 0.3 + 0.7 * (i / len(indices_to_plot)) if len(indices_to_plot) > 1 else 1.0
            label = None
            if i == 0: label = "Start Pose"
            elif i == len(indices_to_plot) - 1: label = "End Pose"

            visualizer.plot_robot_snapshot(
                joint_positions,
                color='blue',
                linewidth=1.5,
                style='-',
                alpha=alpha_val,
                label=label,
                plot_tcp_frame=(i == 0 or i == len(indices_to_plot) - 1), # Plot frame only at start/end
                T_tcp=T_tcp
            )
        # else: FK failed for this point (error already printed in FK function)

    print(f"Successfully performed FK for {fk_success_count} snapshots.")
    if fk_success_count == 0:
         print("ERROR: Forward kinematics failed for all plotted snapshots.")
    else:
         visualizer.add_legend(loc='best')
         print("Displaying plot...")
         visualizer.show()

    print("Visualization complete.")