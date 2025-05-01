import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw # Import fastdtw
import seaborn as sns # For setting style

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# Option 1: Load all CSV files in the folder
LOAD_ALL_FILES = True
# Option 2: Specify a list of filenames (if LOAD_ALL_FILES is False)
# FILE_LIST = ['1.csv', '3.csv', '5.csv']
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# Features to use for DTW distance calculation
# Defaulting to position, but can be changed
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']

print(f"Using features for DTW: {DTW_FEATURE_COLS}")

# --- Data Loading ---

def load_selected_data(parent_folder, load_all=True, file_list=None, feature_columns=None):
    """Loads specified trajectories and features from CSV files."""
    all_trajectories = []
    loaded_filenames = []

    if feature_columns is None:
        feature_columns = ['tx', 'ty', 'tz'] # Default if not provided

    if load_all:
        try:
            # List only .csv files
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            # Optional: Sort filenames numerically if they represent sequence numbers
            try:
                 filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                 filenames.sort() # Sort alphabetically if not purely numeric
            print(f"Found {len(filenames)} CSV files to load: {filenames}")
            if not filenames:
                 print(f"Warning: No CSV files found in {parent_folder}")
                 return [], []
        except FileNotFoundError:
            print(f"Error: Parent folder not found - {parent_folder}")
            return [], []
    elif file_list:
        filenames = file_list
        print(f"Loading specified files: {filenames}")
    else:
        print("Error: No files specified to load.")
        return [], []

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        print(f"  Loading {file_path}...")
        try:
            df = pd.read_csv(file_path)
            # Select only the specified feature columns for DTW
            trajectory_data = df[feature_columns].values
            if trajectory_data.shape[1] != len(feature_columns):
                 raise ValueError(f"Expected {len(feature_columns)} feature columns, found {trajectory_data.shape[1]}")

            # Basic NaN check (consider more robust handling if needed)
            if np.isnan(trajectory_data).any():
                print(f"Warning: NaN values found in {file_path}. Skipping file.")
                continue
            if trajectory_data.shape[0] == 0:
                 print(f"Warning: Trajectory data is empty in {file_path}. Skipping file.")
                 continue

            all_trajectories.append(trajectory_data)
            loaded_filenames.append(filename)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except KeyError as e:
            print(f"Error: Missing expected column in {file_path}: {e}")
            print(f"  Expected columns: {feature_columns}")
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories:
         print("Warning: No valid trajectory data loaded.")

    return all_trajectories, loaded_filenames

# --- DTW Alignment ---

def align_trajectories_dtw(trajectories):
    """
    Aligns all trajectories to the longest one using DTW.

    Args:
        trajectories (list): A list of numpy arrays, where each array is a trajectory
                             (n_points, n_features).

    Returns:
        tuple: (aligned_trajectories, reference_trajectory, reference_index)
               aligned_trajectories: List of warped trajectories (same length as reference).
               reference_trajectory: The longest trajectory used as reference.
               reference_index: The index of the reference trajectory in the original list.
    """
    if not trajectories:
        return [], None, -1

    # Find the longest trajectory as the reference
    lengths = [len(t) for t in trajectories]
    if not lengths: # Handle case where all files were skipped
         return [], None, -1
    reference_index = np.argmax(lengths)
    reference_trajectory = trajectories[reference_index]
    ref_len = len(reference_trajectory)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference.")

    aligned_trajectories = []

    for i, traj in enumerate(trajectories):
        print(f"  Aligning trajectory {i} (length {len(traj)}) to reference...")
        if len(traj) == 0:
             print(f"    Skipping empty trajectory {i}")
             # Add an empty array or handle appropriately if needed downstream
             # For now, let's add an array of zeros matching ref shape
             aligned_trajectories.append(np.zeros_like(reference_trajectory))
             continue

        if i == reference_index:
            # The reference trajectory is already aligned to itself
            aligned_trajectories.append(reference_trajectory.copy())
            continue

        # Perform DTW using fastdtw
        # The distance function uses Euclidean distance on the feature vectors
        distance, path = fastdtw(reference_trajectory, traj, dist=euclidean)
        print(f"    DTW distance: {distance:.4f}")

        # Warp the current trajectory based on the DTW path
        path_array = np.array(path)
        warped_traj = np.zeros_like(reference_trajectory) # Initialize with ref shape

        ref_indices_in_path = path_array[:, 0]
        traj_indices_in_path = path_array[:, 1]

        for ref_idx in range(ref_len):
            matching_path_indices = np.where(ref_indices_in_path == ref_idx)[0]
            if len(matching_path_indices) > 0:
                traj_idx = traj_indices_in_path[matching_path_indices[-1]]
                traj_idx = min(traj_idx, len(traj) - 1)
                warped_traj[ref_idx] = traj[traj_idx]
            else:
                print(f"Warning: No matching path point found for reference index {ref_idx} while warping trajectory {i}. Repeating previous point.")
                if ref_idx > 0:
                    warped_traj[ref_idx] = warped_traj[ref_idx - 1]
                # else: Handle first point case if necessary (e.g., use traj[0])
                elif len(traj) > 0:
                     warped_traj[ref_idx] = traj[0]


        aligned_trajectories.append(warped_traj)

    return aligned_trajectories, reference_trajectory, reference_index

# --- Visualization ---

# MODIFIED: Plot all original trajectories together in black
def plot_all_original_trajectories(original_trajectories, feature_cols):
    """Plots X, Y, Z components of all original trajectories on the same axes in black."""
    n_traj = len(original_trajectories)
    if n_traj == 0:
        print("No valid original trajectories to plot together.")
        return

    # Ensure trajectories are not empty before proceeding
    original_trajectories = [t for t in original_trajectories if len(t)>0]
    if not original_trajectories:
         print("Not enough valid non-empty trajectories to plot originals.")
         return

    n_features = original_trajectories[0].shape[1]
    n_plot_dims = min(n_features, 3) # Plot up to 3 dimensions (e.g., X, Y, Z)

    if len(feature_cols) >= n_plot_dims:
        dim_labels = feature_cols[:n_plot_dims]
    else:
        dim_labels = [f'Dim {i}' for i in range(n_plot_dims)]

    fig, axs = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=False) # Don't share X for originals
    if n_plot_dims == 1: # Ensure axs is always indexable
        axs = [axs]
    fig.suptitle('All Original Trajectories (Overlaid)', fontsize=16)

    sns.set_style("whitegrid")

    for i in range(n_traj):
        traj = original_trajectories[i]
        time_steps = np.arange(len(traj))

        for j in range(n_plot_dims):
            ax = axs[j]
            # Plot in black with some transparency
            ax.plot(time_steps, traj[:, j], color='black', alpha=0.5, linewidth=1)
            ax.set_ylabel(dim_labels[j])
            if j == n_plot_dims - 1: # X-label only on bottom plot
                ax.set_xlabel('Time Step (Original)')
            ax.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for title
    plt.show()

# NEW: Plot all aligned trajectories together in black
def plot_all_aligned_trajectories(aligned_trajectories, feature_cols):
    """Plots X, Y, Z components of all aligned trajectories on the same axes in black."""
    n_traj = len(aligned_trajectories)
    if n_traj == 0:
        print("No valid aligned trajectories to plot together.")
        return

    # Ensure trajectories are not empty before proceeding
    aligned_trajectories = [t for t in aligned_trajectories if len(t)>0]
    if not aligned_trajectories:
         print("Not enough valid non-empty trajectories to plot aligned.")
         return

    n_features = aligned_trajectories[0].shape[1]
    n_plot_dims = min(n_features, 3) # Plot up to 3 dimensions (e.g., X, Y, Z)
    ref_len = len(aligned_trajectories[0]) # All should have same length
    aligned_time = np.arange(ref_len)

    if len(feature_cols) >= n_plot_dims:
        dim_labels = feature_cols[:n_plot_dims]
    else:
        dim_labels = [f'Dim {i}' for i in range(n_plot_dims)]

    fig, axs = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=True) # Share X for aligned
    if n_plot_dims == 1: # Ensure axs is always indexable
        axs = [axs]
    fig.suptitle('All DTW-Aligned Trajectories (Overlaid)', fontsize=16)

    sns.set_style("whitegrid")

    for i in range(n_traj):
        traj = aligned_trajectories[i]
        if len(traj) != ref_len: # Skip if length mismatch (e.g., from skipped empty original)
            print(f"Warning: Skipping aligned trajectory {i} due to length mismatch.")
            continue

        for j in range(n_plot_dims):
            ax = axs[j]
            # Plot in black with some transparency
            ax.plot(aligned_time, traj[:, j], color='black', alpha=0.5, linewidth=1)
            ax.set_ylabel(dim_labels[j])
            if j == n_plot_dims - 1: # X-label only on bottom plot
                ax.set_xlabel('Time Step (Aligned to Reference)')
            ax.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for title
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    original_trajs, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH,
        load_all=LOAD_ALL_FILES,
        file_list=FILE_LIST,
        feature_columns=DTW_FEATURE_COLS
    )

    if not original_trajs:
        print("Exiting: No trajectories loaded.")
        exit()

    # 2. Plot all original trajectories together
    print("\nPlotting all original trajectories...")
    plot_all_original_trajectories(original_trajs, DTW_FEATURE_COLS)

    # 3. Perform DTW Alignment
    aligned_trajs, ref_traj, ref_idx = align_trajectories_dtw(original_trajs)

    # 4. Plot all aligned trajectories together
    if aligned_trajs:
        print("\nPlotting all aligned trajectories...")
        plot_all_aligned_trajectories(aligned_trajs, DTW_FEATURE_COLS)
    else:
        print("Alignment failed or produced no valid trajectories, cannot plot aligned results.")

    # --- Ready for HMM ---
    # Filter out potentially empty/zero arrays added during alignment if originals were skipped
    valid_aligned_trajs = [t for t in aligned_trajs if len(t) > 0 and np.any(t)]
    if valid_aligned_trajs:
        print(f"\nVariable 'valid_aligned_trajs' contains {len(valid_aligned_trajs)} non-empty warped trajectories ready for HMM input.")
    else:
        print("\nNo valid aligned trajectories available for HMM input.")

