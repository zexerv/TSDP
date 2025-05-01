import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw # Import fastdtw
import seaborn as sns # For setting style
from sklearn.preprocessing import MinMaxScaler # For normalization

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# Option 1: Load all CSV files in the folder
LOAD_ALL_FILES = True
# Option 2: Specify a list of filenames (if LOAD_ALL_FILES is False)
# FILE_LIST = ['1.csv', '3.csv', '5.csv']
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# ***** NEW: Select Alignment Type *****
# Options: 'DTW' (position only), 'DDTW' (derivative only), 'Combined' (pos + deriv)
ALIGNMENT_TYPE = 'Combined'

# Features to use for Alignment distance calculation
# These are the base features from which derivatives might also be calculated
ALIGNMENT_BASE_FEATURE_COLS = ['tx', 'ty', 'tz']

# --- Normalization Config ---
# Apply normalization before calculating combined DTW distance? Highly recommended for 'Combined'.
NORMALIZE_FOR_COMBINED = True

print(f"Alignment Type: {ALIGNMENT_TYPE}")
print(f"Base features for Alignment: {ALIGNMENT_BASE_FEATURE_COLS}")
if ALIGNMENT_TYPE == 'Combined':
    print(f"Normalization before combined DTW: {NORMALIZE_FOR_COMBINED}")

# --- Data Loading ---
def load_selected_data(parent_folder, load_all=True, file_list=None, feature_columns=None):
    """Loads specified trajectories and features from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    # Define base column lists globally for clarity
    POS_COLS = ['tx', 'ty', 'tz']
    ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    # Load all columns initially to allow warping full data later
    required_columns = list(set(POS_COLS + ROT_COLS + (feature_columns if feature_columns else [])))


    if feature_columns is None:
        feature_columns = ['tx', 'ty', 'tz'] # Default if not provided

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            try: filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError: filenames.sort()
            print(f"Found {len(filenames)} CSV files to load: {filenames}")
            if not filenames: print(f"Warning: No CSV files found in {parent_folder}"); return [], []
        except FileNotFoundError: print(f"Error: Parent folder not found - {parent_folder}"); return [], []
    elif file_list: filenames = file_list; print(f"Loading specified files: {filenames}")
    else: print("Error: No files specified to load."); return [], []

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols: raise KeyError(f"Missing columns: {missing_cols}")

            # Store the full DataFrame initially
            all_trajectories.append(df[required_columns])
            loaded_filenames.append(filename)

        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error: Missing expected column in {file_path}: {e}")
        except Exception as e: print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames # Return list of DataFrames

# --- Alignment (DTW/DDTW/Combined) ---

def calculate_derivative(trajectory):
    """Estimates derivative using simple finite difference."""
    if len(trajectory) < 2:
        return np.zeros_like(trajectory)
    derivative = np.diff(trajectory, axis=0, prepend=trajectory[0:1,:])
    return derivative

# ***** MODIFIED: Handles DTW, DDTW, and Combined *****
def align_trajectories(trajectories_df_list, alignment_type, alignment_base_feature_cols, normalize_combined=True):
    """
    Aligns all trajectories to the longest one using DTW, DDTW, or a combined approach.

    Args:
        trajectories_df_list (list): List of pandas DataFrames containing trajectories.
        alignment_type (str): 'DTW', 'DDTW', or 'Combined'.
        alignment_base_feature_cols (list): Columns used for position/base features.
        normalize_combined (bool): Whether to normalize features before combined DTW.

    Returns:
        tuple: (aligned_trajectories_df, reference_index)
               aligned_trajectories_df: List of warped trajectory DataFrames.
               reference_index: Index of the reference trajectory.
    """
    if not trajectories_df_list: return [], -1
    if alignment_type not in ['DTW', 'DDTW', 'Combined']:
        raise ValueError("alignment_type must be 'DTW', 'DDTW', or 'Combined'")

    # Keep original full DFs for warping later
    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist() # Get col order

    # Extract only the base features needed for alignment/derivative calculation
    trajectories_base_np = [df[alignment_base_feature_cols].values for df in trajectories_df_list]

    lengths = [len(t) for t in trajectories_base_np]
    if not lengths: return [], -1
    reference_index = np.argmax(lengths)
    reference_trajectory_base = trajectories_base_np[reference_index]
    reference_trajectory_full = trajectories_full_np[reference_index]
    ref_len = len(reference_trajectory_base)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for {alignment_type}.")

    aligned_trajectories_np = []
    for i, traj_base in enumerate(trajectories_base_np):
        traj_full = trajectories_full_np[i] # Get corresponding full trajectory
        print(f"  Aligning trajectory {i} (length {len(traj_base)}) to reference...")
        if len(traj_base) == 0:
             print(f"    Skipping empty trajectory {i}")
             aligned_trajectories_np.append(np.zeros_like(reference_trajectory_full))
             continue
        if i == reference_index:
            aligned_trajectories_np.append(reference_trajectory_full.copy())
            continue

        # --- Prepare data for fastdtw based on alignment_type ---
        ref_data_for_dtw = None
        traj_data_for_dtw = None

        if alignment_type == 'DTW':
            ref_data_for_dtw = reference_trajectory_base
            traj_data_for_dtw = traj_base
        elif alignment_type == 'DDTW':
            ref_data_for_dtw = calculate_derivative(reference_trajectory_base)
            traj_data_for_dtw = calculate_derivative(traj_base)
            if not np.any(ref_data_for_dtw) and not np.any(traj_data_for_dtw):
                 print("    Warning: Both derivative sequences zero. Using DTW fallback.")
                 ref_data_for_dtw = reference_trajectory_base
                 traj_data_for_dtw = traj_base
        elif alignment_type == 'Combined':
            # Calculate derivatives
            ref_deriv = calculate_derivative(reference_trajectory_base)
            traj_deriv = calculate_derivative(traj_base)

            # Select base features
            ref_pos = reference_trajectory_base
            traj_pos = traj_base

            if normalize_combined:
                # Normalize position and derivatives separately across the whole dataset (or per pair)
                # For simplicity, normalizing per pair here. For better results, fit scaler on all data.
                scaler_pos = MinMaxScaler()
                scaler_deriv = MinMaxScaler()

                # Fit on combined data for consistent scaling (important!)
                combined_pos = np.vstack((ref_pos, traj_pos))
                combined_deriv = np.vstack((ref_deriv, traj_deriv))

                # Avoid fitting on zero-variance data
                if np.ptp(combined_pos, axis=0).min() > 1e-6: # Check range > epsilon
                     scaler_pos.fit(combined_pos)
                     ref_pos_norm = scaler_pos.transform(ref_pos)
                     traj_pos_norm = scaler_pos.transform(traj_pos)
                else:
                     print("    Warning: Low variance in position data, skipping pos normalization.")
                     ref_pos_norm = ref_pos
                     traj_pos_norm = traj_pos

                if np.ptp(combined_deriv, axis=0).min() > 1e-6:
                     scaler_deriv.fit(combined_deriv)
                     ref_deriv_norm = scaler_deriv.transform(ref_deriv)
                     traj_deriv_norm = scaler_deriv.transform(traj_deriv)
                else:
                     print("    Warning: Low variance in derivative data, skipping deriv normalization.")
                     ref_deriv_norm = ref_deriv
                     traj_deriv_norm = traj_deriv

                # Concatenate normalized features
                ref_data_for_dtw = np.hstack((ref_pos_norm, ref_deriv_norm))
                traj_data_for_dtw = np.hstack((traj_pos_norm, traj_deriv_norm))
            else:
                # Concatenate raw features (might lead to scaling issues)
                ref_data_for_dtw = np.hstack((ref_pos, ref_deriv))
                traj_data_for_dtw = np.hstack((traj_pos, traj_deriv))


        # Perform fastdtw on the prepared data
        distance, path = fastdtw(ref_data_for_dtw, traj_data_for_dtw, dist=euclidean)
        print(f"    Alignment distance ({alignment_type}): {distance:.4f}")

        # --- Warp the ORIGINAL FULL trajectory using the obtained path ---
        path_array = np.array(path)
        # Initialize warped trajectory to hold ALL original features
        warped_traj_full = np.zeros_like(reference_trajectory_full)
        ref_indices_in_path = path_array[:, 0]; traj_indices_in_path = path_array[:, 1]

        for ref_idx in range(ref_len):
            matching_path_indices = np.where(ref_indices_in_path == ref_idx)[0]
            if len(matching_path_indices) > 0:
                traj_idx = traj_indices_in_path[matching_path_indices[-1]]
                traj_idx = min(traj_idx, len(traj_full) - 1)
                warped_traj_full[ref_idx] = traj_full[traj_idx] # Warp the full data
            else:
                if ref_idx > 0: warped_traj_full[ref_idx] = warped_traj_full[ref_idx - 1]
                elif len(traj_full) > 0: warped_traj_full[ref_idx] = traj_full[0]
        aligned_trajectories_np.append(warped_traj_full)

    # Convert back to DataFrames
    aligned_trajectories_df = [pd.DataFrame(data=arr, columns=required_columns) for arr in aligned_trajectories_np]

    return aligned_trajectories_df, reference_index

# --- Visualization ---

def plot_all_original_trajectories(original_trajectories_df_list, feature_cols):
    """Plots X, Y, Z components of all original trajectories on the same axes in black."""
    n_traj = len(original_trajectories_df_list)
    if n_traj == 0: print("No valid original trajectories to plot."); return

    # Extract numpy arrays for the features to plot
    original_trajectories_np = [df[feature_cols].values for df in original_trajectories_df_list if not df.empty and all(c in df.columns for c in feature_cols)]
    original_trajectories_np = [t for t in original_trajectories_np if len(t)>0] # Filter empty after selection
    if not original_trajectories_np: print("Not enough valid trajectories with specified features to plot originals."); return

    n_features = original_trajectories_np[0].shape[1]
    n_plot_dims = min(n_features, 3)
    dim_labels = feature_cols[:n_plot_dims]

    fig, axs = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=False)
    if n_plot_dims == 1: axs = [axs]
    fig.suptitle('All Original Trajectories (Overlaid)', fontsize=16)
    sns.set_style("whitegrid")

    for i in range(len(original_trajectories_np)):
        traj = original_trajectories_np[i]
        time_steps = np.arange(len(traj))
        for j in range(n_plot_dims):
            axs[j].plot(time_steps, traj[:, j], color='black', alpha=0.5, linewidth=1)
            axs[j].set_ylabel(dim_labels[j])
            if j == n_plot_dims - 1: axs[j].set_xlabel('Time Step (Original)')
            axs[j].grid(True, linestyle=':')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

def plot_all_aligned_trajectories(aligned_trajectories_df_list, feature_cols, alignment_type):
    """Plots X, Y, Z components of all aligned trajectories on the same axes in black."""
    n_traj = len(aligned_trajectories_df_list)
    if n_traj == 0: print("No valid aligned trajectories to plot."); return

    # Extract numpy arrays for the features to plot
    aligned_trajectories_np = [df[feature_cols].values for df in aligned_trajectories_df_list if not df.empty and all(c in df.columns for c in feature_cols)]
    aligned_trajectories_np = [t for t in aligned_trajectories_np if len(t)>0]
    if not aligned_trajectories_np: print("Not enough valid trajectories with specified features to plot aligned."); return

    n_features = aligned_trajectories_np[0].shape[1]
    n_plot_dims = min(n_features, 3)
    ref_len = len(aligned_trajectories_np[0])
    aligned_time = np.arange(ref_len)
    dim_labels = feature_cols[:n_plot_dims]

    fig, axs = plt.subplots(n_plot_dims, 1, figsize=(12, 4 * n_plot_dims), sharex=True)
    if n_plot_dims == 1: axs = [axs]
    fig.suptitle(f'All {alignment_type}-Aligned Trajectories (Overlaid)', fontsize=16)
    sns.set_style("whitegrid")

    for i in range(len(aligned_trajectories_np)):
        traj = aligned_trajectories_np[i]
        if len(traj) != ref_len: continue
        for j in range(n_plot_dims):
            axs[j].plot(aligned_time, traj[:, j], color='black', alpha=0.5, linewidth=1)
            axs[j].set_ylabel(dim_labels[j])
            if j == n_plot_dims - 1: axs[j].set_xlabel(f'Time Step ({alignment_type} Aligned)')
            axs[j].grid(True, linestyle=':')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data (loads all pos+rot columns)
    original_trajs_df, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH,
        load_all=LOAD_ALL_FILES,
        file_list=FILE_LIST
        # feature_columns argument removed, loads all necessary columns by default
    )

    if not original_trajs_df:
        print("Exiting: No trajectories loaded.")
        exit()

    # 2. Plot all original trajectories together (using base alignment features)
    print("\nPlotting all original trajectories...")
    plot_all_original_trajectories(original_trajs_df, ALIGNMENT_BASE_FEATURE_COLS)

    # 3. Perform Alignment (DTW, DDTW, or Combined)
    # ***** MODIFIED: Call generic alignment function *****
    aligned_trajs_df, ref_idx = align_trajectories(
        original_trajs_df,
        ALIGNMENT_TYPE,
        ALIGNMENT_BASE_FEATURE_COLS,
        NORMALIZE_FOR_COMBINED # Pass normalization flag
        )

    # 4. Plot all aligned trajectories together
    if aligned_trajs_df:
        print("\nPlotting all aligned trajectories...")
        # Plot using the same base features used for alignment distance
        plot_all_aligned_trajectories(aligned_trajs_df, ALIGNMENT_BASE_FEATURE_COLS, ALIGNMENT_TYPE)
    else:
        print("Alignment failed or produced no valid trajectories, cannot plot aligned results.")

    # --- Ready for downstream processing ---
    valid_aligned_trajs_df = [df for df in aligned_trajs_df if not df.empty]
    if valid_aligned_trajs_df:
        print(f"\nVariable 'valid_aligned_trajs_df' contains {len(valid_aligned_trajs_df)} non-empty warped trajectory DataFrames.")
        # These DataFrames contain ALL original columns, warped according to the alignment path.
        # You can select the columns needed for HMM/CPD/ProMP from these DataFrames.
    else:
        print("\nNo valid aligned trajectories available.")

