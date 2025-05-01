import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import ruptures as rpt # Import ruptures library
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import seaborn as sns

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- File Loading Config ---
LOAD_ALL_FILES = True
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Preprocessing Config ---
APPLY_DTW_ALIGNMENT = True # Apply DTW before CPD?

# --- Feature Config ---
# Features for DTW distance calculation (if APPLY_DTW_ALIGNMENT is True)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to use for Change Point Detection
CPD_POS_COLS = ['tx', 'ty', 'tz']
INCLUDE_ROTATION_CPD = False
CPD_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- CPD Algorithm Config ---
# Choose algorithm: 'Binseg' (Binary Segmentation), 'Pelt' (Penalized), 'Window'
CPD_ALGORITHM = 'Pelt'
# Model assumption for cost function within algorithm (e.g., "l2" for Gaussian process)
CPD_MODEL = 'l2' # Common choices: 'l1', 'l2', 'rbf', 'cosine', 'rank'

# Parameters specific to algorithms:
# For Binseg: Specify the number of change points (segments = n_bkps + 1)
# Corresponds roughly to N_STATES-1 in HMM
N_CHANGEPOINTS_BINSEG = 3 # e.g., for 4 segments (N_STATES=4 -> N_BKPS=3)
# For Pelt: Specify the penalty value (higher penalty -> fewer change points)
# Finding a good penalty often requires experimentation or model selection criteria
PENALTY_PELT = 10 # Example value, TUNE THIS!
# For Window: Specify window width
WINDOW_WIDTH = 50 # Example value

# --- Determine CPD Features ---
# Define base column lists globally for clarity
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
all_load_features = POS_COLS + ROT_COLS # All columns potentially needed

CPD_FEATURE_COLS = CPD_POS_COLS[:] # Start with position
if INCLUDE_ROTATION_CPD:
    CPD_FEATURE_COLS.extend(CPD_ROT_COLS)
N_FEATURES_CPD = len(CPD_FEATURE_COLS)

print(f"CPD using {N_FEATURES_CPD} features: {CPD_FEATURE_COLS}")
print(f"CPD Algorithm: {CPD_ALGORITHM}, Model: {CPD_MODEL}")
if APPLY_DTW_ALIGNMENT:
    print(f"Using features for DTW: {DTW_FEATURE_COLS}")


# --- Data Loading (Identical to HMM script) ---
def load_selected_data(parent_folder, load_all=True, file_list=None):
    """Loads specified trajectories (all pose columns) from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = list(set(all_load_features + DTW_FEATURE_COLS)) # Load all needed cols

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
            all_trajectories.append(df[required_columns]) # Store DataFrame
            loaded_filenames.append(filename)
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error in {file_path}: {e}")
        except Exception as e: print(f"Error loading/processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames # Return list of DataFrames


# --- DTW Alignment (Identical to HMM script) ---
def align_trajectories_dtw(trajectories_df_list, dtw_feature_cols):
    """Aligns trajectories using DTW based on specified features."""
    if not trajectories_df_list: return [], None, -1
    trajectories_dtw_np = [df[dtw_feature_cols].values for df in trajectories_df_list]
    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist()

    lengths = [len(t) for t in trajectories_dtw_np]
    if not lengths: return [], None, -1
    reference_index = np.argmax(lengths)
    reference_trajectory_dtw = trajectories_dtw_np[reference_index]
    reference_trajectory_full = trajectories_full_np[reference_index]
    ref_len = len(reference_trajectory_dtw)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for DTW.")

    aligned_trajectories_np = []
    for i, traj_dtw in enumerate(trajectories_dtw_np):
        traj_full = trajectories_full_np[i]
        if len(traj_dtw) == 0:
             aligned_trajectories_np.append(np.zeros_like(reference_trajectory_full)); continue
        if i == reference_index:
            aligned_trajectories_np.append(reference_trajectory_full.copy()); continue

        distance, path = fastdtw(reference_trajectory_dtw, traj_dtw, dist=euclidean)
        path_array = np.array(path)
        warped_traj_full = np.zeros_like(reference_trajectory_full)
        ref_indices_in_path = path_array[:, 0]; traj_indices_in_path = path_array[:, 1]

        for ref_idx in range(ref_len):
            matching_path_indices = np.where(ref_indices_in_path == ref_idx)[0]
            if len(matching_path_indices) > 0:
                traj_idx = traj_indices_in_path[matching_path_indices[-1]]
                traj_idx = min(traj_idx, len(traj_full) - 1)
                warped_traj_full[ref_idx] = traj_full[traj_idx]
            else:
                if ref_idx > 0: warped_traj_full[ref_idx] = warped_traj_full[ref_idx - 1]
                elif len(traj_full) > 0: warped_traj_full[ref_idx] = traj_full[0]
        aligned_trajectories_np.append(warped_traj_full)

    aligned_trajectories_df = [pd.DataFrame(data=arr, columns=required_columns) for arr in aligned_trajectories_np]
    return aligned_trajectories_df, reference_index

# --- Change Point Detection ---
def detect_change_points(trajectory_data, algorithm, model, n_bkps=None, penalty=None, width=None):
    """
    Detects change points in a trajectory using the specified algorithm.

    Args:
        trajectory_data (np.ndarray): The trajectory data (n_points, n_features).
        algorithm (str): 'Binseg', 'Pelt', or 'Window'.
        model (str): Cost function model (e.g., 'l2').
        n_bkps (int, optional): Number of breakpoints for Binseg. Defaults to None.
        penalty (float, optional): Penalty value for Pelt. Defaults to None.
        width (int, optional): Window width for Window. Defaults to None.

    Returns:
        list: List of change point indices (excluding the end point).
              Returns None if detection fails.
    """
    if trajectory_data.shape[0] < 2: # Need at least 2 points
        print("Warning: Trajectory too short for CPD.")
        return []

    # Initialize the algorithm
    if algorithm == 'Binseg':
        if n_bkps is None:
            print("Error: n_bkps must be specified for Binseg.")
            return None
        algo = rpt.Binseg(model=model)
        param = {'n_bkps': n_bkps}
    elif algorithm == 'Pelt':
        if penalty is None:
            print("Error: penalty must be specified for Pelt.")
            return None
        algo = rpt.Pelt(model=model)
        param = {'pen': penalty}
    elif algorithm == 'Window':
         if width is None or width <= 0:
             print("Error: width must be specified and positive for Window.")
             return None
         algo = rpt.Window(width=width, model=model)
         param = {} # No predict parameter needed for Window fit_predict
    else:
        print(f"Error: Unknown CPD algorithm '{algorithm}'")
        return None

    try:
        # Fit and predict change points
        if algorithm == 'Window':
             # Window based detection might require fit_predict or specific handling
             # For simplicity using Pelt/Binseg predict style
             # This part might need adjustment based on specific Window usage needs
             # Let's use fit().predict() for consistency, assuming it works
             # Note: Window might be better suited for online scenarios
             # Using Pelt as a fallback if Window predict fails easily
             print("Warning: Using Pelt as fallback for Window predict example.")
             algo = rpt.Pelt(model=model)
             param = {'pen': penalty if penalty is not None else 10} # Use default penalty
             result = algo.fit(trajectory_data).predict(**param)

        else:
             result = algo.fit(trajectory_data).predict(**param)

        # result contains the index *after* the change point.
        # The last element is usually the length of the signal.
        # We typically want the indices *before* the change.
        change_points = [idx -1 for idx in result[:-1] if idx > 0] # Exclude end, adjust index
        return sorted(change_points) # Return sorted indices before change

    except Exception as e:
        print(f"Error during change point detection with {algorithm}: {e}")
        return None

# --- Visualization ---
def plot_cpd_segmentation(trajectory_df, change_points, trajectory_filename, algorithm_params_str, feature_cols_to_plot):
    """Plots trajectory components and marks detected change points."""
    trajectory_data = trajectory_df[feature_cols_to_plot].values
    n_points = len(trajectory_data)
    if n_points == 0:
        print(f"Skipping plot for {trajectory_filename} due to empty data.")
        return

    time_steps = np.arange(n_points)
    n_plot_dims = trajectory_data.shape[1]

    fig, axs = plt.subplots(n_plot_dims, 1, sharex=True, figsize=(12, 2.5 * n_plot_dims))
    if n_plot_dims == 1: axs = [axs] # Make indexable
    fig.suptitle(f'CPD Segmentation: {trajectory_filename} ({algorithm_params_str})', fontsize=14)

    sns.set_style("whitegrid")

    for i in range(n_plot_dims):
        axs[i].plot(time_steps, trajectory_data[:, i], label=f'{feature_cols_to_plot[i]}', color='black')
        axs[i].set_ylabel(f'{feature_cols_to_plot[i]}')

        # Mark change points with vertical lines
        if change_points is not None:
            for j, cp in enumerate(change_points):
                label = 'Change Point' if i == 0 and j == 0 else "" # Label only once
                axs[i].axvline(cp, color='red', linestyle='--', linewidth=1.5, label=label)

    axs[-1].set_xlabel('Time Step')
    if change_points is not None and len(change_points) > 0:
         handles, labels = axs[0].get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    original_trajs_df, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH, load_all=LOAD_ALL_FILES, file_list=FILE_LIST
    )
    if not original_trajs_df: exit("Exiting: No trajectories loaded.")

    # 2. Optional DTW Alignment
    if APPLY_DTW_ALIGNMENT:
        try: dtw_col_indices_ignored = [all_load_features.index(col) for col in DTW_FEATURE_COLS]
        except ValueError as e: exit(f"Error: DTW feature column not found: {e}")
        aligned_trajs_df, ref_idx = align_trajectories_dtw(original_trajs_df, DTW_FEATURE_COLS)
        if not aligned_trajs_df: exit("\nDTW alignment failed. Exiting.")
        print(f"\nSuccessfully aligned {len(aligned_trajs_df)} trajectories.")
        trajs_for_cpd = aligned_trajs_df
        time_label = "Time Step (Aligned)"
    else:
        print("\nSkipping DTW alignment.")
        trajs_for_cpd = original_trajs_df
        time_label = "Time Step (Original)"


    # 3. Perform CPD for each trajectory
    print(f"\nPerforming Change Point Detection using {CPD_ALGORITHM}...")
    algo_params_str = f"Alg={CPD_ALGORITHM}, Model={CPD_MODEL}"
    if CPD_ALGORITHM == 'Binseg': algo_params_str += f", n_bkps={N_CHANGEPOINTS_BINSEG}"
    elif CPD_ALGORITHM == 'Pelt': algo_params_str += f", pen={PENALTY_PELT}"
    elif CPD_ALGORITHM == 'Window': algo_params_str += f", width={WINDOW_WIDTH}"


    for i, traj_df in enumerate(trajs_for_cpd):
        filename = loaded_files[i] if i < len(loaded_files) else f"Trajectory {i}"
        print(f"\n--- Processing: {filename} ---")
        traj_cpd_features = traj_df[CPD_FEATURE_COLS].values

        if traj_cpd_features.shape[0] < 2:
            print("Skipping CPD due to short trajectory length.")
            continue

        change_points = detect_change_points(
            traj_cpd_features,
            algorithm=CPD_ALGORITHM,
            model=CPD_MODEL,
            n_bkps=N_CHANGEPOINTS_BINSEG if CPD_ALGORITHM == 'Binseg' else None,
            penalty=PENALTY_PELT if CPD_ALGORITHM == 'Pelt' else None,
            width=WINDOW_WIDTH if CPD_ALGORITHM == 'Window' else None
        )

        if change_points is not None:
            print(f"Detected change points (indices before change): {change_points}")
            # Plot the segmentation
            plot_cpd_segmentation(traj_df, change_points, filename, algo_params_str, POS_COLS) # Plot position dims
        else:
            print("Change point detection failed for this trajectory.")

