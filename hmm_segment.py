import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from hmmlearn import hmm
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
APPLY_DTW_ALIGNMENT = True # Apply DTW before HMM?

# --- Feature Config ---
# Features for DTW distance calculation (if APPLY_DTW_ALIGNMENT is True)
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to use for HMM training
HMM_POS_COLS = ['tx', 'ty', 'tz']
INCLUDE_ROTATION_HMM = False
HMM_ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']

# --- HMM Config ---
N_STATES = 4            # Number of hidden states (segments) for the HMM
N_MIX = 1               # Number of Gaussian mixtures per state
N_ITER_HMM = 50         # Max iterations for HMM training
COVARIANCE_TYPE = 'diag'# 'diag' or 'full'.
RANDOM_STATE_HMM = 42   # For reproducible results

# --- Determine HMM Features ---
# Define base column lists globally for clarity
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
all_load_features = POS_COLS + ROT_COLS # All columns potentially needed

HMM_FEATURE_COLS = HMM_POS_COLS[:] # Start with position
if INCLUDE_ROTATION_HMM:
    HMM_FEATURE_COLS.extend(HMM_ROT_COLS)
N_FEATURES_HMM = len(HMM_FEATURE_COLS)

print(f"HMM using {N_FEATURES_HMM} features: {HMM_FEATURE_COLS}")
if APPLY_DTW_ALIGNMENT:
    print(f"Using features for DTW: {DTW_FEATURE_COLS}")


# --- Data Loading ---
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

            # Store data as DataFrame to easily select columns later
            all_trajectories.append(df[required_columns])
            loaded_filenames.append(filename)
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
        except KeyError as e: print(f"Error in {file_path}: {e}")
        except Exception as e: print(f"Error loading/processing {file_path}: {e}")

    if not all_trajectories: print("Warning: No valid trajectory data loaded.")
    return all_trajectories, loaded_filenames # Return list of DataFrames

# --- DTW Alignment ---
def align_trajectories_dtw(trajectories_df_list, dtw_feature_cols):
    """Aligns trajectories using DTW based on specified features."""
    if not trajectories_df_list: return [], None, -1
    # Convert DFs to numpy arrays for DTW features
    trajectories_dtw_np = [df[dtw_feature_cols].values for df in trajectories_df_list]
    # Keep original full DFs
    trajectories_full_np = [df.values for df in trajectories_df_list]
    required_columns = trajectories_df_list[0].columns.tolist() # Get col order

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
                warped_traj_full[ref_idx] = traj_full[traj_idx] # Warp the full data
            else:
                if ref_idx > 0: warped_traj_full[ref_idx] = warped_traj_full[ref_idx - 1]
                elif len(traj_full) > 0: warped_traj_full[ref_idx] = traj_full[0]
        aligned_trajectories_np.append(warped_traj_full)

    # Convert back to DataFrames
    aligned_trajectories_df = [pd.DataFrame(data=arr, columns=required_columns) for arr in aligned_trajectories_np]

    return aligned_trajectories_df, reference_index

# --- HMM Training ---
def train_hmm(data, n_states, n_mix, n_iter, cov_type, random_state, lengths=None):
    """Trains a GMM-HMM model."""
    print(f"\nTraining GMM-HMM with {n_states} states and {n_mix} mixture components...")
    model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type=cov_type,
                       n_iter=n_iter, tol=1e-3, random_state=random_state, verbose=True,
                       params='stmcw', init_params='stmcw')
    try:
        # Pass lengths only if DTW was NOT applied
        if lengths:
            model.fit(data, lengths=lengths)
        else:
            model.fit(data) # Assumes all sequences are concatenated and have same length
        print("HMM Training complete.")
        if hasattr(model, 'monitor_') and not model.monitor_.converged:
            print("Warning: HMM training did not converge.")
        return model
    except ValueError as e: print(f"Error during HMM fitting: {e}"); return None
    except Exception as e: print(f"Unexpected error during HMM fitting: {e}"); return None

# --- Visualization ---
def plot_segmentation(trajectory_df, state_sequence, trajectory_filename, n_states, feature_cols_to_plot):
    """Plots trajectory components and colors the background by state."""
    trajectory_data = trajectory_df[feature_cols_to_plot].values
    n_points = len(trajectory_data)
    if n_points == 0 or n_points != len(state_sequence):
        print(f"Skipping plot for {trajectory_filename} due to length mismatch or empty data.")
        return

    time_steps = np.arange(n_points)
    n_plot_dims = trajectory_data.shape[1]

    fig, axs = plt.subplots(n_plot_dims, 1, sharex=True, figsize=(12, 2.5 * n_plot_dims))
    if n_plot_dims == 1: axs = [axs] # Make indexable
    fig.suptitle(f'HMM Segmentation: {trajectory_filename} ({n_states} States)', fontsize=14)

    colors = sns.color_palette("viridis", n_states)
    plotted_state_labels = set()

    for i in range(n_plot_dims):
        axs[i].plot(time_steps, trajectory_data[:, i], label=f'{feature_cols_to_plot[i]}', color='black', zorder=5)
        axs[i].set_ylabel(f'{feature_cols_to_plot[i]}')

        for state in range(n_states):
            state_indices = np.where(state_sequence == state)[0]
            if len(state_indices) == 0: continue
            diff = np.diff(state_indices)
            change_points = np.where(diff != 1)[0]
            segment_starts = np.insert(state_indices[change_points + 1], 0, state_indices[0])
            segment_ends = np.append(state_indices[change_points], state_indices[-1])

            for start, end in zip(segment_starts, segment_ends):
                 label = f'State {state}' if state not in plotted_state_labels else ""
                 axs[i].axvspan(max(0, start - 0.5), min(n_points - 1, end + 0.5),
                                facecolor=colors[state], alpha=0.3, label=label, zorder=1)
                 if label: plotted_state_labels.add(state)

    axs[-1].set_xlabel('Time Step')
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
        try: dtw_col_indices_ignored = [all_load_features.index(col) for col in DTW_FEATURE_COLS] # Get indices, not used after this
        except ValueError as e: exit(f"Error: DTW feature column not found: {e}")
        # Align using DFs, returns list of DFs
        aligned_trajs_df, ref_idx = align_trajectories_dtw(original_trajs_df, DTW_FEATURE_COLS)
        if not aligned_trajs_df: exit("\nDTW alignment failed. Exiting.")
        print(f"\nSuccessfully aligned {len(aligned_trajs_df)} trajectories.")
        trajs_for_hmm = aligned_trajs_df
        hmm_lengths = None # No lengths needed for fit if aligned
    else:
        print("\nSkipping DTW alignment.")
        trajs_for_hmm = original_trajs_df
        hmm_lengths = [len(df) for df in trajs_for_hmm] # Need lengths if not aligned

    # 3. Prepare data for HMM (select features and concatenate)
    try:
        hmm_input_data_list = [df[HMM_FEATURE_COLS].values for df in trajs_for_hmm]
        concatenated_hmm_data = np.concatenate(hmm_input_data_list)
    except KeyError as e:
        exit(f"Error: HMM feature column not found in data: {e}")
    except Exception as e:
        exit(f"Error preparing HMM data: {e}")

    # Check HMM data validity
    if concatenated_hmm_data.shape[0] == 0 or concatenated_hmm_data.shape[1] != N_FEATURES_HMM:
         exit(f"Error: HMM input data has unexpected shape {concatenated_hmm_data.shape}.")
    if np.isnan(concatenated_hmm_data).any():
        exit("Error: HMM input data contains NaN values.")

    # 4. Train the HMM
    hmm_model = train_hmm(concatenated_hmm_data, N_STATES, N_MIX, N_ITER_HMM,
                          COVARIANCE_TYPE, RANDOM_STATE_HMM, lengths=hmm_lengths)

    # 5. Predict states (segment) and plot for each trajectory
    if hmm_model:
        print("\nPredicting states (segmenting) and plotting results...")
        start_idx = 0
        for i, traj_df in enumerate(trajs_for_hmm):
            traj_hmm_features = traj_df[HMM_FEATURE_COLS].values
            traj_len = len(traj_hmm_features)
            if traj_len == 0:
                print(f"Skipping prediction for empty trajectory {i} ({loaded_files[i]}).")
                continue

            try:
                # Predict sequence for the individual trajectory
                # Note: predict expects a single sequence, no lengths needed here
                predicted_states = hmm_model.predict(traj_hmm_features)

                # --- Output Segmentation ---
                print(f"\n--- Trajectory: {loaded_files[i]} ---")
                print(f"Predicted State Sequence (length {len(predicted_states)}):")
                print(predicted_states)
                # You could save this sequence to a file or use it directly

                # Plot the segmentation
                plot_segmentation(traj_df, predicted_states, loaded_files[i], N_STATES, POS_COLS) # Plot position dims

            except Exception as e:
                print(f"Error predicting or plotting for trajectory {i} ({loaded_files[i]}): {e}")
                # import traceback
                # traceback.print_exc()

    else:
        print("HMM model training failed.")

