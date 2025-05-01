import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For legend handles if needed
import math
import os
from hmmlearn import hmm
from scipy import linalg # Potentially needed for covariance checks
import seaborn as sns # For improved aesthetics and kdeplot
# Required for DTW
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'

# --- File Loading Config ---
# Option 1: Load all CSV files in the folder
LOAD_ALL_FILES = True
# Option 2: Specify a list of filenames (if LOAD_ALL_FILES is False)
# FILE_LIST = ['1.csv', '3.csv', '5.csv']
FILE_LIST = [] # Ignored if LOAD_ALL_FILES is True

# --- Feature Config ---
# Features to use for DTW distance calculation
DTW_FEATURE_COLS = ['tx', 'ty', 'tz']
# Features to use for HMM training (can be same or different)
INCLUDE_ROTATION_HMM = False # <<< Set to True to include rotation matrix (9 elements) in HMM features

POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
if INCLUDE_ROTATION_HMM:
    HMM_FEATURE_COLS = POS_COLS + ROT_COLS
    N_FEATURES_HMM = 12
else:
    HMM_FEATURE_COLS = POS_COLS
    N_FEATURES_HMM = 3

print(f"Using features for DTW: {DTW_FEATURE_COLS}")
print(f"Using {N_FEATURES_HMM} features for HMM: {HMM_FEATURE_COLS}")


# --- HMM Config ---
N_STATES = 4            # Number of hidden states (phases) for the HMM
N_MIX = 1               # Number of Gaussian mixtures per state (start with 1)
N_ITER_HMM = 50         # Max iterations for HMM training
COVARIANCE_TYPE = 'diag'# 'diag' or 'full'. 'diag' is often better for small datasets
RANDOM_STATE_HMM = 42   # For reproducible results


# --- Data Loading (from DTW script) ---

def load_selected_data(parent_folder, load_all=True, file_list=None, feature_columns=None):
    """Loads specified trajectories and features from CSV files."""
    all_trajectories = []
    loaded_filenames = []
    required_columns = POS_COLS + ROT_COLS # Need all pose columns initially

    if feature_columns is None:
        feature_columns_dtw = ['tx', 'ty', 'tz'] # Default DTW features
    else:
        feature_columns_dtw = feature_columns

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            try:
                 filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                 filenames.sort()
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
            # Check if all required columns exist before selecting
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                 raise KeyError(f"Missing columns: {missing_cols}")

            # Store the full data (pos+rot) first
            full_trajectory_data = df[required_columns].values

            # Basic NaN check
            if np.isnan(full_trajectory_data).any():
                print(f"Warning: NaN values found in {file_path}. Skipping file.")
                continue
            if full_trajectory_data.shape[0] == 0:
                 print(f"Warning: Trajectory data is empty in {file_path}. Skipping file.")
                 continue

            # Append the full data (pos+rot)
            all_trajectories.append(full_trajectory_data)
            loaded_filenames.append(filename)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except KeyError as e:
            print(f"Error in {file_path}: {e}")
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories:
         print("Warning: No valid trajectory data loaded.")

    return all_trajectories, loaded_filenames

# --- DTW Alignment (from DTW script) ---

def align_trajectories_dtw(trajectories, dtw_feature_indices):
    """
    Aligns all trajectories to the longest one using DTW based on specified features.

    Args:
        trajectories (list): List of numpy arrays (n_points, n_full_features).
                             Each array contains ALL features (e.g., pos+rot).
        dtw_feature_indices (list): Indices of columns to use for DTW distance calculation.

    Returns:
        tuple: (aligned_trajectories, reference_trajectory, reference_index)
               aligned_trajectories: List of warped trajectories (containing ALL features).
               reference_trajectory: The longest trajectory (containing ALL features).
               reference_index: The index of the reference trajectory.
    """
    if not trajectories:
        return [], None, -1

    lengths = [len(t) for t in trajectories]
    if not lengths:
         return [], None, -1
    reference_index = np.argmax(lengths)
    reference_trajectory_full = trajectories[reference_index]
    ref_len = len(reference_trajectory_full)
    print(f"\nUsing trajectory {reference_index} (length {ref_len}) as reference for DTW.")

    aligned_trajectories = []

    for i, traj_full in enumerate(trajectories):
        print(f"  Aligning trajectory {i} (length {len(traj_full)}) to reference...")
        if len(traj_full) == 0:
             print(f"    Skipping empty trajectory {i}")
             aligned_trajectories.append(np.zeros_like(reference_trajectory_full))
             continue

        if i == reference_index:
            aligned_trajectories.append(reference_trajectory_full.copy())
            continue

        # Extract features specifically for DTW distance calculation
        traj_dtw_features = traj_full[:, dtw_feature_indices]
        ref_dtw_features = reference_trajectory_full[:, dtw_feature_indices]

        # Perform DTW using fastdtw on the selected features
        distance, path = fastdtw(ref_dtw_features, traj_dtw_features, dist=euclidean)
        print(f"    DTW distance: {distance:.4f}")

        # Warp the *full* trajectory based on the DTW path
        path_array = np.array(path)
        # Initialize warped trajectory to hold ALL features
        warped_traj_full = np.zeros_like(reference_trajectory_full)

        ref_indices_in_path = path_array[:, 0]
        traj_indices_in_path = path_array[:, 1]

        for ref_idx in range(ref_len):
            matching_path_indices = np.where(ref_indices_in_path == ref_idx)[0]
            if len(matching_path_indices) > 0:
                traj_idx = traj_indices_in_path[matching_path_indices[-1]]
                traj_idx = min(traj_idx, len(traj_full) - 1)
                # Copy the full feature vector from the original trajectory
                warped_traj_full[ref_idx] = traj_full[traj_idx]
            else:
                print(f"Warning: No matching path point found for reference index {ref_idx} while warping trajectory {i}. Repeating previous point.")
                if ref_idx > 0:
                    warped_traj_full[ref_idx] = warped_traj_full[ref_idx - 1]
                elif len(traj_full) > 0:
                     warped_traj_full[ref_idx] = traj_full[0]

        aligned_trajectories.append(warped_traj_full)

    return aligned_trajectories, reference_trajectory_full, reference_index


# --- HMM Training ---

def train_hmm(data, n_states, n_mix, n_iter, cov_type, random_state):
    """
    Trains a GMM-HMM model on concatenated data.
    NOTE: Assumes all sequences in 'data' have the same length (post-DTW).
    """
    print(f"\nTraining GMM-HMM with {n_states} states and {n_mix} mixture components...")
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=cov_type,
                       n_iter=n_iter,
                       tol=1e-3,
                       random_state=random_state,
                       verbose=True,
                       params='stmcw',
                       init_params='stmcw')
    try:
        # Fit the model - NO 'lengths' parameter needed after DTW alignment
        model.fit(data)
        print("HMM Training complete.")
        if hasattr(model, 'monitor_') and not model.monitor_.converged:
            print("Warning: HMM training did not converge within max iterations.")
        return model
    except ValueError as e:
        print(f"Error during HMM fitting: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during HMM fitting: {e}")
        return None

# --- GMM Contour Visualization ---

def plot_gmm_contours(ax, data, labels, plot_dims, model, colors, title):
    """
    Plots data points (position only) colored by state and overlays GMM contours
    for ALL states using seaborn's kdeplot.
    """
    dim1, dim2 = plot_dims
    dim_labels = ['tx', 'ty', 'tz'] # Use consistent labels
    pos_data = data[:, :3] # Extract X, Y, Z for plotting

    sns.set_style("whitegrid")
    # palette = colors # Use the passed palette

    plot_handles = []
    plot_labels = []

    for state in range(model.n_components):
        state_indices = np.where(labels == state)[0]
        if len(state_indices) > 0:
            state_pos_data = pos_data[state_indices]
            scatter = ax.scatter(state_pos_data[:, dim1], state_pos_data[:, dim2], s=15, alpha=0.5,
                       color=colors[state], label=f'State {state} Data')
            if f'State {state} Data' not in plot_labels:
                plot_handles.append(scatter)
                plot_labels.append(f'State {state} Data')

            try:
                if len(state_indices) > 5 and np.linalg.matrix_rank(np.cov(state_pos_data[:, plot_dims].T)) == 2:
                     sns.kdeplot(
                         x=state_pos_data[:, dim1],
                         y=state_pos_data[:, dim2],
                         ax=ax,
                         color=colors[state],
                         levels=4,
                         linewidths=1.5,
                     )
                     contour_proxy = plt.Line2D([0], [0], linestyle='-', color=colors[state], lw=1.5, label=f'State {state} Density')
                     if f'State {state} Density' not in plot_labels:
                          plot_handles.append(contour_proxy)
                          plot_labels.append(f'State {state} Density')
                else:
                     print(f"Skipping KDE plot for State {state}: Insufficient points or variance in dimensions {plot_dims}.")
                     if state < model.n_components and model.n_mix > 0:
                          mean_3d = model.means_[state, 0, :3]
                          mean_plot = ax.scatter(mean_3d[dim1], mean_3d[dim2], marker='X', s=100, color=colors[state], edgecolor='black', zorder=11, label=f'State {state} Mean')
                          if f'State {state} Mean' not in plot_labels:
                             plot_handles.append(mean_plot)
                             plot_labels.append(f'State {state} Mean')
            except Exception as e:
                print(f"Error plotting KDE for State {state}: {e}")
                if state < model.n_components and model.n_mix > 0:
                     mean_3d = model.means_[state, 0, :3]
                     mean_plot = ax.scatter(mean_3d[dim1], mean_3d[dim2], marker='X', s=100, color=colors[state], edgecolor='black', zorder=11, label=f'State {state} Mean')
                     if f'State {state} Mean' not in plot_labels:
                         plot_handles.append(mean_plot)
                         plot_labels.append(f'State {state} Mean')

    ax.set_xlabel(f'{dim_labels[dim1]}')
    ax.set_ylabel(f'{dim_labels[dim2]}')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    return plot_handles, plot_labels

# --- State vs Aligned Time Plotting ---
def plot_state_vs_time(ax, all_labels, ref_len, n_states, colors):
    """
    Plots the predicted state against the aligned time steps.
    Args:
        ax: Matplotlib axes object.
        all_labels: Array of predicted state labels for the concatenated aligned data.
        ref_len: The length of the reference trajectory (and all aligned trajectories).
        n_states: Total number of states.
        colors: List of colors for states (assumes seaborn palette).
    """
    n_aligned_traj = len(all_labels) // ref_len
    aligned_time = np.arange(ref_len)

    # Reshape labels to (n_traj, ref_len) if needed or plot directly
    # Plotting directly might be easier if number of trajectories is large

    # Add small jitter to states for better visualization
    state_jitter = all_labels + np.random.normal(0, 0.08, size=all_labels.shape)
    # Create time vector that repeats for each trajectory
    time_repeated = np.tile(aligned_time, n_aligned_traj)

    # Plot using scatter plot
    for state in range(n_states):
        state_mask = (all_labels == state)
        if np.any(state_mask):
            ax.scatter(time_repeated[state_mask], state_jitter[state_mask],
                       color=colors[state], alpha=0.3, s=5, label=f'State {state}') # Smaller points

    ax.set_xlabel("Time Step (Aligned)")
    ax.set_ylabel("Predicted State Index")
    ax.set_title("State Occurrence vs. Aligned Trajectory Time")
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([str(i) for i in range(n_states)])
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3) # Increase legend marker size


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data (loading all features initially)
    all_features = POS_COLS + ROT_COLS # Ensure we load everything needed
    original_trajs_full, loaded_files = load_selected_data(
        PARENT_FOLDER_PATH,
        load_all=LOAD_ALL_FILES,
        file_list=FILE_LIST,
        feature_columns=all_features
    )

    if not original_trajs_full:
        print("Exiting: No trajectories loaded.")
        exit()

    # 2. Perform DTW Alignment using specified DTW features
    # Find indices for DTW features within the loaded full data
    try:
        dtw_col_indices = [all_features.index(col) for col in DTW_FEATURE_COLS]
    except ValueError as e:
        print(f"Error: One of the DTW feature columns not found in loaded data: {e}")
        exit()

    aligned_trajs_full, ref_traj_full, ref_idx = align_trajectories_dtw(
        original_trajs_full, dtw_col_indices
    )

    # Filter out potentially empty/zero arrays added during alignment
    valid_aligned_trajs_full = [t for t in aligned_trajs_full if len(t) > 0 and np.any(t)]

    if not valid_aligned_trajs_full:
        print("\nNo valid aligned trajectories available after DTW. Exiting.")
        exit()
    else:
         print(f"\nSuccessfully aligned {len(valid_aligned_trajs_full)} trajectories.")

    # 3. Prepare data for HMM (select HMM features)
    # Find indices for HMM features within the loaded full data
    try:
        hmm_col_indices = [all_features.index(col) for col in HMM_FEATURE_COLS]
    except ValueError as e:
        print(f"Error: One of the HMM feature columns not found in loaded data: {e}")
        exit()

    # Select HMM features from the aligned trajectories
    hmm_input_trajs = [traj[:, hmm_col_indices] for traj in valid_aligned_trajs_full]
    concatenated_hmm_data = np.concatenate(hmm_input_trajs)

    # Check HMM data validity
    if concatenated_hmm_data.shape[0] == 0 or concatenated_hmm_data.shape[1] != N_FEATURES_HMM:
         print(f"Error: HMM input data has unexpected shape {concatenated_hmm_data.shape}. Expected (n_points, {N_FEATURES_HMM}).")
         exit()
    if np.isnan(concatenated_hmm_data).any():
        print("Error: HMM input data contains NaN values after processing.")
        exit()

    # 4. Train the HMM on aligned data (NO lengths needed)
    hmm_model = train_hmm(concatenated_hmm_data, N_STATES, N_MIX, N_ITER_HMM, COVARIANCE_TYPE, RANDOM_STATE_HMM)

    # 5. Predict states for all aligned data points
    if hmm_model:
        try:
            # Predict states for the entire concatenated aligned dataset
            # NO lengths parameter needed here either
            all_predicted_states = hmm_model.predict(concatenated_hmm_data)
            print("\nPredicted states for all aligned data points.")

            # --- Visualization ---
            sns.set_theme(style="whitegrid")
            state_colors = sns.color_palette("viridis", N_STATES)

            # 6. Visualize GMM Contours
            print("Generating GMM contour visualizations...")
            fig_gmm, axs_gmm = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
            fig_gmm.suptitle(f'GMM State Distributions (Post-DTW, {N_STATES} States, Features: {N_FEATURES_HMM})', fontsize=16)

            # Note: plot_gmm_contours plots based on first 3 dims (pos) of HMM data
            h1, l1 = plot_gmm_contours(axs_gmm[0], concatenated_hmm_data, all_predicted_states, [0, 1], hmm_model, state_colors, "XY Projection")
            h2, l2 = plot_gmm_contours(axs_gmm[1], concatenated_hmm_data, all_predicted_states, [0, 2], hmm_model, state_colors, "XZ Projection")
            h3, l3 = plot_gmm_contours(axs_gmm[2], concatenated_hmm_data, all_predicted_states, [1, 2], hmm_model, state_colors, "YZ Projection")

            all_handles = h1 + h2 + h3
            all_labels = l1 + l2 + l3
            by_label = dict(zip(all_labels, all_handles))
            fig_gmm.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize='medium')
            plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
            plt.show()

            # 7. Visualize State vs Aligned Time
            print("Generating State vs Aligned Time plot...")
            fig_time, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            ref_len = len(valid_aligned_trajs_full[0]) # Get length from first valid aligned traj
            plot_state_vs_time(ax_time, all_predicted_states, ref_len, N_STATES, state_colors)
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
            plt.show()


            # Print learned parameters (optional)
            print("\nLearned HMM Parameters (Post-DTW):")
            print(f"Features used for HMM: {HMM_FEATURE_COLS}")
            print("Initial Probabilities (pi):")
            print(hmm_model.startprob_)
            print("\nTransition Matrix (A):")
            print(np.round(hmm_model.transmat_, 3))
            print("\nGMM Weights per State:")
            print(np.round(hmm_model.weights_, 3))
            print("\nGMM Means per State (showing first 3 dims - Pos):")
            # Ensure means have enough dimensions before slicing
            if hmm_model.means_.shape[-1] >= 3:
                 print(np.round(hmm_model.means_[:, :, :3], 3))
            else:
                 print(np.round(hmm_model.means_, 3)) # Print all if fewer than 3 features

        except Exception as e:
            print(f"Error during prediction or plotting: {e}")
            # import traceback
            # traceback.print_exc()

    else:
        print("HMM model training failed. Cannot visualize GMMs.")

