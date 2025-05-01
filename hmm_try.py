import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from hmmlearn import hmm

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'
N_TRAJECTORIES = 3      # How many trajectory files to load
N_STATES = 4            # Number of hidden states (phases) for the HMM
INCLUDE_ROTATION = True # <<< Set to True to include rotation matrix (9 elements) in HMM features
N_MIX = 1               # Number of Gaussian mixtures per state (start with 1)
N_ITER_HMM = 50         # Max iterations for HMM training
COVARIANCE_TYPE = 'diag'# 'diag' or 'full'. 'diag' is often better for small datasets
RANDOM_STATE_HMM = 42   # For reproducible results

# Define column names based on whether rotation is included
POS_COLS = ['tx', 'ty', 'tz']
ROT_COLS = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
if INCLUDE_ROTATION:
    FEATURE_COLS = POS_COLS + ROT_COLS
    N_FEATURES = 12
else:
    FEATURE_COLS = POS_COLS
    N_FEATURES = 3

print(f"Using {N_FEATURES} features: {FEATURE_COLS}")

# --- Data Loading and Processing ---

def load_and_process_data(parent_folder, n_trajectories, feature_columns):
    """Loads trajectories from CSV files using specified feature columns."""
    all_trajectories_features = []
    trajectory_lengths = []

    print(f"Loading {n_trajectories} trajectories...")
    for i in range(1, n_trajectories + 1):
        # Assuming file names are 1.csv, 2.csv, ...
        # Adjust if file naming is different (e.g., requires leading zeros)
        file_path = os.path.join(parent_folder, f"{i}.csv")
        print(f"  Loading {file_path}...")
        try:
            df = pd.read_csv(file_path)
            # Select only the specified feature columns
            trajectory_data = df[feature_columns].values
            if trajectory_data.shape[1] != len(feature_columns):
                 raise ValueError(f"Expected {len(feature_columns)} feature columns, found {trajectory_data.shape[1]}")
            if np.isnan(trajectory_data).any():
                print(f"Warning: NaN values found in {file_path}. Attempting to handle.")
                # Option 1: Forward fill NaNs
                # trajectory_data = pd.DataFrame(trajectory_data).fillna(method='ffill').fillna(method='bfill').values
                # Option 2: Remove rows with NaNs (might shorten sequence)
                nan_rows = np.isnan(trajectory_data).any(axis=1)
                if np.all(nan_rows):
                    print(f"Warning: All rows in {file_path} contain NaN after selection. Skipping trajectory.")
                    continue
                trajectory_data = trajectory_data[~nan_rows]
                if trajectory_data.shape[0] == 0:
                    print(f"Warning: Trajectory {i} became empty after removing NaN rows. Skipping.")
                    continue


            all_trajectories_features.append(trajectory_data)
            trajectory_lengths.append(len(trajectory_data))

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except KeyError as e:
            print(f"Error: Missing expected column in {file_path}: {e}")
            print(f"  Expected columns: {feature_columns}")
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories_features:
         raise RuntimeError("Could not load or process any valid trajectory data.")

    # Concatenate data for HMM training
    hmm_data = np.concatenate(all_trajectories_features)

    return all_trajectories_features, hmm_data, trajectory_lengths

# --- HMM Training ---

def train_hmm(data, lengths, n_states, n_mix, n_iter, cov_type, random_state):
    """Trains a GMM-HMM model."""
    print(f"\nTraining GMM-HMM with {n_states} states and {n_mix} mixture components...")
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=cov_type,
                       n_iter=n_iter,
                       random_state=random_state,
                       verbose=True,
                       params='stmcw', # Train states, transitions, means, covars, weights
                       init_params='stmcw') # Initialize all params

    try:
        model.fit(data, lengths=lengths)
        print("HMM Training complete.")
        if hasattr(model, 'monitor_') and not model.monitor_.converged:
            print("Warning: HMM training did not converge within max iterations.")
        return model
    except ValueError as e:
        print(f"Error during HMM fitting: {e}")
        print("This might happen if sequences are too short or data has issues (e.g., variance is zero in some dimension).")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during HMM fitting: {e}")
        return None

# --- Segmentation and Plotting ---

def plot_segmentation(trajectory_data_pos, state_sequence, trajectory_index, n_states):
    """
    Plots the trajectory position components (X, Y, Z) and colors
    the background by predicted state.
    Assumes trajectory_data_pos contains only the X, Y, Z columns.
    """
    n_points = len(trajectory_data_pos)
    if n_points == 0:
        print(f"Skipping plotting for empty trajectory {trajectory_index + 1}")
        return
    if n_points != len(state_sequence):
        print(f"Warning: Data length ({n_points}) and state sequence length ({len(state_sequence)}) mismatch for trajectory {trajectory_index + 1}. Skipping plot.")
        return

    time_steps = np.arange(n_points)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig.suptitle(f'Trajectory {trajectory_index + 1} - HMM Segmentation ({n_states} States)')

    # Use a consistent color map
    colors = plt.cm.viridis(np.linspace(0, 1, n_states))

    # Plot X, Y, Z positions
    coord_labels = ['X', 'Y', 'Z']
    plotted_state_labels = set() # Track labels to avoid duplicates in legend

    for i in range(3): # Iterate through X, Y, Z
        axs[i].plot(time_steps, trajectory_data_pos[:, i], label=f'{coord_labels[i]}-Position', color='black', zorder=5)
        axs[i].set_ylabel(f'{coord_labels[i]} (m)')

        # Color background based on predicted state
        for state in range(n_states):
            # Find contiguous segments for the current state
            state_indices = np.where(state_sequence == state)[0]
            if len(state_indices) == 0: continue # Skip if state not present

            # Find changes to identify segments
            diff = np.diff(state_indices)
            change_points = np.where(diff != 1)[0]
            segment_starts = np.insert(state_indices[change_points + 1], 0, state_indices[0])
            segment_ends = np.append(state_indices[change_points], state_indices[-1])

            for start, end in zip(segment_starts, segment_ends):
                 label = f'State {state}' if state not in plotted_state_labels else ""
                 axs[i].axvspan(max(0, start - 0.5), min(n_points - 1, end + 0.5),
                                facecolor=colors[state], alpha=0.3, label=label, zorder=1)
                 if label: plotted_state_labels.add(state)


    axs[2].set_xlabel('Time Step')
    # Create a single legend for the figure
    handles, labels = axs[0].get_legend_handles_labels()
    # Filter unique labels for the legend
    by_label = dict(zip(labels, handles)) # Use dict to automatically handle uniqueness
    fig.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load data
    try:
        all_trajectories, concatenated_data, traj_lengths = load_and_process_data(
            PARENT_FOLDER_PATH, N_TRAJECTORIES, FEATURE_COLS
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Failed to load data: {e}")
        exit()

    # Check data validity
    if concatenated_data.shape[0] == 0 or concatenated_data.shape[1] != N_FEATURES:
         print(f"Error: Processed data has unexpected shape {concatenated_data.shape}. Expected (n_points, {N_FEATURES}). Cannot train HMM.")
         exit()
    if np.isnan(concatenated_data).any():
        print("Error: Concatenated data contains NaN values after processing. Cannot train HMM.")
        exit()
    if not traj_lengths or sum(traj_lengths) != concatenated_data.shape[0]:
        print("Error: Trajectory lengths do not match concatenated data size.")
        exit()

    # 2. Train the HMM
    hmm_model = train_hmm(concatenated_data, traj_lengths, N_STATES, N_MIX, N_ITER_HMM, COVARIANCE_TYPE, RANDOM_STATE_HMM)

    # 3. Predict states and plot results for each trajectory
    if hmm_model:
        print("\nPredicting states and plotting results...")
        for i, traj_features in enumerate(all_trajectories):
            if len(traj_features) > 0: # Ensure trajectory is not empty
                try:
                    # Predict the most likely sequence of states (Viterbi algorithm)
                    predicted_states = hmm_model.predict(traj_features)

                    # Extract only position data (first 3 columns) for plotting
                    traj_pos = traj_features[:, :3]

                    plot_segmentation(traj_pos, predicted_states, i, N_STATES)
                except Exception as e:
                    print(f"Error predicting or plotting for trajectory {i+1}: {e}")
            else:
                print(f"Skipping empty trajectory {i+1}")
    else:
        print("HMM model training failed. Cannot proceed with prediction and plotting.")

