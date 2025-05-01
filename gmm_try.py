import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For legend handles if needed
import math
import os
from hmmlearn import hmm
from scipy import linalg # Potentially needed for covariance checks
import seaborn as sns # For improved aesthetics and kdeplot

# --- Configuration ---
# !!! Path to the folder containing 1.csv, 2.csv, etc. !!!
PARENT_FOLDER_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/transformed/button'
N_TRAJECTORIES = 9      # How many trajectory files to load
N_STATES = 4            # Number of hidden states (phases) for the HMM
INCLUDE_ROTATION = False # <<< Set to True to include rotation matrix (9 elements) in HMM features
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
        file_path = os.path.join(parent_folder, f"{i}.csv")
        print(f"  Loading {file_path}...")
        try:
            df = pd.read_csv(file_path)
            trajectory_data = df[feature_columns].values
            if trajectory_data.shape[1] != len(feature_columns):
                 raise ValueError(f"Expected {len(feature_columns)} feature columns, found {trajectory_data.shape[1]}")
            if np.isnan(trajectory_data).any():
                print(f"Warning: NaN values found in {file_path}. Attempting to handle.")
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
    hmm_data = np.concatenate(all_trajectories_features)
    return all_trajectories_features, hmm_data, trajectory_lengths

# --- HMM Training ---

def train_hmm(data, lengths, n_states, n_mix, n_iter, cov_type, random_state):
    """Trains a GMM-HMM model."""
    print(f"\nTraining GMM-HMM with {n_states} states and {n_mix} mixture components...")
    # Increase tolerance slightly for potentially complex data
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=cov_type,
                       n_iter=n_iter,
                       tol=1e-3, # Slightly larger tolerance
                       random_state=random_state,
                       verbose=True,
                       params='stmcw',
                       init_params='stmcw')
    try:
        model.fit(data, lengths=lengths)
        print("HMM Training complete.")
        if hasattr(model, 'monitor_') and not model.monitor_.converged:
            print("Warning: HMM training did not converge within max iterations.")
        return model
    except ValueError as e:
        print(f"Error during HMM fitting: {e}")
        print("This might happen if sequences are too short or data has issues (e.g., singular covariance). Try increasing data or reducing N_MIX/N_STATES.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during HMM fitting: {e}")
        return None

# --- GMM Contour Visualization ---

def plot_gmm_contours(ax, data, labels, plot_dims, model, colors, title):
    """
    Plots data points (position only) colored by state and overlays GMM contours
    for ALL states using seaborn's kdeplot.
    Args:
        ax: Matplotlib axes object.
        data: The full concatenated data (can be 3D or 12D).
        labels: Predicted state labels for each point in data.
        plot_dims: A list/tuple of 2 indices indicating which dimensions to plot (e.g., [0, 1] for XY).
        model: The trained HMM model.
        colors: List of colors for states (assumes seaborn palette).
        title: Title for the subplot.
    """
    dim1, dim2 = plot_dims # Indices for dimensions (0=X, 1=Y, 2=Z)
    dim_labels = ['X', 'Y', 'Z'] # Assuming first 3 dims are always position
    pos_data = data[:, :3] # Extract X, Y, Z for plotting

    # Use seaborn styles for better aesthetics
    sns.set_style("whitegrid")
    # palette = sns.color_palette("viridis", model.n_components) # Use seaborn palette passed via 'colors'

    plot_handles = []
    plot_labels = []

    # Plot data points for each state
    for state in range(model.n_components):
        state_indices = np.where(labels == state)[0]
        if len(state_indices) > 0:
            state_pos_data = pos_data[state_indices]
            scatter = ax.scatter(state_pos_data[:, dim1], state_pos_data[:, dim2], s=15, alpha=0.5,
                       color=colors[state], label=f'State {state} Data')
            # Only add handle/label once per state for the scatter plot
            if f'State {state} Data' not in plot_labels:
                plot_handles.append(scatter)
                plot_labels.append(f'State {state} Data')


            # --- Plot KDE Contours for this state ---
            try:
                # Check for sufficient points and variance
                if len(state_indices) > 5 and np.linalg.matrix_rank(np.cov(state_pos_data[:, plot_dims].T)) == 2:
                     sns.kdeplot(
                         x=state_pos_data[:, dim1],
                         y=state_pos_data[:, dim2],
                         ax=ax,
                         color=colors[state], # Use state color
                         levels=4, # Number of contour levels
                         linewidths=1.5,
                     )
                     # Add a proxy artist for the contour legend entry
                     contour_proxy = plt.Line2D([0], [0], linestyle='-', color=colors[state], lw=1.5, label=f'State {state} Density')
                     if f'State {state} Density' not in plot_labels:
                          plot_handles.append(contour_proxy)
                          plot_labels.append(f'State {state} Density')
                else:
                     print(f"Skipping KDE plot for State {state}: Insufficient points or variance in dimensions {plot_dims}.")
                     # Optionally plot the mean if KDE fails
                     if state < model.n_components and model.n_mix > 0:
                          mean_3d = model.means_[state, 0, :3] # Mean of first component
                          mean_plot = ax.scatter(mean_3d[dim1], mean_3d[dim2], marker='X', s=100, color=colors[state], edgecolor='black', zorder=11, label=f'State {state} Mean')
                          if f'State {state} Mean' not in plot_labels:
                             plot_handles.append(mean_plot)
                             plot_labels.append(f'State {state} Mean')
            except Exception as e:
                print(f"Error plotting KDE for State {state}: {e}")
                # Optionally plot the mean if KDE fails
                if state < model.n_components and model.n_mix > 0:
                     mean_3d = model.means_[state, 0, :3] # Mean of first component
                     mean_plot = ax.scatter(mean_3d[dim1], mean_3d[dim2], marker='X', s=100, color=colors[state], edgecolor='black', zorder=11, label=f'State {state} Mean')
                     if f'State {state} Mean' not in plot_labels:
                         plot_handles.append(mean_plot)
                         plot_labels.append(f'State {state} Mean')

    ax.set_xlabel(f'{dim_labels[dim1]} (m)')
    ax.set_ylabel(f'{dim_labels[dim2]} (m)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    # Return handles and labels for unified legend
    return plot_handles, plot_labels

# --- NEW: State vs Normalized Time Plotting ---
def plot_state_vs_time(ax, all_labels, traj_lengths, n_states, colors):
    """
    Plots the predicted state against normalized time for all trajectories.
    Args:
        ax: Matplotlib axes object.
        all_labels: Array of predicted state labels for the concatenated data.
        traj_lengths: List containing the length of each original trajectory.
        n_states: Total number of states.
        colors: List of colors for states (assumes seaborn palette).
    """
    start_index = 0
    all_norm_times = []
    all_states_ordered = []

    # Generate normalized time for each trajectory
    for length in traj_lengths:
        end_index = start_index + length
        # Create normalized time vector [0, 1] for this trajectory
        # Avoid division by zero if length is 1
        if length > 1:
            norm_time = np.linspace(0, 1, length)
        elif length == 1:
             norm_time = np.array([0.5]) # Or [0] or [1], placing it in the middle
        else: # length is 0
             norm_time = np.array([])

        all_norm_times.append(norm_time)
        all_states_ordered.append(all_labels[start_index:end_index])
        start_index = end_index

    # Concatenate for plotting
    plot_times = np.concatenate(all_norm_times)
    plot_states = np.concatenate(all_states_ordered)

    # Add small jitter to states for better visualization of overlapping points
    state_jitter = plot_states + np.random.normal(0, 0.08, size=plot_states.shape)

    # Plot using scatter plot
    for state in range(n_states):
        state_mask = (plot_states == state)
        if np.any(state_mask):
            ax.scatter(plot_times[state_mask], state_jitter[state_mask],
                       color=colors[state], alpha=0.4, s=10, label=f'State {state}')

    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Predicted State Index")
    ax.set_title("State Occurrence vs. Normalized Trajectory Time")
    # Set Y-axis ticks to be integer state numbers
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([str(i) for i in range(n_states)])
    ax.grid(True, linestyle=':', alpha=0.7, axis='y') # Grid lines for states
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


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

    # 3. Predict states for all data points
    if hmm_model:
        try:
            # Predict states for the entire concatenated dataset
            all_predicted_states = hmm_model.predict(concatenated_data, lengths=traj_lengths)
            print("\nPredicted states for all data points.")

            # --- Visualization ---
            sns.set_theme(style="whitegrid") # Set seaborn style globally
            state_colors = sns.color_palette("viridis", N_STATES)

            # 4. Visualize GMM Contours (Existing Plot)
            print("Generating GMM contour visualizations...")
            fig_gmm, axs_gmm = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
            fig_gmm.suptitle(f'GMM State Distributions ({N_STATES} States, {N_MIX} Mix/State, Features: {N_FEATURES})', fontsize=16)

            h1, l1 = plot_gmm_contours(axs_gmm[0], concatenated_data, all_predicted_states, [0, 1], hmm_model, state_colors, "XY Projection")
            h2, l2 = plot_gmm_contours(axs_gmm[1], concatenated_data, all_predicted_states, [0, 2], hmm_model, state_colors, "XZ Projection")
            h3, l3 = plot_gmm_contours(axs_gmm[2], concatenated_data, all_predicted_states, [1, 2], hmm_model, state_colors, "YZ Projection")

            all_handles = h1 + h2 + h3
            all_labels = l1 + l2 + l3
            by_label = dict(zip(all_labels, all_handles))
            fig_gmm.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize='medium')
            plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
            plt.show() # Show GMM plot

            # 5. Visualize State vs Normalized Time (New Plot)
            print("Generating State vs Normalized Time plot...")
            fig_time, ax_time = plt.subplots(1, 1, figsize=(12, 6))
            plot_state_vs_time(ax_time, all_predicted_states, traj_lengths, N_STATES, state_colors)
            plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) # Adjust layout for legend
            plt.show() # Show time plot


            # Print learned parameters (optional)
            print("\nLearned HMM Parameters:")
            print(f"Features used: {FEATURE_COLS}")
            print("Initial Probabilities (pi):")
            print(hmm_model.startprob_)
            print("\nTransition Matrix (A):")
            print(np.round(hmm_model.transmat_, 3))
            print("\nGMM Weights per State:")
            print(np.round(hmm_model.weights_, 3))
            print("\nGMM Means per State (showing first 3 dims - Pos):")
            print(np.round(hmm_model.means_[:, :, :3], 3))

        except Exception as e:
            print(f"Error during prediction or plotting: {e}")
            # import traceback
            # traceback.print_exc()

    else:
        print("HMM model training failed. Cannot visualize GMMs.")
