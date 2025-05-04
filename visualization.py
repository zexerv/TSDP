import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import config # Import configuration
# --- NEW IMPORTS ---
try:
    from scipy.spatial.transform import Rotation as R
    import numpy.linalg # For SVD in central orientation calculation
except ImportError:
    print("\n*** WARNING: SciPy library not found or incomplete. Orientation stats cannot be calculated/plotted. ***")
    print("*** Please install it: pip install scipy ***\n")
    R = None # Set to None to disable orientation features if import fails
# --- END NEW IMPORTS ---


# --- Orientation Helper Functions (Copied/adapted from segment_analyzer.py) ---

def get_orientations_in_segment(aligned_trajs_list, segment_start, segment_end):
    """Extracts all 3x3 rotation matrices from a segment across trajectories."""
    orientations = []
    # Define the standard 9 rotation matrix column names
    rot_cols = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    # Check if *all* required columns are present in the first trajectory's DataFrame
    if not aligned_trajs_list or not all(col in aligned_trajs_list[0].columns for col in rot_cols):
        # print("  Debug: Rotation matrix columns (r11-r33) not found in first trajectory.")
        return None # Return None if columns are missing

    for traj_df in aligned_trajs_list:
        # Double-check columns for each trajectory (though less likely to differ after alignment)
        if not all(col in traj_df.columns for col in rot_cols):
            # print(f"  Debug: Skipping trajectory {aligned_trajs_list.index(traj_df)} due to missing rotation columns.")
            continue # Skip trajectory if it's missing columns

        if not traj_df.empty and segment_start < len(traj_df):
            end_slice = min(segment_end + 1, len(traj_df))
            segment_part = traj_df.iloc[segment_start:end_slice]
            if not segment_part.empty:
                # Extract rotation matrices for each time step
                try:
                    rot_matrices = segment_part[rot_cols].values.reshape(-1, 3, 3)
                    orientations.extend(list(rot_matrices)) # Add matrices from this traj
                except ValueError as e:
                    print(f"  Warning: Error reshaping rotation data in segment {segment_start}-{segment_end}. Skipping. Error: {e}")
                    continue # Skip if data is malformed

    if not orientations:
        # print(f"  Debug: No orientation matrices found in segment {segment_start}-{segment_end}.")
        return None # Return None if no orientations found

    return np.array(orientations) # Return as a numpy array (N, 3, 3)

def calculate_central_orientation(orientation_matrices):
    """
    Calculates a 'central' rotation matrix from a list of matrices.
    Uses element-wise mean followed by SVD projection to the nearest valid rotation.
    """
    if orientation_matrices is None or len(orientation_matrices) == 0:
        return None
    if R is None: return None # Check if SciPy is available

    mean_matrix = np.mean(orientation_matrices, axis=0)
    try:
        U, _, Vt = numpy.linalg.svd(mean_matrix)
        R_central = U @ Vt
        if numpy.linalg.det(R_central) < 0:
            Vt[-1, :] *= -1
            R_central = U @ Vt
        return R_central
    except numpy.linalg.LinAlgError:
        print("  Warning: SVD did not converge while calculating central orientation.")
        return None

def angle_between_matrices(R1, R2):
    """Calculates the angle (in degrees) between two rotation matrices."""
    if R1 is None or R2 is None or R is None: return np.nan
    try:
        R_rel = R1.T @ R2
        trace = np.trace(R_rel)
        trace = np.clip(trace, -1.0, 3.0)
        angle_rad = np.arccos((trace - 1.0) / 2.0)
        return np.degrees(angle_rad)
    except Exception as e:
        # print(f"  Warning: Error calculating angle between matrices: {e}") # Can be verbose
        return np.nan

def calculate_orientation_stats_for_plot(orientation_matrices):
    """
    Calculates central orientation matrix and max angular deviation.

    Returns:
        tuple: (central_R_matrix, max_angle_degrees) - Both can be None.
    """
    if orientation_matrices is None or len(orientation_matrices) == 0 or R is None:
        return None, None

    central_R = calculate_central_orientation(orientation_matrices)
    if central_R is None:
        return None, None

    max_angle = 0.0
    valid_angle_found = False
    for R_i in orientation_matrices:
        angle = angle_between_matrices(central_R, R_i)
        if not np.isnan(angle):
            max_angle = max(max_angle, angle)
            valid_angle_found = True

    max_angle_final = float(max_angle) if valid_angle_found else None
    return central_R, max_angle_final

# --- END Orientation Helper Functions ---


def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments, lambda_penalty):
    """Plots the raw and penalized segmentation cost vs. number of segments. (Unchanged)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = config.PLOT_FONT
    num_segments_axis = np.arange(1, max_segments + 1)
    plot_costs_raw = np.full(max_segments, np.inf)
    valid_len = min(len(raw_costs), max_segments)
    if valid_len > 0: plot_costs_raw[:valid_len] = raw_costs[:valid_len]
    penalized_costs = plot_costs_raw + lambda_penalty * num_segments_axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_raw = 'tab:blue'
    ax1.set_xlabel('Number of Segments')
    ax1.set_ylabel('Total Raw Segmentation Cost', color=color_raw)
    valid_idx_raw = np.isfinite(plot_costs_raw)
    if np.any(valid_idx_raw):
        ax1.plot(num_segments_axis[valid_idx_raw], plot_costs_raw[valid_idx_raw], marker='o', linestyle='-', color=color_raw, label='Raw Cost')
        ax1.tick_params(axis='y', labelcolor=color_raw)
        min_finite_cost_raw = np.min(plot_costs_raw[valid_idx_raw])
        ax1.set_ylim(bottom=min(0, min_finite_cost_raw * 0.9))
    else: ax1.text(0.5, 0.5, 'No finite raw costs', ha='center', va='center', transform=ax1.transAxes)
    ax2 = ax1.twinx()
    color_penalized = 'tab:red'
    ax2.set_ylabel('Total Penalized Cost (Raw + λ*N)', color=color_penalized)
    valid_idx_pen = np.isfinite(penalized_costs)
    if np.any(valid_idx_pen):
        ax2.plot(num_segments_axis[valid_idx_pen], penalized_costs[valid_idx_pen], marker='x', linestyle='--', color=color_penalized, label='Penalized Cost')
        ax2.tick_params(axis='y', labelcolor=color_penalized)
        min_finite_cost_pen = np.min(penalized_costs[valid_idx_pen])
        max_finite_cost_pen = np.max(penalized_costs[valid_idx_pen])
        padding = (max_finite_cost_pen - min_finite_cost_pen) * 0.05 if max_finite_cost_pen > min_finite_cost_pen else 1.0
        ax2.set_ylim(bottom=min(0, min_finite_cost_pen) - padding, top=max_finite_cost_pen + padding)
    else: ax2.text(0.5, 0.4, 'No finite penalized costs', ha='center', va='center', transform=ax2.transAxes)
    if 1 <= optimal_num_segments <= max_segments:
        optimal_raw_cost = plot_costs_raw[optimal_num_segments - 1]
        optimal_pen_cost = penalized_costs[optimal_num_segments - 1]
        if np.isfinite(optimal_raw_cost): ax1.scatter(optimal_num_segments, optimal_raw_cost, color='green', s=150, zorder=5, marker='*', label=f'Optimal ({optimal_num_segments}) - Raw')
        if np.isfinite(optimal_pen_cost): ax2.scatter(optimal_num_segments, optimal_pen_cost, color='magenta', s=150, zorder=5, marker='P', label=f'Optimal ({optimal_num_segments}) - Penalized')
    plt.title(f'Segmentation Cost vs. Number of Segments (λ={lambda_penalty:.3f})')
    ax1.set_xticks(np.arange(1, max_segments + 1))
    ax1.grid(True, linestyle=':', which='major', axis='x')
    ax2.grid(False)
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95))
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# --- MODIFIED FUNCTION ---
def plot_segmentation_with_trapezoids(
    min_vals_orig, max_vals_orig, segment_indices_orig,
    aligned_trajectories_list,
    all_mapped_events_list,
    feature_names, title="Optimal Tube Segmentation"):
    """
    Plots tube, segmentation, approximations, trajectories, and events.
    Adds central orientation line and max deviation text to rotation plots.
    """
    if R is None:
        print("SciPy not available. Cannot plot orientation statistics.")

    n_timesteps_orig, n_dims = min_vals_orig.shape
    if n_dims == 0: print("Warning: Cannot plot segmentation, no dimensions found."); return
    if len(aligned_trajectories_list) != len(all_mapped_events_list):
        print("Warning: Mismatch between trajectories and events. Skipping event plotting.")
        all_mapped_events_list = [{} for _ in aligned_trajectories_list]

    time_axis_orig = np.arange(n_timesteps_orig)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = config.PLOT_FONT

    ncols = config.PLOT_GRID_COLUMNS
    nrows = math.ceil(n_dims / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16, y=1.0)
    axs_flat = axs.flatten()

    event_markers = { # (Event marker config unchanged)
        'start': {'marker': 'o', 'color': 'green', 'size': 7, 'label': 'Start Point'},
        'end': {'marker': 'o', 'color': 'red', 'size': 7, 'label': 'End Point'},
        'state_change': {'marker': '*', 'color': '#FFA500', 'size': 10, 'label': 'State Change'},
        'wp_saved': {'marker': 's', 'color': 'blue', 'size': 7, 'label': 'WP Saved'},
        'gripper_change': {'marker': 'X', 'color': 'magenta', 'size': 7, 'label': 'Gripper Change'}
    }
    plotted_legend_labels = set() # Track labels for combined legend

    # --- Pre-calculate orientation stats for each segment ---
    segment_orientation_stats = []
    if R is not None: # Only calculate if SciPy is available
        print("Calculating orientation stats for plotting...")
        for i in range(len(segment_indices_orig) - 1):
            start_idx_orig = segment_indices_orig[i]
            end_idx_orig = segment_indices_orig[i+1] - 1
            if start_idx_orig > end_idx_orig:
                segment_orientation_stats.append({'R_center': None, 'theta_max': None})
                continue

            segment_orientations = get_orientations_in_segment(
                aligned_trajectories_list, start_idx_orig, end_idx_orig
            )
            R_center, theta_max = calculate_orientation_stats_for_plot(segment_orientations)
            segment_orientation_stats.append({'R_center': R_center, 'theta_max': theta_max})
            # print(f"  Segment {i}: R_center calculated: {R_center is not None}, theta_max: {theta_max}") # Debug print
    else:
        # Fill with None if SciPy not available
        segment_orientation_stats = [{'R_center': None, 'theta_max': None}] * (len(segment_indices_orig) - 1)
    # --- End pre-calculation ---


    # --- Plot Each Dimension ---
    rot_feature_indices = {name: (int(name[1])-1, int(name[2])-1) for name in feature_names if name.startswith('r') and len(name)==3 and name[1:].isdigit()}
    plot_idx_for_angle_text = list(rot_feature_indices.keys())[0] if rot_feature_indices else None # e.g., 'r11'

    for d in range(n_dims):
        row, col = d // ncols, d % ncols
        ax = axs[row, col]
        feature_name = feature_names[d]
        is_rotation_feature = feature_name in rot_feature_indices

        # 1. Plot Tube (Unchanged)
        ax.plot(time_axis_orig, max_vals_orig[:, d], color='lightblue', linestyle='-', linewidth=1.0, alpha=0.8, label='Max Bound' if d==0 else "")
        ax.plot(time_axis_orig, min_vals_orig[:, d], color='lightcoral', linestyle='-', linewidth=1.0, alpha=0.8, label='Min Bound' if d==0 else "")
        ax.fill_between(time_axis_orig, min_vals_orig[:, d], max_vals_orig[:, d], color='lightgrey', alpha=0.4, label='Tube Range' if d==0 else "")
        if d==0: plotted_legend_labels.update(['Max Bound', 'Min Bound', 'Tube Range'])

        # 2. Plot Boundaries (Unchanged)
        for i, boundary_idx_orig in enumerate(segment_indices_orig):
            if i > 0 and boundary_idx_orig <= n_timesteps_orig: # Use <= to draw line at the very end
                 label = 'Segment Boundary' if 'Segment Boundary' not in plotted_legend_labels else ""
                 ax.axvline(x=boundary_idx_orig, color='black', linestyle='--', linewidth=1.2, label=label)
                 if label: plotted_legend_labels.add('Segment Boundary')

        # 3. Plot Trapezoids & ADD Orientation Stats Overlay
        for i in range(len(segment_indices_orig) - 1):
            start_idx_orig = segment_indices_orig[i]
            end_idx_orig = segment_indices_orig[i+1] - 1
            if end_idx_orig < start_idx_orig or start_idx_orig >= n_timesteps_orig: continue
            end_idx_orig = min(end_idx_orig, n_timesteps_orig - 1)
            segment_time_axis_orig = np.arange(start_idx_orig, end_idx_orig + 1)
            segment_len_points = len(segment_time_axis_orig)
            if segment_len_points <= 1: continue # Skip single points for lines

            # Plot Trapezoids (Unchanged)
            min_start_val = min_vals_orig[start_idx_orig, d]; min_end_val = min_vals_orig[end_idx_orig, d]
            max_start_val = max_vals_orig[start_idx_orig, d]; max_end_val = max_vals_orig[end_idx_orig, d]
            t_interp = np.arange(segment_len_points)
            min_slope = (min_end_val - min_start_val) / (segment_len_points - 1); approx_min = min_start_val + min_slope * t_interp
            max_slope = (max_end_val - max_start_val) / (segment_len_points - 1); approx_max = max_start_val + max_slope * t_interp
            label_approx_max = 'Linear Approx (Max)' if 'Linear Approx (Max)' not in plotted_legend_labels else ""; label_approx_min = 'Linear Approx (Min)' if 'Linear Approx (Min)' not in plotted_legend_labels else ""
            ax.plot(segment_time_axis_orig, approx_max, color='blue', linestyle='-', linewidth=1.5, label=label_approx_max)
            ax.plot(segment_time_axis_orig, approx_min, color='red', linestyle='-', linewidth=1.5, label=label_approx_min)
            if label_approx_max: plotted_legend_labels.add('Linear Approx (Max)')
            if label_approx_min: plotted_legend_labels.add('Linear Approx (Min)')

            # --- ADD Orientation Stats Overlay ---
            if is_rotation_feature and R is not None:
                stats = segment_orientation_stats[i]
                R_center = stats['R_center']
                theta_max = stats['theta_max']

                if R_center is not None:
                    # Get the specific element (e.g., r11 -> [0,0])
                    row_idx, col_idx = rot_feature_indices[feature_name]
                    center_value = R_center[row_idx, col_idx]
                    # Plot horizontal line for central value
                    label_center = 'Central Orient.' if 'Central Orient.' not in plotted_legend_labels else ""
                    ax.plot(segment_time_axis_orig, [center_value] * segment_len_points,
                            color='purple', linestyle='--', linewidth=2.0, label=label_center, zorder=5)
                    if label_center: plotted_legend_labels.add('Central Orient.')

                # Add text annotation for theta_max on the designated plot
                if theta_max is not None and feature_name == plot_idx_for_angle_text:
                    text_x = start_idx_orig + (end_idx_orig - start_idx_orig) / 2 # Center of segment
                    # Position text slightly above the max tube boundary for visibility
                    text_y = ax.get_ylim()[1] * 0.95 # Adjust vertical position as needed
                    ax.text(text_x, text_y, f"θmax:{theta_max:.1f}°",
                            color='purple', ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
            # --- END Orientation Stats Overlay ---


        # 4. Plot Individual Trajectories and Events (Unchanged)
        for traj_idx, aligned_traj_df in enumerate(aligned_trajectories_list):
            if feature_name in aligned_traj_df.columns:
                label_traj = 'Aligned Traj.' if 'Aligned Traj.' not in plotted_legend_labels else ""
                ax.plot(time_axis_orig, aligned_traj_df[feature_name], linewidth=0.5, alpha=0.5, color='grey', label=label_traj, zorder=1)
                if label_traj: plotted_legend_labels.add('Aligned Traj.')
                current_events = all_mapped_events_list[traj_idx]
                if current_events:
                    for event_type, indices in current_events.items():
                        if event_type in event_markers and indices:
                            marker_cfg = event_markers[event_type]
                            valid_indices = [idx for idx in indices if 0 <= idx < n_timesteps_orig]
                            if not valid_indices: continue
                            try:
                                y_values = aligned_traj_df[feature_name].iloc[valid_indices].values
                                label = marker_cfg['label'] if marker_cfg['label'] not in plotted_legend_labels else ""
                                ax.plot(valid_indices, y_values, marker=marker_cfg['marker'], color=marker_cfg['color'], markersize=marker_cfg['size'], linestyle='None', label=label, alpha=0.8, zorder=10)
                                if label: plotted_legend_labels.add(marker_cfg['label'])
                            except IndexError: continue # Ignore if index somehow invalid

        ax.set_ylabel(feature_name)
        ax.grid(True, linestyle=':', which='both', axis='both')


    # --- Clean up Empty Subplots ---
    for i in range(n_dims, nrows * ncols): fig.delaxes(axs_flat[i])

    # --- Add Combined Legend ---
    handles, labels = [], []
    for ax in axs_flat[:n_dims]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(len(by_label), 6), fontsize='small') # Increased ncol slightly

    # --- Final Touches ---
    last_row_idx = (n_dims -1) // ncols
    for c in range(ncols):
        ax_idx = last_row_idx * ncols + c
        if ax_idx < n_dims: axs[last_row_idx, c].set_xlabel('Time Step (Aligned Scale)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

