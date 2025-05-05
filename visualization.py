# -*- coding: utf-8 -*-
"""
Visualization functions for the trajectory segmentation pipeline.
Includes plotting segmentation results, cost curves, and orientation deviation.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import yaml # For loading cross-section results if needed
from pathlib import Path
import sys 
import plotly.graph_objects as go
from itertools import combinations
# --- Project Modules ---
try:
    import config # Import configuration
except ImportError:
    print("FATAL ERROR: config.py not found. Ensure it's in the same directory or Python path.")
    sys.exit(1)

# --- SciPy for Orientation ---
try:
    from scipy.spatial.transform import Rotation as R
    # Import necessary helper functions (assuming they are now in segment_analyzer or a utils module)
    # If they remain here, keep them. If moved, import them.
    # For now, assume we need calculate_geodesic_distance locally or import it.
    # Let's define it here for simplicity, mirroring the one in segment_analyzer.
    def calculate_geodesic_distance(q1_wxyz, q2_wxyz):
        """ Calculates geodesic distance (angle in degrees) between two quaternions [w, x, y, z]. """
        if q1_wxyz is None or q2_wxyz is None or not isinstance(q1_wxyz, np.ndarray) or not isinstance(q2_wxyz, np.ndarray) or q1_wxyz.shape != (4,) or q2_wxyz.shape != (4,):
            return np.nan
        try:
            q1_xyzw = q1_wxyz[[1, 2, 3, 0]]; q2_xyzw = q2_wxyz[[1, 2, 3, 0]]
            r1 = R.from_quat(q1_xyzw); r2 = R.from_quat(q2_xyzw)
            diff_rotvec = (r1.inv() * r2).as_rotvec()
            angle_rad = np.linalg.norm(diff_rotvec)
            angle_rad = np.clip(angle_rad, 0, np.pi)
            return np.degrees(angle_rad)
        except Exception: return np.nan
except ImportError:
    print("\n*** WARNING: SciPy library not found or incomplete. Orientation deviation plot cannot be generated. ***")
    print("*** Please install it: pip install scipy ***\n")
    R = None
    calculate_geodesic_distance = None # Disable function if SciPy not available

# --- Plotting Functions ---

# Add this helper function (can be placed anywhere before it's used)
def _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0, n_points=50):
    """
    Generates x, y points for a 2D confidence ellipse for Plotly shapes or scatter lines.

    Args:
        mean_2d (np.ndarray): 2D mean vector [x, y].
        cov_2d (np.ndarray): 2x2 covariance matrix.
        n_std (float): Number of standard deviations for the ellipse boundary.
        n_points (int): Number of points to generate for the ellipse contour.

    Returns:
        tuple: (x_points, y_points) for the ellipse, or (None, None) if error.
    """
    try:
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(cov_2d)
        # Ensure eigenvalues are positive for valid ellipse
        if np.any(vals <= 0):
             print("Warning: Non-positive eigenvalue encountered in covariance matrix. Cannot draw ellipse.")
             return None, None

        # Angle of the major axis
        angle = np.arctan2(*vecs[:, 0][::-1])
        # Ellipse radii (scaled by n_std)
        width, height = 2 * n_std * np.sqrt(vals)

        # Parametric equation for ellipse points
        t = np.linspace(0, 2 * np.pi, n_points)
        xs = width / 2 * np.cos(t)
        ys = height / 2 * np.sin(t)

        # Rotation matrix
        R_mat = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

        # Rotate and translate points
        points = R_mat @ np.vstack([xs, ys])
        x_points = points[0, :] + mean_2d[0]
        y_points = points[1, :] + mean_2d[1]

        return x_points, y_points
    except np.linalg.LinAlgError:
        print("Warning: Linear algebra error (e.g., SVD did not converge) calculating ellipse. Cannot draw ellipse.")
        return None, None
    except Exception as e:
        print(f"Warning: Unexpected error calculating ellipse points: {e}")
        return None, None


# Add this new function
def plot_gmm_cross_section_interactive(gmm_params, snapshot_data,
                                       feature_names=config.GMM_STATE_COLS,
                                       title="Interactive GMM Cross-Section Visualization"):
    """
    Creates an interactive Plotly plot showing pairwise projections of GMM components
    and snapshot data for a single cross-section.

    Args:
        gmm_params (dict): Dictionary containing 'weights', 'means', 'covariances'
                           for the GMM at this cross-section. Should also contain
                           'n_components_used'.
        snapshot_data (np.ndarray): The N x D array of state snapshots used to train the GMM.
                                    (N = num trajectories, D = num features).
        feature_names (list): List of names for the D dimensions.
        title (str): Title for the plot.
    """
    print(f"--- Generating Interactive GMM Plot: {title} ---")
    if gmm_params is None:
        print("  Error: gmm_params is None. Cannot generate plot.")
        return
    if snapshot_data is None or snapshot_data.ndim != 2 or snapshot_data.shape[0] == 0:
        print("  Error: snapshot_data is invalid or empty. Cannot generate plot.")
        return

    weights = np.array(gmm_params['weights'])
    means = np.array(gmm_params['means'])
    covariances = np.array(gmm_params['covariances'])
    n_components = gmm_params.get('n_components_used', len(weights)) # Get actual components used
    n_features = len(feature_names)

    if means.shape[1] != n_features or covariances.shape[2] != n_features:
         print(f"  Error: GMM parameter dimensions mismatch feature_names length ({n_features}).")
         return

    # --- Create Figure ---
    # Use a single subplot that will be updated by the dropdown
    fig = go.Figure()

    # --- Generate Traces for All Pairs ---
    dimension_indices = list(range(n_features))
    pairwise_combinations = list(combinations(dimension_indices, 2))
    traces_visibility = [] # List to store visibility flags for update menu

    print(f"  Generating traces for {len(pairwise_combinations)} dimension pairs...")
    component_colors = sns.color_palette("viridis", n_components).as_hex() # Colors for components

    for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
        visible = (i == 0) # Make only the first pair visible initially
        visibility_flags = [False] * len(pairwise_combinations) * (1 + n_components) # (1 scatter + K ellipses per pair)
        start_idx_for_pair = i * (1 + n_components)

        # 1. Snapshot Scatter Trace
        fig.add_trace(go.Scatter(
            x=snapshot_data[:, dim_x_idx],
            y=snapshot_data[:, dim_y_idx],
            mode='markers',
            marker=dict(color='grey', size=5, opacity=0.7),
            name='Snapshots',
            visible=visible,
            showlegend=(i==0) # Show legend only once for snapshots
        ))
        visibility_flags[start_idx_for_pair] = visible

        # 2. GMM Component Ellipse Traces
        for k in range(n_components):
            mean_k = means[k]
            cov_k = covariances[k]
            weight_k = weights[k]

            # Extract 2D marginal mean and covariance
            mean_2d = mean_k[[dim_x_idx, dim_y_idx]]
            cov_2d = cov_k[np.ix_([dim_x_idx, dim_y_idx], [dim_x_idx, dim_y_idx])]

            # Generate ellipse points
            x_ellipse, y_ellipse = _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0) # 2 std deviations

            if x_ellipse is not None:
                fig.add_trace(go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode='lines',
                    line=dict(color=component_colors[k], width=2),
                    fill='toself', # Fill the ellipse
                    fillcolor=component_colors[k],
                    opacity=0.3 + 0.6 * weight_k, # Opacity based on weight
                    name=f'Comp {k+1} (w={weight_k:.2f})',
                    visible=visible,
                    showlegend=(i==0) # Show legend only once per component
                ))
                visibility_flags[start_idx_for_pair + 1 + k] = visible
            else:
                 # Add a dummy invisible trace if ellipse fails, to keep indexing consistent
                 fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f'Comp {k+1} Error', visible=False))
                 visibility_flags[start_idx_for_pair + 1 + k] = False # Ensure it's false

        traces_visibility.append(visibility_flags)

    # --- Create Dropdown Menu ---
    buttons = []
    for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
        # Create visibility mask for this button press
        visibility_mask = [False] * len(fig.data) # Start with all false
        start_idx = i * (1 + n_components)
        end_idx = start_idx + (1 + n_components)
        for trace_idx in range(start_idx, end_idx):
             # Check if trace exists before setting visibility
             if trace_idx < len(visibility_mask):
                  # Only make visible if ellipse calculation succeeded for this component
                  # (Snapshot trace at start_idx is always assumed okay if data exists)
                  is_snapshot_trace = (trace_idx == start_idx)
                  component_trace_idx = trace_idx - start_idx - 1
                  # Check if corresponding ellipse trace had data
                  ellipse_had_data = fig.data[trace_idx].x is not None and fig.data[trace_idx].x[0] is not None

                  if is_snapshot_trace or ellipse_had_data:
                       visibility_mask[trace_idx] = True


        buttons.append(dict(
            label=f"{feature_names[dim_x_idx]} vs {feature_names[dim_y_idx]}",
            method="update",
            args=[{"visible": visibility_mask}, # Update visibility
                  {"xaxis.title": feature_names[dim_x_idx], # Update x-axis label
                   "yaxis.title": feature_names[dim_y_idx]}] # Update y-axis label
        ))

    # Add dropdown to layout
    fig.update_layout(
        updatemenus=[dict(
            active=0, # Default selection
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        title=title,
        hovermode="closest" # Enable hover information
    )

    # Set initial axis labels for the default view (first pair)
    initial_dim_x_idx, initial_dim_y_idx = pairwise_combinations[0]
    fig.update_layout(
        xaxis_title=feature_names[initial_dim_x_idx],
        yaxis_title=feature_names[initial_dim_y_idx],
        legend_title_text="Components",
        # Use equal scaling for better ellipse visualization? Optional.
        # yaxis_scaleanchor="x",
        # yaxis_scaleratio=1,
    )

    print("  Displaying interactive plot...")
    fig.show() # Display in browser or compatible environment
    # Or save to HTML:
    # html_filename = f"gmm_interactive_{title.replace(' ', '_')}.html"
    # fig.write_html(html_filename)
    # print(f"  Interactive plot saved to: {html_filename}")

def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments, lambda_penalty):
    """Plots the raw and penalized segmentation cost vs. number of segments. (Unchanged from previous version)"""
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
    plt.title(f'Segmentation Cost vs. Number of Segments (λ={config.LAMBDA_PENALTY:.3f})')
    ax1.set_xticks(np.arange(1, max_segments + 1))
    ax1.grid(True, linestyle=':', which='major', axis='x')
    ax2.grid(False)
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95))
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# --- MODIFIED Main Plotting Function ---
def plot_segmentation_with_trapezoids(
    min_vals_orig, max_vals_orig, # Tube data for features in PLOT_FEATURE_COLS
    segment_indices_orig, # List of boundary indices [0, t1, t2, ..., tend]
    processed_aligned_trajs, # List of DataFrames (output from analyzer preprocessing)
    all_mapped_events_list, # List of event dicts corresponding to trajectories
    feature_names=config.PLOT_FEATURE_COLS, # Features to plot (e.g., pos, logmap)
    cross_section_results_path=config.CROSS_SECTION_STATS_OUTPUT_PATH, # Path to load results
    title="Optimal Tube Segmentation"):
    """
    Plots tube, segmentation, approximations, trajectories, and events for specified features.
    Optionally adds a plot for orientation geodesic deviation based on cross-section analysis results.
    Removes the old central orientation line and theta_max text from rXX plots.
    """
    print("--- Generating Segmentation Plots ---")

    # --- Input Validation ---
    if min_vals_orig is None or max_vals_orig is None:
        print("Warning: Tube data (min/max vals) is None. Cannot generate main plots.")
        return
    if not processed_aligned_trajs:
        print("Warning: No processed aligned trajectories provided. Cannot plot trajectories/events.")
        return
    if len(processed_aligned_trajs) != len(all_mapped_events_list):
        print("Warning: Mismatch between trajectories and events list length. Event plotting might be incorrect.")
        # Adjust events list length for safety, though ideally this shouldn't happen
        all_mapped_events_list = all_mapped_events_list[:len(processed_aligned_trajs)]

    n_timesteps_orig, n_dims_tube = min_vals_orig.shape
    n_features_to_plot = len(feature_names)
    if n_features_to_plot == 0:
        print("Warning: No features specified in config.PLOT_FEATURE_COLS. Nothing to plot.")
        return
    if n_dims_tube != n_features_to_plot:
         print(f"Warning: Tube data dimension ({n_dims_tube}) does not match number of features to plot ({n_features_to_plot}). Check config.PLOT_FEATURE_COLS.")
         # Attempt to plot based on feature_names, assuming columns exist in traj data
         # Tube plotting might fail if dimensions mismatch.

    time_axis_orig = np.arange(n_timesteps_orig)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = config.PLOT_FONT

    # --- Load Cross-Section Results (Needed for Deviation Plot) ---
    cross_section_data = None
    if config.PLOT_ORIENTATION_DEVIATION and R is not None and calculate_geodesic_distance is not None:
        print(f"  Loading cross-section results from: {cross_section_results_path}")
        try:
            cs_path = Path(cross_section_results_path)
            if cs_path.is_file():
                with open(cs_path, 'r') as f:
                    cross_section_data = yaml.safe_load(f)
                if not isinstance(cross_section_data, list):
                     print(f"  Warning: Loaded cross-section data is not a list. Cannot plot deviation.")
                     cross_section_data = None
                else:
                     print(f"  Successfully loaded {len(cross_section_data)} cross-section analysis results.")
            else:
                print(f"  Warning: Cross-section results file not found at {cross_section_results_path}. Cannot plot deviation.")
        except Exception as e:
            print(f"  Error loading cross-section results: {e}. Cannot plot deviation.")
            cross_section_data = None
    elif config.PLOT_ORIENTATION_DEVIATION:
         print("  Orientation deviation plot enabled, but SciPy is not available. Skipping deviation plot.")


    # --- Setup Figure Grid ---
    plot_deviation = config.PLOT_ORIENTATION_DEVIATION and (cross_section_data is not None)
    total_plots = n_features_to_plot + (1 if plot_deviation else 0)
    ncols = config.PLOT_GRID_COLUMNS
    nrows = math.ceil(total_plots / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16, y=1.0)
    axs_flat = axs.flatten()

    # --- Event Markers Config --- (Unchanged)
    event_markers = {
        'start': {'marker': 'o', 'color': 'green', 'size': 7, 'label': 'Start Point'},
        'end': {'marker': 'o', 'color': 'red', 'size': 7, 'label': 'End Point'},
        'state_change': {'marker': '*', 'color': '#FFA500', 'size': 10, 'label': 'State Change'},
        'wp_saved': {'marker': 's', 'color': 'blue', 'size': 7, 'label': 'WP Saved'},
        'gripper_change': {'marker': 'X', 'color': 'magenta', 'size': 7, 'label': 'Gripper Change'}
    }
    plotted_legend_labels = set() # Track labels for combined legend

    # --- Plot Each Feature ---
    print("  Plotting individual features...")
    for d_idx, feature_name in enumerate(feature_names):
        row, col = d_idx // ncols, d_idx % ncols
        ax = axs[row, col]
        print(f"    Plotting feature: {feature_name}")

        # Check if feature exists in trajectory data (more robust)
        if feature_name not in processed_aligned_trajs[0].columns:
             print(f"    Warning: Feature '{feature_name}' not found in trajectory data. Skipping plot for this feature.")
             ax.text(0.5, 0.5, f"Feature '{feature_name}'\nnot found in data", ha='center', va='center', transform=ax.transAxes, color='red')
             ax.set_ylabel(feature_name)
             continue

        # 1. Plot Tube (if dimensions match)
        if d_idx < n_dims_tube:
            ax.plot(time_axis_orig, max_vals_orig[:, d_idx], color='lightblue', linestyle='-', linewidth=1.0, alpha=0.8, label='Max Bound' if d_idx==0 else "")
            ax.plot(time_axis_orig, min_vals_orig[:, d_idx], color='lightcoral', linestyle='-', linewidth=1.0, alpha=0.8, label='Min Bound' if d_idx==0 else "")
            ax.fill_between(time_axis_orig, min_vals_orig[:, d_idx], max_vals_orig[:, d_idx], color='lightgrey', alpha=0.4, label='Tube Range' if d_idx==0 else "")
            if d_idx==0: plotted_legend_labels.update(['Max Bound', 'Min Bound', 'Tube Range'])
        else:
             print(f"    Skipping tube plot for '{feature_name}' due to dimension mismatch.")

        # 2. Plot Boundaries
        for i, boundary_idx_orig in enumerate(segment_indices_orig):
            if i > 0 and boundary_idx_orig <= n_timesteps_orig: # Use <= to draw line at the very end
                label = 'Segment Boundary' if 'Segment Boundary' not in plotted_legend_labels else ""
                ax.axvline(x=boundary_idx_orig, color='black', linestyle='--', linewidth=1.2, label=label)
                if label: plotted_legend_labels.add('Segment Boundary')

        # 3. Plot Trapezoids (if tube dimensions match)
        if d_idx < n_dims_tube:
            for i in range(len(segment_indices_orig) - 1):
                start_idx_orig = segment_indices_orig[i]
                end_idx_orig = segment_indices_orig[i+1] - 1
                if end_idx_orig < start_idx_orig or start_idx_orig >= n_timesteps_orig: continue
                end_idx_orig = min(end_idx_orig, n_timesteps_orig - 1)
                segment_time_axis_orig = np.arange(start_idx_orig, end_idx_orig + 1)
                segment_len_points = len(segment_time_axis_orig)
                if segment_len_points <= 1: continue # Skip single points for lines

                min_start_val = min_vals_orig[start_idx_orig, d_idx]; min_end_val = min_vals_orig[end_idx_orig, d_idx]
                max_start_val = max_vals_orig[start_idx_orig, d_idx]; max_end_val = max_vals_orig[end_idx_orig, d_idx]
                t_interp = np.arange(segment_len_points)
                min_slope = (min_end_val - min_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
                approx_min = min_start_val + min_slope * t_interp
                max_slope = (max_end_val - max_start_val) / (segment_len_points - 1) if segment_len_points > 1 else 0
                approx_max = max_start_val + max_slope * t_interp
                label_approx_max = 'Linear Approx (Max)' if 'Linear Approx (Max)' not in plotted_legend_labels else ""; label_approx_min = 'Linear Approx (Min)' if 'Linear Approx (Min)' not in plotted_legend_labels else ""
                ax.plot(segment_time_axis_orig, approx_max, color='blue', linestyle='-', linewidth=1.5, label=label_approx_max)
                ax.plot(segment_time_axis_orig, approx_min, color='red', linestyle='-', linewidth=1.5, label=label_approx_min)
                if label_approx_max: plotted_legend_labels.add('Linear Approx (Max)')
                if label_approx_min: plotted_legend_labels.add('Linear Approx (Min)')
        else:
            print(f"    Skipping trapezoid plot for '{feature_name}' due to dimension mismatch.")

        # NOTE: Removed the central orientation line and theta_max text plotting logic here.

        # 4. Plot Individual Trajectories and Events
        for traj_idx, aligned_traj_df in enumerate(processed_aligned_trajs):
            # Plot trajectory line
            label_traj = 'Aligned Traj.' if 'Aligned Traj.' not in plotted_legend_labels else ""
            ax.plot(time_axis_orig, aligned_traj_df[feature_name], linewidth=0.5, alpha=0.5, color='grey', label=label_traj, zorder=1)
            if label_traj: plotted_legend_labels.add('Aligned Traj.')

            # Plot events on this trajectory
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

    # --- Plot Orientation Deviation (Optional) ---
    if plot_deviation:
        print("  Plotting orientation deviation...")
        dev_plot_idx = n_features_to_plot # Index for the next available subplot
        row, col = dev_plot_idx // ncols, dev_plot_idx % ncols
        ax = axs[row, col]
        ax.set_ylabel("Geodesic Dist from Mean (deg)")
        ax.grid(True, linestyle=':', which='both', axis='both')

        # Need to map segment intervals to cross-section results (use start boundary time)
        cs_data_map = {d['time_index']: d for d in cross_section_data if d['type'] == 'boundary'}

        max_overall_deviation = 0 # Track max deviation for y-limit

        for i in range(len(segment_indices_orig) - 1):
            start_idx_orig = segment_indices_orig[i]
            end_idx_orig = segment_indices_orig[i+1] - 1
            if end_idx_orig < start_idx_orig or start_idx_orig >= n_timesteps_orig: continue
            end_idx_orig = min(end_idx_orig, n_timesteps_orig - 1)
            segment_time_axis_orig = np.arange(start_idx_orig, end_idx_orig + 1)

            # Find the corresponding boundary cross-section data
            # Use the exact boundary index as the key
            boundary_cs_data = cs_data_map.get(float(start_idx_orig)) # Use float for key matching

            if boundary_cs_data and boundary_cs_data.get('orient_mean_quaternion'):
                mean_quat_list = boundary_cs_data['orient_mean_quaternion']
                max_dist_segment = boundary_cs_data.get('orient_geodesic_max_distance')
                mean_quat_wxyz = np.array(mean_quat_list)

                # Plot max distance line for the segment
                if max_dist_segment is not None:
                     label_max_dev = r'Max $\theta$ (Cross-Section)' if r'Max $\theta$ (Cross-Section)' not in plotted_legend_labels else ""
                     ax.plot(segment_time_axis_orig, [max_dist_segment] * len(segment_time_axis_orig),
                             color='purple', linestyle='--', linewidth=1.5, label=label_max_dev, zorder=5)
                     if label_max_dev: plotted_legend_labels.add(r'Max $\theta$ (Cross-Section)')
                     max_overall_deviation = max(max_overall_deviation, max_dist_segment)

                # Plot individual trajectory deviations from this segment's mean
                for traj_idx, aligned_traj_df in enumerate(processed_aligned_trajs):
                     if 'quat_wxyz' in aligned_traj_df.columns:
                         # Extract quaternions for the segment duration
                         segment_quats = np.array(aligned_traj_df['quat_wxyz'].iloc[start_idx_orig:end_idx_orig+1].tolist())
                         if segment_quats.ndim == 2 and segment_quats.shape[1] == 4:
                             # Calculate geodesic distance for each point in the segment
                             distances = [calculate_geodesic_distance(mean_quat_wxyz, q) for q in segment_quats]
                             valid_distances = [d if not np.isnan(d) else 0 for d in distances] # Replace NaN with 0 for plotting
                             label_traj_dev = r'$\theta$ from Mean (Traj)' if r'$\theta$ from Mean (Traj)' not in plotted_legend_labels else ""
                             ax.plot(segment_time_axis_orig, valid_distances, linewidth=0.5, alpha=0.5, color='grey', label=label_traj_dev, zorder=1)
                             if label_traj_dev: plotted_legend_labels.add(r'$\theta$ from Mean (Traj)')
                             max_overall_deviation = max(max_overall_deviation, np.max(valid_distances) if valid_distances else 0)

            else:
                 print(f"    Warning: Mean orientation data not found for segment starting at {start_idx_orig}. Skipping deviation plot for this segment.")

            # Add segment boundary line to deviation plot too
            if i < len(segment_indices_orig) - 2: # Don't draw after last segment
                 boundary_idx_orig = segment_indices_orig[i+1]
                 ax.axvline(x=boundary_idx_orig, color='black', linestyle='--', linewidth=1.2)

        # Set y-limit for deviation plot
        ax.set_ylim(bottom=0, top=max_overall_deviation * 1.1 if max_overall_deviation > 0 else 10) # Add padding

    # --- Clean up Empty Subplots ---
    for i in range(total_plots, nrows * ncols):
        fig.delaxes(axs_flat[i])

    # --- Add Combined Legend ---
    handles, labels = [], []
    # Collect handles/labels from all axes that were plotted on
    for i in range(total_plots):
        if i < len(axs_flat):
            ax = axs_flat[i]
            # Check if axis has any lines or collections (handles potential empty plots)
            if ax.has_data():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

    # Create unique legend entries
    by_label = dict(zip(labels, handles))
    if by_label: # Only show legend if there's something to show
        fig.legend(by_label.values(), by_label.keys(), loc='lower center',
                   bbox_to_anchor=(0.5, -0.05 if nrows > 1 else -0.1), # Adjust vertical position based on rows
                   ncol=min(len(by_label), 6), fontsize='small')

    # --- Final Touches ---
    # Add X-label to bottom-most plots
    last_row_plots_indices = range((nrows - 1) * ncols, total_plots)
    for plot_idx in last_row_plots_indices:
         row, col = plot_idx // ncols, plot_idx % ncols
         if plot_idx < len(axs_flat): # Check if subplot exists
              axs[row, col].set_xlabel('Time Step (Aligned Scale)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to make space for legend
    print("--- Plotting Complete ---")
    plt.show()

# Example Usage (if run directly, requires data to be generated first)
# if __name__ == "__main__":
#     print("This script contains plotting functions.")
#     print("Run main.py or segment_analyzer.py to generate data and then call these functions.")
#     # Add example calls here if desired, loading dummy data or saved results
