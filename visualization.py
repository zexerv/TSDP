# -*- coding: utf-8 -*-
"""
Visualization functions for the trajectory segmentation pipeline.
Includes plotting segmentation results (tube plot), cost curves,
and a simple plot for GMM K=1 results (mean/covariance).
MODIFIED TO:
- Remove interactive GMM plot and orientation deviation plot.
- Correct NameError in simple GMM plot.
- Add print stats function.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches # For ellipses
import numpy as np
import seaborn as sns
import math
import yaml # For loading cross-section results if needed
from pathlib import Path
import sys

# --- Project Modules ---
try:
    import config # Import configuration
except ImportError:
    print("FATAL ERROR: config.py not found. Ensure it's in the same directory or Python path.")
    sys.exit(1)

# --- SciPy (Optional - only needed if deviation plot were re-enabled) ---
try:
    from scipy.spatial.transform import Rotation as R
    # Define helper locally if needed by other functions
    def calculate_geodesic_distance(q1_wxyz, q2_wxyz):
        if q1_wxyz is None or q2_wxyz is None or not isinstance(q1_wxyz, np.ndarray) or not isinstance(q2_wxyz, np.ndarray) or q1_wxyz.shape != (4,) or q2_wxyz.shape != (4,): return np.nan
        try:
            q1_xyzw = q1_wxyz[[1, 2, 3, 0]]; q2_xyzw = q2_wxyz[[1, 2, 3, 0]]
            r1 = R.from_quat(q1_xyzw); r2 = R.from_quat(q2_xyzw)
            diff_rotvec = (r1.inv() * r2).as_rotvec()
            angle_rad = np.linalg.norm(diff_rotvec); angle_rad = np.clip(angle_rad, 0, np.pi)
            return np.degrees(angle_rad)
        except Exception: return np.nan
except ImportError:
    # print("\nSciPy library not found. Advanced orientation plots disabled.") # Less verbose
    R = None
    calculate_geodesic_distance = None

# --- Plotting Functions ---

def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments, lambda_penalty):
    """Plots the raw and penalized segmentation cost vs. number of segments."""
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
    # Combine legends carefully
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    all_labels = labels1 + labels2; all_handles = lines1 + lines2
    unique_labels = {};
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels: unique_labels[label] = handle
    if unique_labels: fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.99, 0.95))
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_segmentation_with_trapezoids(
    min_vals_orig, max_vals_orig, # Tube data for features in PLOT_FEATURE_COLS
    segment_indices_orig, # List of boundary indices [0, t1, t2, ..., tend]
    processed_aligned_trajs, # List of DataFrames (output from analyzer preprocessing)
    all_mapped_events_list, # List of event dicts corresponding to trajectories
    feature_names=config.PLOT_FEATURE_COLS, # Features to plot (e.g., pos, rotmat)
    title="Optimal Tube Segmentation"):
    """
    Plots tube, segmentation, approximations, trajectories, and events for specified features.
    """
    print("--- Generating Segmentation Plots ---")

    # Input Validation
    if min_vals_orig is None or max_vals_orig is None:
        print("Warning: Tube data (min/max vals) is None. Cannot generate main plots.")
        return
    if not processed_aligned_trajs:
        print("Warning: No processed aligned trajectories provided. Cannot plot trajectories/events.")
        return
    if len(processed_aligned_trajs) != len(all_mapped_events_list):
        print(f"Warning: Mismatch between trajectories ({len(processed_aligned_trajs)}) and events list ({len(all_mapped_events_list)}) length. Event plotting might be incorrect.")
        min_len = min(len(processed_aligned_trajs), len(all_mapped_events_list))
        processed_aligned_trajs = processed_aligned_trajs[:min_len]
        all_mapped_events_list = all_mapped_events_list[:min_len]

    n_timesteps_orig, n_dims_tube = min_vals_orig.shape
    n_features_to_plot = len(feature_names)
    if n_features_to_plot == 0:
        print("Warning: No features specified in config.PLOT_FEATURE_COLS. Nothing to plot.")
        return
    if n_dims_tube != n_features_to_plot:
          print(f"Warning: Tube data dimension ({n_dims_tube}) does not match number of features to plot ({n_features_to_plot}). Check config.PLOT_FEATURE_COLS.")

    time_axis_orig = np.arange(n_timesteps_orig)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = config.PLOT_FONT

    # Setup Figure Grid
    total_plots = n_features_to_plot
    ncols = config.PLOT_GRID_COLUMNS
    nrows = math.ceil(total_plots / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16, y=1.0) # Adjust y maybe
    axs_flat = axs.flatten()

    # Event Markers Config
    event_markers = {
        'start': {'marker': 'o', 'color': 'green', 'size': 7, 'label': 'Start Point'},
        'end': {'marker': 'o', 'color': 'red', 'size': 7, 'label': 'End Point'},
        'state_change': {'marker': '*', 'color': '#FFA500', 'size': 10, 'label': 'State Change'},
        'wp_saved': {'marker': 's', 'color': 'blue', 'size': 7, 'label': 'WP Saved'},
        'gripper_change': {'marker': 'X', 'color': 'magenta', 'size': 7, 'label': 'Gripper Change'}
    }
    plotted_legend_labels = set()

    # Plot Each Feature
    print("  Plotting individual features...")
    for d_idx, feature_name in enumerate(feature_names):
        if d_idx >= len(axs_flat): break
        row, col = d_idx // ncols, d_idx % ncols
        ax = axs[row, col]
        print(f"    Plotting feature: {feature_name}")

        # Check if feature exists in trajectory data (use first traj for check)
        if not processed_aligned_trajs or feature_name not in processed_aligned_trajs[0].columns:
             print(f"    Warning: Feature '{feature_name}' not found in trajectory data. Skipping plot for this feature.")
             ax.text(0.5, 0.5, f"Feature '{feature_name}'\nnot found in data", ha='center', va='center', transform=ax.transAxes, color='red')
             ax.set_ylabel(feature_name)
             continue

        # 1. Plot Tube (if dimensions match plot index)
        if d_idx < n_dims_tube:
            ax.plot(time_axis_orig, max_vals_orig[:, d_idx], color='lightblue', linestyle='-', linewidth=1.0, alpha=0.8, label='Max Bound' if d_idx==0 else "")
            ax.plot(time_axis_orig, min_vals_orig[:, d_idx], color='lightcoral', linestyle='-', linewidth=1.0, alpha=0.8, label='Min Bound' if d_idx==0 else "")
            ax.fill_between(time_axis_orig, min_vals_orig[:, d_idx], max_vals_orig[:, d_idx], color='lightgrey', alpha=0.4, label='Tube Range' if d_idx==0 else "")
            if d_idx==0: plotted_legend_labels.update(['Max Bound', 'Min Bound', 'Tube Range'])

        # 2. Plot Boundaries
        for i, boundary_idx_orig in enumerate(segment_indices_orig):
            if i > 0 and boundary_idx_orig <= n_timesteps_orig:
                label = 'Segment Boundary' if 'Segment Boundary' not in plotted_legend_labels else ""
                ax.axvline(x=boundary_idx_orig, color='black', linestyle='--', linewidth=1.2, label=label)
                if label: plotted_legend_labels.add('Segment Boundary')

        # 3. Plot Trapezoids (if tube dimensions match plot index)
        if d_idx < n_dims_tube:
            for i in range(len(segment_indices_orig) - 1):
                start_idx_orig = segment_indices_orig[i]; end_idx_orig = segment_indices_orig[i+1] - 1
                if end_idx_orig < start_idx_orig or start_idx_orig >= n_timesteps_orig: continue
                end_idx_orig = min(end_idx_orig, n_timesteps_orig - 1)
                segment_time_axis_orig = np.arange(start_idx_orig, end_idx_orig + 1)
                segment_len_points = len(segment_time_axis_orig)
                if segment_len_points <= 1: continue
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

        # 4. Plot Individual Trajectories and Events
        for traj_idx, aligned_traj_df in enumerate(processed_aligned_trajs):
            label_traj = 'Aligned Traj.' if 'Aligned Traj.' not in plotted_legend_labels else ""
            if feature_name in aligned_traj_df.columns:
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
                             except IndexError: continue

        ax.set_ylabel(feature_name)
        ax.grid(True, linestyle=':', which='both', axis='both')

    # Clean up Empty Subplots
    for i in range(total_plots, nrows * ncols):
         if i < len(axs_flat): fig.delaxes(axs_flat[i])

    # Add Combined Legend
    handles, labels = [], []
    for i in range(total_plots):
        if i < len(axs_flat):
             ax = axs_flat[i]
             if ax.has_data():
                h, l = ax.get_legend_handles_labels()
                handles.extend(h); labels.extend(l)
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc='lower center',
                   bbox_to_anchor=(0.5, -0.05 if nrows > 1 else -0.1),
                   ncol=min(len(by_label), 6), fontsize='small')

    # Final Touches
    last_row_plots_indices = range((nrows - 1) * ncols, total_plots)
    for plot_idx in last_row_plots_indices:
        if plot_idx < len(axs_flat):
            row, col = plot_idx // ncols, plot_idx % ncols
            axs[row, col].set_xlabel('Time Step (Aligned Scale)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    print("--- Plotting Complete ---")
    plt.show()


# --- CORRECTED Simple GMM Plot ---
def plot_gmm_mean_cov_simple(results_path=config.CROSS_SECTION_STATS_OUTPUT_PATH,
                             pairs_to_plot=[('tx','ty'), ('tx','tz'), ('ty','tz')]):
    """
    Loads GMM results (expecting K=1) and plots mean position trajectory
    and optionally 2D covariance ellipses for specified feature pairs.
    """
    print(f"--- Generating Simple GMM (K=1) Mean/Cov Plot from {results_path} ---")
    cs_path = Path(results_path)
    results_data = None # Initialize

    if not cs_path.is_file():
        print(f"  Warning: Results file not found at {results_path}. Cannot plot.")
        return

    try:
        with open(cs_path, 'r') as f:
            results_data = yaml.safe_load(f)
    except Exception as e:
        print(f"  Error loading or parsing results file: {e}")
        # results_data remains None

    # Check if loading succeeded and data is valid
    if results_data is None or not isinstance(results_data, list) or not results_data:
        print("  Warning: Results data is empty, invalid, or failed to load. Cannot plot.")
        return

    times = []
    means_pos = {p: [] for p in config.ANALYSIS_POS_COLS}
    means_all = []
    covariances_all = []
    feature_names = []
    valid_cs_count = 0

    # Extract data assuming K=1 structure
    for cs in results_data:
        gmm_params = cs.get('segment_gmm_params')
        if gmm_params and gmm_params.get('n_components_used') == 1:
            if 'means' in gmm_params and 'covariances' in gmm_params and \
               isinstance(gmm_params['means'], list) and len(gmm_params['means']) == 1 and \
               isinstance(gmm_params['covariances'], list) and len(gmm_params['covariances']) == 1:

                mean_vec = gmm_params['means'][0]
                cov_mat = gmm_params['covariances'][0]
                current_feature_names = cs.get('state_feature_names', [])
                if not feature_names: feature_names = current_feature_names

                if len(mean_vec) != len(feature_names) or len(cov_mat) != len(feature_names):
                     # print(f"  Warning: Data dimension mismatch at t={cs.get('time_index', 'N/A')}. Skipping.") # Reduce verbosity
                     continue

                times.append(cs['time_index'])
                means_all.append(mean_vec)
                covariances_all.append(cov_mat)

                for i, fname in enumerate(feature_names):
                    if fname in means_pos:
                        means_pos[fname].append(mean_vec[i])
                valid_cs_count += 1

    if valid_cs_count == 0:
        print("  No cross-sections with valid GMM (K=1) results found.")
        return
    if not feature_names:
        print("  Could not determine feature names from results.")
        return

    # Plot 1: Mean Position Components
    num_pos_cols = len(config.ANALYSIS_POS_COLS)
    if num_pos_cols > 0:
        fig_pos, axs_pos = plt.subplots(num_pos_cols, 1, figsize=(10, 2 + num_pos_cols * 1.5), sharex=True, squeeze=False)
        fig_pos.suptitle('Mean Position Components at Cross-Sections (GMM K=1)')
        axs_pos = axs_pos.flatten()
        plot_successful = False
        for i, pos_col in enumerate(config.ANALYSIS_POS_COLS):
            if i < len(axs_pos) and means_pos.get(pos_col):
                axs_pos[i].plot(times, means_pos[pos_col], marker='o', linestyle='-')
                axs_pos[i].set_ylabel(pos_col); axs_pos[i].grid(True)
                plot_successful = True
            elif i < len(axs_pos):
                 axs_pos[i].text(0.5, 0.5, f"No data for {pos_col}", ha='center', va='center', transform=axs_pos[i].transAxes)
                 axs_pos[i].set_ylabel(pos_col)
        if plot_successful:
            axs_pos[-1].set_xlabel('Time Index (Aligned Scale)')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show(block=False)
        else: plt.close(fig_pos); print("  Skipping mean position plot (no valid data).")

    # Plot 2: Covariance Ellipses
    if not means_all or not covariances_all:
         print("  Skipping ellipse plot (no mean/covariance data extracted).")
         return
    num_pairs = len(pairs_to_plot)
    if num_pairs == 0: return

    ncols_ellipse = min(3, num_pairs); nrows_ellipse = (num_pairs + ncols_ellipse - 1) // ncols_ellipse
    fig_ellipse, axs_ellipse = plt.subplots(nrows_ellipse, ncols_ellipse, figsize=(ncols_ellipse * 5, nrows_ellipse * 5), squeeze=False)
    fig_ellipse.suptitle('Mean +/- Std Dev Ellipses at Cross-Sections (GMM K=1)')
    axs_ellipse = axs_ellipse.flatten()
    print(f"  Available features for ellipse plot: {feature_names}")
    print(f"  Plotting pairs: {pairs_to_plot}")
    plot_idx = 0
    for feat1, feat2 in pairs_to_plot:
        if plot_idx >= len(axs_ellipse): break
        ax = axs_ellipse[plot_idx]
        try:
            idx1 = feature_names.index(feat1); idx2 = feature_names.index(feat2)
        except ValueError:
            print(f"  Warning: Cannot plot pair ('{feat1}', '{feat2}'). Features not found in results list: {feature_names}")
            ax.text(0.5, 0.5, f"Features\n'{feat1}' or '{feat2}'\nnot found", ha='center', va='center', transform=ax.transAxes, color='red')
            plot_idx += 1; continue
        try:
             means_f1 = [m[idx1] for m in means_all]; means_f2 = [m[idx2] for m in means_all]
        except IndexError:
             print(f"  Warning: Index out of bounds extracting means for pair ('{feat1}', '{feat2}'). Skipping.")
             ax.text(0.5, 0.5, f"Data Index Error\nfor '{feat1}' or '{feat2}'", ha='center', va='center', transform=ax.transAxes, color='red')
             plot_idx +=1; continue

        ax.plot(means_f1, means_f2, marker='o', linestyle='-', color='black', label='Mean Trajectory')
        for i, t in enumerate(times):
            if i >= len(covariances_all) or i >= len(means_all): continue
            cov_mat_full_list = covariances_all[i]; mean_vec_list = means_all[i]
            try:
                 cov_mat_full = np.array(cov_mat_full_list); mean_vec = np.array(mean_vec_list)
                 if not (idx1 < cov_mat_full.shape[0] and idx1 < cov_mat_full.shape[1] and \
                         idx2 < cov_mat_full.shape[0] and idx2 < cov_mat_full.shape[1] and \
                         idx1 < len(mean_vec) and idx2 < len(mean_vec)): continue
                 mean_2d = mean_vec[[idx1, idx2]]; cov_2d = cov_mat_full[np.ix_([idx1, idx2], [idx1, idx2])]
            except Exception as e_idx: continue
            try:
                vals, vecs = np.linalg.eigh(cov_2d)
                if np.any(vals < 0):
                    vals[vals < 0] = 1e-9
                    if np.any(vals < 0): continue
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]) # +/- 1 std dev
                ellipse = patches.Ellipse(xy=mean_2d, width=width, height=height, angle=angle, edgecolor='blue', facecolor='lightblue', alpha=0.3, linestyle='--')
                ax.add_patch(ellipse)
                ax.plot(mean_2d[0], mean_2d[1], marker='x', color='red', markersize=5)
            except np.linalg.LinAlgError: pass # Ignore LinAlgError for ellipse
            except Exception as e_el: pass # Ignore other ellipse errors silently

        ax.set_xlabel(feat1); ax.set_ylabel(feat2); ax.grid(True); ax.axis('equal'); plot_idx += 1
    for i in range(plot_idx, len(axs_ellipse)): axs_ellipse[i].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


def print_cross_section_stats(results_path=config.CROSS_SECTION_STATS_OUTPUT_PATH):
    """Loads and prints a summary of the cross-section results."""
    print(f"\n--- Cross-Section Statistics Summary from {results_path} ---")
    cs_path = Path(results_path)
    results_data = None
    if not cs_path.is_file(): print(f"  ERROR: Results file not found at {results_path}."); return
    try:
        with open(cs_path, 'r') as f: results_data = yaml.safe_load(f)
    except Exception as e: print(f"  ERROR: Failed to load or parse results file: {e}"); return
    if results_data is None or not isinstance(results_data, list) or not results_data: print("  No valid cross-section data found in file."); return

    print(f"Found {len(results_data)} cross-section entries.")
    for i, cs in enumerate(results_data):
        time_idx = cs.get('time_index', 'N/A'); cs_type = cs.get('type', 'N/A')
        n_snap = cs.get('num_snapshots', 'N/A'); n_valid_snap = cs.get('num_valid_snapshots', 'N/A')
        rep = cs.get('state_representation_used', 'N/A'); features = cs.get('state_feature_names', [])
        print("-" * 20); print(f"Cross-Section {i+1}: Time={time_idx:.2f}, Type='{cs_type}'")
        print(f"  Snapshots: {n_snap} (Valid: {n_valid_snap})")
        print(f"  Representation: '{rep}' ({len(features)} features)")
        gmm_params = cs.get('segment_gmm_params')
        if gmm_params and gmm_params.get('n_components_used') == 1:
            try:
                mean_vec = np.array(gmm_params['means'][0])
                cov_mat = np.array(gmm_params['covariances'][0])
                # Format numpy array output for cleaner printing
                mean_str = np.array2string(mean_vec, precision=3, suppress_small=True)
                cov_diag_str = np.array2string(np.diag(cov_mat), precision=3, suppress_small=True)
                print(f"  Mean Vector (K=1): {mean_str}")
                print(f"  Covariance Diag (K=1): {cov_diag_str}")
            except Exception as e_print: print(f"  Error formatting GMM K=1 stats: {e_print}")
        elif gmm_params: print(f"  GMM Components Used: {gmm_params.get('n_components_used', 'N/A')}, Converged: {gmm_params.get('converged', 'N/A')}")
        else: print("  GMM Parameters: Not Available")
        pos_min = cs.get('pos_min_bounds', 'N/A'); pos_max = cs.get('pos_max_bounds', 'N/A')
        print(f"  Pos Bounds Min: {pos_min}"); print(f"  Pos Bounds Max: {pos_max}")
    print("-" * 20)

