# -*- coding: utf-8 -*-
"""
Visualization functions for the trajectory segmentation pipeline.
Includes plotting segmentation results, cost curves, orientation deviation,
and GMM waypoint optimization details.
REVISED import logic for orientation functions.
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


# --- Project Modules & Dependencies ---
# ... (Keep existing imports for plt, np, sns, etc.) ...
import traceback # For detailed error printing

try:
    import config # Import configuration
    import utils # Import utils for orientation math
    # Import the flags needed for checks
    from utils import IK_SOLVER_AVAILABLE, SCIPY_AVAILABLE
except ImportError:
    print("FATAL ERROR: config.py or utils.py not found. Ensure they are in the same directory or Python path.")
    IK_SOLVER_AVAILABLE = False; SCIPY_AVAILABLE = False
    class DummyUtils: SCIPY_AVAILABLE = False; IK_SOLVER_AVAILABLE = False; 
    def forward_kinematics(*args, **kwargs): return None, None, None; 
    def log_map_so3(*args, **kwargs): return None; 
    def calculate_geodesic_distance(*args, **kwargs): return np.nan; 
    def exp_map_so3(*args, **kwargs): return None
    utils = DummyUtils()

# --- Define local placeholders for orientation functions ---
# These will be overwritten IF the import from utils succeeds.
_log_map_so3_func = None
_calculate_geodesic_distance_func = None
_exp_map_so3_func = None # Initialize this one too

if SCIPY_AVAILABLE:
    try:
        # Attempt to import directly from utils
        from utils import log_map_so3 as log_map_so3_util
        from utils import calculate_geodesic_distance as calculate_geodesic_distance_util
        from utils import exp_map_so3 as exp_map_so3_util # *** ADD IMPORT FOR EXP MAP ***

        # Assign only if callable (basic check)
        if callable(log_map_so3_util): _log_map_so3_func = log_map_so3_util
        if callable(calculate_geodesic_distance_util): _calculate_geodesic_distance_func = calculate_geodesic_distance_util
        if callable(exp_map_so3_util): _exp_map_so3_func = exp_map_so3_util # *** ASSIGN EXP MAP ***

        # Verify that functions were actually assigned
        if _log_map_so3_func and _calculate_geodesic_distance_func and _exp_map_so3_func:
             print("INFO (visualizer.py): Successfully imported orientation helpers from utils.")
        else:
             print("ERROR (visualizer.py): One or more orientation helpers imported from utils are not callable!")
             # Keep them as None if not callable
             _log_map_so3_func = None; _calculate_geodesic_distance_func = None; _exp_map_so3_func = None

    except ImportError:
        print("ERROR (visualizer.py): Failed to import orientation helpers from utils even though SCIPY_AVAILABLE=True.")
        # Keep them as None
        _log_map_so3_func = None; _calculate_geodesic_distance_func = None; _exp_map_so3_func = None
else:
     print("INFO (visualizer.py): Scipy not available via utils. Orientation functions unavailable.")


# # --- Project Modules ---
# try:
#     import config # Import configuration
#     import utils # Import utils for orientation math
#     # Import the flags needed for checks
#     from utils import IK_SOLVER_AVAILABLE, SCIPY_AVAILABLE
# except ImportError:
#     print("FATAL ERROR: config.py or utils.py not found. Ensure they are in the same directory or Python path.")
#     IK_SOLVER_AVAILABLE = False; SCIPY_AVAILABLE = False
#     class DummyUtils: SCIPY_AVAILABLE = False; IK_SOLVER_AVAILABLE = False; def forward_kinematics(*args, **kwargs): return None, None, None; def log_map_so3(*args, **kwargs): return None; def calculate_geodesic_distance(*args, **kwargs): return np.nan; def exp_map_so3(*args, **kwargs): return None
#     utils = DummyUtils()

# # --- Define local placeholders for orientation functions ---
# # These will be overwritten IF the import from utils succeeds.
# _log_map_so3_func = None
# _calculate_geodesic_distance_func = None
# _exp_map_so3_func = None

# if SCIPY_AVAILABLE:
#     try:
#         # Attempt to import directly from utils
#         from utils import log_map_so3, calculate_geodesic_distance, exp_map_so3
#         # Check if they are callable and assign to local variables
#         if callable(log_map_so3): _log_map_so3_func = log_map_so3
#         if callable(calculate_geodesic_distance): _calculate_geodesic_distance_func = calculate_geodesic_distance
#         if callable(exp_map_so3): _exp_map_so3_func = exp_map_so3
#         print("INFO (visualizer.py): Successfully imported orientation helpers from utils.")
#         if _log_map_so3_func is None or _calculate_geodesic_distance_func is None or _exp_map_so3_func is None:
#              print("ERROR (visualizer.py): One or more orientation helpers imported from utils are not callable!")
#     except ImportError:
#         print("ERROR (visualizer.py): Failed to import orientation helpers from utils even though SCIPY_AVAILABLE=True.")
# else:
#     print("INFO (visualizer.py): Scipy not available via utils. Orientation plots will be skipped.")


# --- Plotting Functions ---

def _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0, n_points=50):
    """ Generates x, y points for a 2D confidence ellipse for Plotly. """
    # ... (Keep existing implementation - unchanged) ...
    try:
        vals, vecs = np.linalg.eigh(cov_2d)
        if np.any(vals <= 1e-9): return None, None
        angle = np.arctan2(*vecs[:, 0][::-1])
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
        t = np.linspace(0, 2 * np.pi, n_points)
        xs = width / 2 * np.cos(t); ys = height / 2 * np.sin(t)
        R_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = R_mat @ np.vstack([xs, ys])
        x_points = points[0, :] + mean_2d[0]; y_points = points[1, :] + mean_2d[1]
        return x_points, y_points
    except np.linalg.LinAlgError: return None, None
    except Exception: return None, None


def plot_gmm_cross_section_interactive(gmm_params, snapshot_data,
                                       feature_names=None, # Default to None
                                       title="Interactive GMM Cross-Section Visualization"):
    """ Creates an interactive Plotly plot showing pairwise projections of GMM components. """
    # ... (Keep existing implementation - unchanged) ...
    if feature_names is None: feature_names = ['px', 'py', 'pz', 'vlog_x', 'vlog_y', 'vlog_z']
    print(f"--- Generating Interactive GMM Plot: {title} ---")
    if gmm_params is None: print("  Error: gmm_params is None."); return
    if snapshot_data is None or snapshot_data.ndim != 2 or snapshot_data.shape[0] == 0: print("  Error: snapshot_data is invalid."); return
    try:
        weights = np.array(gmm_params['weights']); means = np.array(gmm_params['means']); covariances = np.array(gmm_params['covariances'])
        n_components = gmm_params.get('n_components_used', len(weights)); n_features = len(feature_names)
        if means.shape != (n_components, n_features) or covariances.shape != (n_components, n_features, n_features): print(f"  Error: GMM parameter dimensions mismatch."); return
        if snapshot_data.shape[1] != n_features: print(f"  Error: snapshot_data dimension mismatch."); return
        fig = go.Figure(); dimension_indices = list(range(n_features)); pairwise_combinations = list(combinations(dimension_indices, 2))
        component_colors = sns.color_palette("viridis", n_components).as_hex()
        print(f"  Generating traces for {len(pairwise_combinations)} dimension pairs...")
        for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
            visible = (i == 0)
            fig.add_trace(go.Scatter(x=snapshot_data[:, dim_x_idx], y=snapshot_data[:, dim_y_idx], mode='markers', marker=dict(color='grey', size=5, opacity=0.7), name='Snapshots', visible=visible, showlegend=(i==0)))
            for k in range(n_components):
                mean_2d = means[k, [dim_x_idx, dim_y_idx]]; cov_2d = covariances[k, np.ix_([dim_x_idx, dim_y_idx], [dim_x_idx, dim_y_idx])]
                x_ellipse, y_ellipse = _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0)
                if x_ellipse is not None: fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', line=dict(color=component_colors[k], width=2), fill='toself', fillcolor=component_colors[k], opacity=0.3 + 0.6 * weights[k], name=f'Comp {k+1} (w={weights[k]:.2f})', visible=visible, showlegend=(i==0)))
                else: fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=f'Comp {k+1} Error', visible=False))
        buttons = []
        for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
            visibility_mask = [False] * len(fig.data); start_idx = i * (1 + n_components); end_idx = start_idx + (1 + n_components)
            for trace_idx in range(start_idx, end_idx):
                 if trace_idx < len(visibility_mask) and fig.data[trace_idx].x is not None and len(fig.data[trace_idx].x) > 0 and fig.data[trace_idx].x[0] is not None: visibility_mask[trace_idx] = True
            buttons.append(dict(label=f"{feature_names[dim_x_idx]} vs {feature_names[dim_y_idx]}", method="update", args=[{"visible": visibility_mask}, {"xaxis.title": feature_names[dim_x_idx], "yaxis.title": feature_names[dim_y_idx]}]))
        fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, xanchor="left", y=1.15, yanchor="top")], title=title, hovermode="closest")
        initial_dim_x_idx, initial_dim_y_idx = pairwise_combinations[0]
        fig.update_layout(xaxis_title=feature_names[initial_dim_x_idx], yaxis_title=feature_names[initial_dim_y_idx], legend_title_text="Components")
        print("  Displaying interactive plot...")
        fig.show()
    except Exception as e: print(f"  Error generating interactive GMM plot: {e}"); traceback.print_exc()


def plot_cost_vs_segments(raw_costs, max_segments, optimal_num_segments, lambda_penalty):
    """Plots the raw and penalized segmentation cost vs. number of segments."""
    # ... (Keep existing implementation - unchanged) ...
    plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams['font.family'] = config.PLOT_FONT if hasattr(config, 'PLOT_FONT') else 'sans-serif'
    num_segments_axis = np.arange(1, max_segments + 1); plot_costs_raw = np.full(max_segments, np.inf); valid_len = min(len(raw_costs), max_segments)
    if valid_len > 0: plot_costs_raw[:valid_len] = raw_costs[:valid_len]
    penalized_costs = plot_costs_raw + lambda_penalty * num_segments_axis
    fig, ax1 = plt.subplots(figsize=(12, 6)); color_raw = 'tab:blue'; ax1.set_xlabel('Number of Segments'); ax1.set_ylabel('Total Raw Segmentation Cost', color=color_raw)
    valid_idx_raw = np.isfinite(plot_costs_raw)
    if np.any(valid_idx_raw): ax1.plot(num_segments_axis[valid_idx_raw], plot_costs_raw[valid_idx_raw], marker='o', linestyle='-', color=color_raw, label='Raw Cost'); ax1.tick_params(axis='y', labelcolor=color_raw); min_finite_cost_raw = np.min(plot_costs_raw[valid_idx_raw]); ax1.set_ylim(bottom=min(0, min_finite_cost_raw * 0.9))
    else: ax1.text(0.5, 0.5, 'No finite raw costs', ha='center', va='center', transform=ax1.transAxes)
    ax2 = ax1.twinx(); color_penalized = 'tab:red'; ax2.set_ylabel('Total Penalized Cost (Raw + λ*N)', color=color_penalized)
    valid_idx_pen = np.isfinite(penalized_costs)
    if np.any(valid_idx_pen): ax2.plot(num_segments_axis[valid_idx_pen], penalized_costs[valid_idx_pen], marker='x', linestyle='--', color=color_penalized, label='Penalized Cost'); ax2.tick_params(axis='y', labelcolor=color_penalized); min_finite_cost_pen = np.min(penalized_costs[valid_idx_pen]); max_finite_cost_pen = np.max(penalized_costs[valid_idx_pen]); padding = (max_finite_cost_pen - min_finite_cost_pen) * 0.05 if max_finite_cost_pen > min_finite_cost_pen else 1.0; ax2.set_ylim(bottom=min(0, min_finite_cost_pen) - padding, top=max_finite_cost_pen + padding)
    else: ax2.text(0.5, 0.4, 'No finite penalized costs', ha='center', va='center', transform=ax2.transAxes)
    if 1 <= optimal_num_segments <= max_segments: optimal_raw_cost = plot_costs_raw[optimal_num_segments - 1]; optimal_pen_cost = penalized_costs[optimal_num_segments - 1];
    if np.isfinite(optimal_raw_cost): ax1.scatter(optimal_num_segments, optimal_raw_cost, color='green', s=150, zorder=5, marker='*', label=f'Optimal ({optimal_num_segments}) - Raw')
    if np.isfinite(optimal_pen_cost): ax2.scatter(optimal_num_segments, optimal_pen_cost, color='magenta', s=150, zorder=5, marker='P', label=f'Optimal ({optimal_num_segments}) - Penalized')
    plt.title(f'Segmentation Cost vs. Number of Segments (λ={config.LAMBDA_PENALTY if hasattr(config, "LAMBDA_PENALTY") else "N/A"})')
    ax1.set_xticks(np.arange(1, max_segments + 1)); ax1.grid(True, linestyle=':', which='major', axis='x'); ax2.grid(False)
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95)); fig.tight_layout(rect=[0, 0, 0.9, 1])
    # plt.show()


# --- MODIFIED: Plot Interface Frame Pose Components ---
def plot_interface_frame_pose_components(
    tcp_pos_path_interface_np: np.ndarray,
    tcp_rot_matrices_interface_list: list, # List of 3x3 rotation matrices
    segment_boundary_indices: list,
    segment_stats_list: list, # List of dicts (can be empty)
    segment_start_end_pose_components: list, # List of dicts with start/end components
    title="TCP Pose Components vs Waypoint Index (Interface Frame)",
    log_map_feature_names=['vlog_x', 'vlog_y', 'vlog_z'] # Names for log map components
):
    """
    Plots the 3 position components (x, y, z) and the 3 orientation log map
    vector components of the TCP trajectory in the Interface frame against waypoint index.
    Overlays segment boundaries. Adds checks for rotation matrix validity.
    """
    print("--- Generating Interface Frame Pose Component Plot (Position + Log Map) ---")
    if tcp_pos_path_interface_np is None or len(tcp_pos_path_interface_np) == 0: print("  Warning: No TCP position data provided."); return
    if tcp_rot_matrices_interface_list is None or len(tcp_rot_matrices_interface_list) != len(tcp_pos_path_interface_np): print("  Warning: Rotation matrix list mismatch."); tcp_rot_matrices_interface_list = None

    # Use the function handle stored during import checks
    log_map_available = callable(log_map_so3_util)
    if not log_map_available: print("  Warning: log_map_so3 function unavailable. Cannot plot orientation."); tcp_rot_matrices_interface_list = None

    num_waypoints = len(tcp_pos_path_interface_np); time_axis = np.arange(num_waypoints)
    log_map_trajectory = None; log_map_valid = False
    if tcp_rot_matrices_interface_list is not None and log_map_available:
        print("  Calculating orientation log map trajectory with validity checks...")
        log_map_list = []; calculation_errors = 0; invalid_matrix_errors = 0; valid_rots_found = 0
        for idx, R_mat in enumerate(tcp_rot_matrices_interface_list):
            v_log = None; is_valid_rot = False
            if R_mat is not None and isinstance(R_mat, np.ndarray) and R_mat.shape == (3,3) and not np.isnan(R_mat).any() and not np.isinf(R_mat).any():
                try:
                    # Calculate determinant and check orthogonality
                    det = np.linalg.det(R_mat)
                    is_ortho = np.allclose(R_mat @ R_mat.T, np.identity(3), atol=1e-4)

                    # Check if both conditions for a valid rotation matrix are met (within tolerance)
                    if abs(det - 1.0) < 1e-4 and is_ortho:
                        is_valid_rot = True
                    else:
                        # Matrix is not a valid rotation (det != 1 or not orthogonal)
                        invalid_matrix_errors += 1

                except np.linalg.LinAlgError:
                    # Catch errors during determinant calculation or matrix multiplication
                    # (e.g., if R_mat contained non-finite values)
                    invalid_matrix_errors += 1
            else: invalid_matrix_errors += 1
            if is_valid_rot:
                v_log = log_map_so3_util(R_mat) # Use the function handle
                if v_log is None: calculation_errors += 1
                else: valid_rots_found += 1
            log_map_list.append(v_log if v_log is not None else np.full(3, np.nan))
        if invalid_matrix_errors > 0: print(f"  Warning: Found {invalid_matrix_errors} invalid rotation matrices.")
        if calculation_errors > 0: print(f"  Warning: Failed log map calculation for {calculation_errors} valid rotation matrices.")
        if valid_rots_found > 0: log_map_trajectory = np.array(log_map_list); log_map_valid = True; print(f"  Log map calculation succeeded for {valid_rots_found}/{num_waypoints} waypoints.")
        else: print("  ERROR: Log map calculation failed for ALL waypoints or no valid rotation matrices found.")

    n_features_pos = 3; n_features_rot = 3 if log_map_valid else 0; n_features_to_plot = n_features_pos + n_features_rot
    if n_features_to_plot == 3: ncols = 3; nrows = 1
    elif n_features_to_plot > 0: ncols = 3; nrows = 2
    else: print("  No valid data to plot."); return
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, squeeze=False); fig.suptitle(title, fontsize=14, y=1.0); axs_flat = axs.flatten()
    position_feature_names = ['tx', 'ty', 'tz']; all_feature_names = position_feature_names + (log_map_feature_names if log_map_valid else [])
    all_data = np.hstack((tcp_pos_path_interface_np, log_map_trajectory)) if log_map_valid else tcp_pos_path_interface_np
    plotted_legend_labels = set()
    for d_idx, feature_name in enumerate(all_feature_names):
        if d_idx >= len(axs_flat): break
        ax = axs_flat[d_idx]; print(f"  Plotting component: {feature_name}"); component_data = all_data[:, d_idx]; valid_data_mask = ~np.isnan(component_data)
        label_comp = 'TCP Path Component' if 'TCP Path Component' not in plotted_legend_labels else ""
        if np.any(valid_data_mask): ax.plot(time_axis[valid_data_mask], component_data[valid_data_mask], color='black', linewidth=1.0, label=label_comp);
        if label_comp: plotted_legend_labels.add('TCP Path Component')
        else: print(f"    Warning: No valid data for component {feature_name}"); ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes, color='red')
        for b_idx in segment_boundary_indices:
            if 0 < b_idx < num_waypoints: label_bound = "";
            if 'Segment Boundary' not in plotted_legend_labels: label_bound = 'Segment Boundary'
            ax.axvline(x=b_idx, color='grey', linestyle='--', linewidth=1.0, label=label_bound);
            if label_bound: plotted_legend_labels.add('Segment Boundary')
        ax.set_ylabel(feature_name); ax.grid(True, linestyle=':', which='both', axis='both')
    for i in range(n_features_to_plot, nrows * ncols):
        if i < len(axs_flat): fig.delaxes(axs_flat[i])
    last_row_plots_indices = range((nrows - 1) * ncols, n_features_to_plot)
    for plot_idx in last_row_plots_indices:
         if plot_idx < len(axs_flat): axs_flat[plot_idx].set_xlabel('Waypoint Index')
    if plotted_legend_labels: handles, labels = axs_flat[0].get_legend_handles_labels(); by_label = dict(zip(labels, handles)); fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05 if nrows > 1 else -0.1), ncol=min(len(by_label), 6), fontsize='small')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])


# --- 3D Visualization Class ---
class Visualizer3D:
    """ Creates and manages a 3D plot using Matplotlib for robot trajectories and environment. """
    def __init__(self, title="3D Robot Motion"):
        # ... (Keep existing implementation - unchanged) ...
        try: from mpl_toolkits.mplot3d import Axes3D; self._Axes3D = Axes3D
        except ImportError: print("ERROR: mpl_toolkits.mplot3d not found."); self._Axes3D = None
        self.fig = plt.figure(figsize=(10, 8))
        if self._Axes3D: self.ax = self.fig.add_subplot(111, projection='3d')
        else: self.ax = self.fig.add_subplot(111); print("Warning: Creating 2D plot fallback.")
        self.ax.set_title(title); self.ax.set_xlabel("X (Base Frame)"); self.ax.set_ylabel("Y (Base Frame)")
        if self._Axes3D: self.ax.set_zlabel("Z (Base Frame)")
        self.ax.grid(True)
        limits = getattr(config, 'VIS_PLOT_LIMITS', ([-0.5, 0.5], [-0.5, 0.5], [0, 1.0]))
        self.ax.set_xlim(limits[0]); self.ax.set_ylim(limits[1])
        if self._Axes3D: self.ax.set_zlim(limits[2]); self.ax.set_box_aspect([np.ptp(lim) for lim in limits])
        self.plot_frame(np.identity(4), size=config.VIS_BASE_FRAME_SIZE, label="Base Frame")

    def plot_frame(self, T, size=0.1, label=""):
        """ Plots a 3D coordinate frame represented by a 4x4 transform matrix T. """
        # ... (Keep existing implementation - unchanged) ...
        if not self._Axes3D: return
        origin = T[:3, 3]; x_axis = T[:3, 0]; y_axis = T[:3, 1]; z_axis = T[:3, 2]
        colors = ['red', 'green', 'blue']; axes = [x_axis, y_axis, z_axis]; axis_labels = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes): self.ax.quiver(origin[0], origin[1], origin[2], axis[0], axis[1], axis[2], length=size, color=colors[i], normalize=False, label=f"{label} {axis_labels[i]}" if i == 0 and label else "")

    def plot_robot_config(self, q, color='grey', linewidth=2, style='-', label="Robot Config"):
        """ Plots the robot links for a given configuration q. """
        # ... (Keep existing implementation - unchanged) ...
        if not IK_SOLVER_AVAILABLE: print("Warning: Cannot plot robot config, IK Solver unavailable."); return
        try:
            joint_positions, _, _ = utils.forward_kinematics(q)
            if joint_positions:
                xs = [p[0] for p in joint_positions]; ys = [p[1] for p in joint_positions]
                if self._Axes3D: zs = [p[2] for p in joint_positions]; self.ax.plot(xs, ys, zs, color=color, linewidth=linewidth, linestyle=style, label=label, marker='o', markersize=4)
                else: self.ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=style, label=label, marker='o', markersize=4)
        except Exception as e: print(f"Warning: Failed to plot robot config: {e}")

    def plot_trajectory(self, path_cspace_np, num_snapshots=10, color='purple', style='-', label_prefix="Path", plot_tcp_frames=True, linewidth=1.5):
        """ Plots the C-space path in 3D task space with optional snapshots. """
        # ... (Keep existing implementation - unchanged) ...
        if path_cspace_np is None or len(path_cspace_np) == 0: return
        if not IK_SOLVER_AVAILABLE: print("Warning: Cannot plot trajectory, IK Solver unavailable."); return
        tcp_path_taskspace = []; snapshot_indices = np.linspace(0, len(path_cspace_np) - 1, num_snapshots, dtype=int) if num_snapshots > 0 else []
        fk_errors = 0
        for i, q in enumerate(path_cspace_np):
            try:
                _, T_tcp, _ = utils.forward_kinematics(q)
                if T_tcp is not None:
                    tcp_path_taskspace.append(T_tcp[:3, 3])
                    if i in snapshot_indices: label = f"{label_prefix} Snap {list(snapshot_indices).index(i)+1}" if i == snapshot_indices[0] else ""; self.plot_robot_config(q, color=color, linewidth=1, style=':', label=label);
                    if plot_tcp_frames and i in snapshot_indices: self.plot_frame(T_tcp, size=config.VIS_TCP_FRAME_SIZE, label="")
                else: fk_errors += 1; tcp_path_taskspace.append([np.nan]*3)
            except Exception: fk_errors += 1; tcp_path_taskspace.append([np.nan]*3)
        tcp_path_np = np.array(tcp_path_taskspace); valid_tcp_points = ~np.isnan(tcp_path_np).any(axis=1)
        if np.any(valid_tcp_points):
             valid_path = tcp_path_np[valid_tcp_points]
             path_label = label_prefix if label_prefix else f"TCP Path {random.randint(100,999)}" # Ensure unique label
             if self._Axes3D:
                 self.ax.plot(valid_path[:, 0], valid_path[:, 1], valid_path[:, 2], color=color, linestyle=style, linewidth=linewidth, label=path_label)
                 if len(valid_path) > 0: self.ax.scatter(valid_path[0, 0], valid_path[0, 1], valid_path[0, 2], color=color, marker='>', s=50, label=f"_{path_label}_Start"); self.ax.scatter(valid_path[-1, 0], valid_path[-1, 1], valid_path[-1, 2], color=color, marker='<', s=50, label=f"_{path_label}_End")
             else:
                 self.ax.plot(valid_path[:, 0], valid_path[:, 1], color=color, linestyle=style, linewidth=linewidth, label=path_label)
                 if len(valid_path) > 0: self.ax.scatter(valid_path[0, 0], valid_path[0, 1], color=color, marker='>', s=50, label=f"_{path_label}_Start"); self.ax.scatter(valid_path[-1, 0], valid_path[-1, 1], color=color, marker='<', s=50, label=f"_{path_label}_End")

    def plot_task_space_set(self, set_data, color='grey', alpha=0.1, label="Target Set"):
        """ Plots the 3D bounding box for a task space set (Traditional approach). """
        # ... (Keep existing implementation - unchanged) ...
        if not self._Axes3D: return
        if set_data is None or 'pos_bounds' not in set_data: return
        bounds = np.array(set_data['pos_bounds']);
        if bounds.shape != (3, 2): return
        xmin, xmax = bounds[0]; ymin, ymax = bounds[1]; zmin, zmax = bounds[2]
        corners = np.array([[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin], [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]])
        faces = [[corners[0], corners[1], corners[2], corners[3]], [corners[4], corners[5], corners[6], corners[7]], [corners[0], corners[1], corners[5], corners[4]], [corners[2], corners[3], corners[7], corners[6]], [corners[1], corners[2], corners[6], corners[5]], [corners[3], corners[0], corners[4], corners[7]]]
        try: from mpl_toolkits.mplot3d.art3d import Poly3DCollection; poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha); self.ax.add_collection3d(poly3d); proxy = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, ec='k', label=label); current_handles, current_labels = self.ax.get_legend_handles_labels(); self.ax.legend(current_handles + [proxy], current_labels + [label], loc='best')
        except ImportError: print("Warning: Could not import Poly3DCollection.")
        except Exception as e: print(f"Warning: Failed to plot task space set bounds: {e}")

    def plot_tcp_waypoint(self, q, color='red', marker='o', size=50, label="Waypoint"):
        """ Plots a marker at the TCP position for a given configuration q. """
        # ... (Keep existing implementation - unchanged) ...
        if not IK_SOLVER_AVAILABLE: print("Warning: Cannot plot TCP waypoint, IK Solver unavailable."); return
        try:
            _, T_tcp, _ = utils.forward_kinematics(q)
            if T_tcp is not None:
                pos = T_tcp[:3, 3]
                if self._Axes3D: self.ax.scatter(pos[0], pos[1], pos[2], color=color, marker=marker, s=size, label=label, depthshade=True)
                else: self.ax.scatter(pos[0], pos[1], color=color, marker=marker, s=size, label=label)
        except Exception as e: print(f"Warning: Failed to plot TCP waypoint: {e}")

    # --- Keep PSO/GD plotting functions ---
    def plot_pso_final_swarm(self, swarm_positions_T, best_pos_xyz=None, **kwargs):
         """ Plots the final positions of the PSO swarm (task space). """
         # ... (Keep existing implementation) ...
         if swarm_positions_T is None or len(swarm_positions_T) == 0: return
         positions = np.array([T[:3, 3] for T in swarm_positions_T if T is not None]);
         if len(positions) == 0: return
         size = kwargs.get('s', 10)
         if self._Axes3D: self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], **kwargs);
         if best_pos_xyz is not None: self.ax.scatter(best_pos_xyz[0], best_pos_xyz[1], best_pos_xyz[2], color='lime', marker='*', s=size*5, label='PSO Best', depthshade=True, edgecolors='black')
         else: self.ax.scatter(positions[:, 0], positions[:, 1], **kwargs);
         if best_pos_xyz is not None: self.ax.scatter(best_pos_xyz[0], best_pos_xyz[1], color='lime', marker='*', s=size*5, label='PSO Best', edgecolors='black')

    def plot_gd_progress(self, pose_history_T, cost_history, **kwargs):
         """ Plots the task space path taken by GD. """
         # ... (Keep existing implementation) ...
         if pose_history_T is None or len(pose_history_T) == 0: return
         positions = np.array([T[:3, 3] for T in pose_history_T if T is not None]);
         if len(positions) == 0: return
         size = kwargs.pop('s', 20); marker = kwargs.pop('marker', '>')
         if self._Axes3D: self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker=marker, markersize=size/5, **kwargs); self.ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=kwargs.get('color','cyan'), marker='o', s=size*2, label='GD Start'); self.ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='lime', marker='*', s=size*5, label='GD End', depthshade=True, edgecolors='black')
         else: self.ax.plot(positions[:, 0], positions[:, 1], marker=marker, markersize=size/5, **kwargs); self.ax.scatter(positions[0, 0], positions[0, 1], color=kwargs.get('color','cyan'), marker='o', s=size*2, label='GD Start'); self.ax.scatter(positions[-1, 0], positions[-1, 1], color='lime', marker='*', s=size*5, label='GD End', edgecolors='black')

    def add_legend(self, **kwargs):
        """ Adds a legend to the plot, handling potential duplicate labels. """
        # ... (Keep existing implementation - unchanged) ...
        handles, labels = self.ax.get_legend_handles_labels()
        filtered_labels_handles = [(l, h) for l, h in zip(labels, handles) if not l.startswith('_')]
        by_label = dict(filtered_labels_handles)
        if by_label: self.ax.legend(by_label.values(), by_label.keys(), **kwargs)

# --- Standalone Plotting Functions (Matplotlib) ---
def plot_task_space_trajectory(tcp_path_np, title="TCP Task Space Trajectory", interface_frame_id="Interface"):
    """ Plots the 3D position trajectory of the TCP in a specific frame. """
    # ... (Keep existing implementation - unchanged) ...
    if tcp_path_np is None or len(tcp_path_np) < 2: print("Warning: Not enough data points to plot task space trajectory."); return
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(tcp_path_np[:, 0], tcp_path_np[:, 1], tcp_path_np[:, 2], marker='.', linestyle='-', label='TCP Path')
    ax.scatter(tcp_path_np[0, 0], tcp_path_np[0, 1], tcp_path_np[0, 2], color='green', s=100, label='Start', depthshade=False)
    ax.scatter(tcp_path_np[-1, 0], tcp_path_np[-1, 1], tcp_path_np[-1, 2], color='red', s=100, label='End', depthshade=False)
    ax.set_xlabel(f"X ({interface_frame_id} Frame)"); ax.set_ylabel(f"Y ({interface_frame_id} Frame)"); ax.set_zlabel(f"Z ({interface_frame_id} Frame)")
    ax.set_title(title); ax.legend(); ax.grid(True); ax.set_box_aspect([np.ptp(tcp_path_np[:, i]) for i in range(3)])
    plt.tight_layout()
    # plt.show()

def plot_cspace_path(path_np, title="C-Space Path"):
    """ Plots the joint space trajectory. """
    # ... (Keep existing implementation - unchanged) ...
    if path_np is None or len(path_np) < 2: print("Warning: Not enough data points to plot C-space path."); return
    num_joints = path_np.shape[1]; fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True); fig.suptitle(title, fontsize=14)
    time_steps = np.arange(len(path_np))
    for i in range(num_joints):
        ax = axs[i] if num_joints > 1 else axs
        ax.plot(time_steps, np.degrees(path_np[:, i]), label=f'Joint {i+1}')
        if hasattr(config, 'JOINT_LIMITS_MIN') and hasattr(config, 'JOINT_LIMITS_MAX'): min_lim_deg = np.degrees(config.JOINT_LIMITS_MIN[i]); max_lim_deg = np.degrees(config.JOINT_LIMITS_MAX[i]); ax.axhline(min_lim_deg, color='red', linestyle='--', linewidth=0.8, label='Limits' if i==0 else ""); ax.axhline(max_lim_deg, color='red', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'Joint {i+1} (deg)'); ax.grid(True, linestyle=':');
        if i == 0: ax.legend(loc='best', fontsize='small')
    axs[-1].set_xlabel('Waypoint Index'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

# --- NEW: Plot GMM Optimization Details ---
def plot_gmm_optimization_details(opt_info: dict, gmm_params: dict, branch_tuple: tuple, boundary_index: int, H_B_I: np.ndarray):
    """
    Plots the optimization progress for a single GMM waypoint optimization run.
    Shows cost history and 2D projections of the 6D state path.
    """
    # Check if exp_map function is available
    exp_map_available = callable(_exp_map_so3_func)
    if not exp_map_available:
         print(f"Warning: Cannot plot GMM optimization details for Boundary {boundary_index+1}, Branch {branch_tuple}. Exp map function unavailable.")
         print(f"Warning: Cannot plot GMM optimization details for Boundary {boundary_index+1}, Branch {branch_tuple}. Exp map function unavailable.")
         return

    if not opt_info: print("Warning: No optimization info provided to plot."); return
    if not gmm_params: print("Warning: No GMM params provided to plot."); return

    algo = opt_info.get('algorithm', 'UNKNOWN')
    cost_history = opt_info.get('cost_history', [])
    pos_history_6d = opt_info.get('pos_history', []) # List of 6D states [p_I, v_log_I] for GD
    swarm_final_6d = opt_info.get('final_swarm_positions_6d') # Array (n_particles, 6) for PSO
    initial_guess_6d = opt_info.get('initial_guess')
    final_cost = opt_info.get('final_cost', np.inf)

    print(f"\n--- Plotting Optimization Details for Boundary {boundary_index+1}, Branch {branch_tuple} ({algo}) ---")

    has_cost_history = bool(cost_history)
    has_path_history = bool(pos_history_6d) or (swarm_final_6d is not None)
    n_plots = (1 if has_cost_history else 0) + (3 if has_path_history else 0)
    if n_plots == 0: print("  No history data available to plot."); return

    ncols = min(n_plots, 2); nrows = math.ceil(n_plots / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False); axs_flat = axs.flatten(); plot_idx = 0

    if has_cost_history:
        ax = axs_flat[plot_idx]; plot_idx += 1; iterations = np.arange(len(cost_history))
        ax.plot(iterations, cost_history, marker='.', linestyle='-'); ax.set_xlabel("Iteration / Step"); ax.set_ylabel("Combined Cost"); ax.set_title(f"Cost History (Final: {final_cost:.4f})"); ax.grid(True)
        if len(cost_history) > 1: ax.set_yscale('log')

    if has_path_history:
        feature_names = ['px_I', 'py_I', 'pz_I', 'vlog_x_I', 'vlog_y_I', 'vlog_z_I']
        projections = [(0, 1), (2, 3), (4, 5)] # XY pos, Z/VlogX, VlogY/VlogZ
        gmm_means = gmm_params.get('means'); gmm_covs = gmm_params.get('covariances'); gmm_weights = gmm_params.get('weights')
        n_components = len(gmm_weights) if gmm_weights is not None else 0
        component_colors = sns.color_palette("coolwarm", n_components).as_hex() if n_components > 0 else []

        for dim_x_idx, dim_y_idx in projections:
            if plot_idx >= len(axs_flat): break
            ax = axs_flat[plot_idx]; plot_idx += 1
            ax.set_xlabel(feature_names[dim_x_idx]); ax.set_ylabel(feature_names[dim_y_idx]); ax.set_title(f"{feature_names[dim_x_idx]} vs {feature_names[dim_y_idx]}")
            ax.grid(True); ax.set_aspect('equal', adjustable='box')

            if gmm_means is not None and gmm_covs is not None and n_components > 0:
                 for k in range(n_components):
                      mean_2d = gmm_means[k, [dim_x_idx, dim_y_idx]]; cov_2d = gmm_covs[k, np.ix_([dim_x_idx, dim_y_idx], [dim_x_idx, dim_y_idx])]
                      x_ellipse, y_ellipse = _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0)
                      if x_ellipse is not None: ax.plot(x_ellipse, y_ellipse, color=component_colors[k], linestyle='--', alpha=0.6 * gmm_weights[k] + 0.1, label=f"Comp {k}" if dim_x_idx==0 else "")

            if pos_history_6d: # GD History
                path = np.array(pos_history_6d)
                ax.plot(path[:, dim_x_idx], path[:, dim_y_idx], marker='.', linestyle='-', color='black', label="GD Path")
                ax.scatter(path[0, dim_x_idx], path[0, dim_y_idx], color='lime', s=100, marker='o', label="Start", zorder=10)
                ax.scatter(path[-1, dim_x_idx], path[-1, dim_y_idx], color='red', s=100, marker='x', label="End", zorder=10)
            elif swarm_final_6d is not None: # PSO Final Swarm
                ax.scatter(swarm_final_6d[:, dim_x_idx], swarm_final_6d[:, dim_y_idx], marker='.', color='grey', alpha=0.5, label="PSO Particles")
                if initial_guess_6d is not None: ax.scatter(initial_guess_6d[dim_x_idx], initial_guess_6d[dim_y_idx], color='blue', s=100, marker='o', label="Initial Guess", zorder=9)
                # Find and plot the best particle position if available
                if 'final_cost' in opt_info and np.isfinite(opt_info['final_cost']) and 'final_best_pos_6d' in opt_info:
                     best_pos_6d = opt_info['final_best_pos_6d'] # Assuming this is stored in opt_info
                     ax.scatter(best_pos_6d[dim_x_idx], best_pos_6d[dim_y_idx], color='red', s=100, marker='x', label="Best Particle", zorder=10)


            if dim_x_idx == 0: ax.legend(loc='best', fontsize='small')

    for i in range(plot_idx, nrows * ncols):
        if i < len(axs_flat): fig.delaxes(axs_flat[i])
    fig.suptitle(f"GMM Optimization: Boundary {boundary_index+1}, Branch {branch_tuple} ({algo})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()


# --- Keep other standalone plotting functions ---
def plot_task_space_trajectory(tcp_path_np, title="TCP Task Space Trajectory", interface_frame_id="Interface"):
    """ Plots the 3D position trajectory of the TCP in a specific frame. """
    # ... (Keep existing implementation - unchanged) ...
    if tcp_path_np is None or len(tcp_path_np) < 2: print("Warning: Not enough data points to plot task space trajectory."); return
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(tcp_path_np[:, 0], tcp_path_np[:, 1], tcp_path_np[:, 2], marker='.', linestyle='-', label='TCP Path')
    ax.scatter(tcp_path_np[0, 0], tcp_path_np[0, 1], tcp_path_np[0, 2], color='green', s=100, label='Start', depthshade=False)
    ax.scatter(tcp_path_np[-1, 0], tcp_path_np[-1, 1], tcp_path_np[-1, 2], color='red', s=100, label='End', depthshade=False)
    ax.set_xlabel(f"X ({interface_frame_id} Frame)"); ax.set_ylabel(f"Y ({interface_frame_id} Frame)"); ax.set_zlabel(f"Z ({interface_frame_id} Frame)")
    ax.set_title(title); ax.legend(); ax.grid(True); ax.set_box_aspect([np.ptp(tcp_path_np[:, i]) for i in range(3)])
    plt.tight_layout()
    # plt.show()

def plot_cspace_path(path_np, title="C-Space Path"):
    """ Plots the joint space trajectory. """
    # ... (Keep existing implementation - unchanged) ...
    if path_np is None or len(path_np) < 2: print("Warning: Not enough data points to plot C-space path."); return
    num_joints = path_np.shape[1]; fig, axs = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True); fig.suptitle(title, fontsize=14)
    time_steps = np.arange(len(path_np))
    for i in range(num_joints):
        ax = axs[i] if num_joints > 1 else axs
        ax.plot(time_steps, np.degrees(path_np[:, i]), label=f'Joint {i+1}')
        if hasattr(config, 'JOINT_LIMITS_MIN') and hasattr(config, 'JOINT_LIMITS_MAX'): min_lim_deg = np.degrees(config.JOINT_LIMITS_MIN[i]); max_lim_deg = np.degrees(config.JOINT_LIMITS_MAX[i]); ax.axhline(min_lim_deg, color='red', linestyle='--', linewidth=0.8, label='Limits' if i==0 else ""); ax.axhline(max_lim_deg, color='red', linestyle='--', linewidth=0.8)
        ax.set_ylabel(f'Joint {i+1} (deg)'); ax.grid(True, linestyle=':');
        if i == 0: ax.legend(loc='best', fontsize='small')
    axs[-1].set_xlabel('Waypoint Index'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

def plot_gd_progress_2d(pose_history_T, cost_history, grad_history, set_data):
     """ Plots GD cost and gradient norm vs iteration, and 2D position path. """
     # ... (Keep existing implementation - unchanged) ...
     if not pose_history_T or not cost_history: print("Warning: Missing history for GD 2D plot."); return
     iterations = np.arange(len(cost_history)); fig, axs = plt.subplots(1, 3, figsize=(18, 5)); fig.suptitle("Gradient Descent Progress")
     axs[0].plot(iterations, cost_history, marker='o', linestyle='-'); axs[0].set_xlabel("Iteration"); axs[0].set_ylabel("Cost"); axs[0].set_title("Cost vs. Iteration"); axs[0].grid(True)
     if grad_history: grad_norms = [np.linalg.norm(g) for g in grad_history]; axs[1].plot(iterations, grad_norms, marker='x', linestyle='-'); axs[1].set_xlabel("Iteration"); axs[1].set_ylabel("Gradient Norm"); axs[1].set_title("Gradient Norm vs. Iteration"); axs[1].grid(True); axs[1].set_yscale('log')
     else: axs[1].text(0.5, 0.5, "Gradient History\nNot Available", ha='center', va='center', transform=axs[1].transAxes)
     positions = np.array([T[:3, 3] for T in pose_history_T if T is not None])
     if len(positions) > 0: axs[2].plot(positions[:, 0], positions[:, 1], marker='.', linestyle='-', label='GD Path (XY)'); axs[2].scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start', zorder=5); axs[2].scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End', zorder=5)
     if set_data and 'pos_bounds' in set_data: bounds = np.array(set_data['pos_bounds']); xmin, xmax = bounds[0]; ymin, ymax = bounds[1]; rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=True, color='lightgrey', alpha=0.5, label='Set Bounds (XY)'); axs[2].add_patch(rect)
     axs[2].set_xlabel("X Position"); axs[2].set_ylabel("Y Position"); axs[2].set_title("Position Path (XY Plane)"); axs[2].grid(True); axs[2].legend(); axs[2].set_aspect('equal', adjustable='box')
     plt.tight_layout(rect=[0, 0, 1, 0.95])
     # plt.show()

def plot_gd_orientation_progress(quat_history_wxyz, cost_history):
     """ Plots quaternion components and angular distance from start vs iteration. """
     # ... (Keep existing implementation - unchanged) ...
     if not quat_history_wxyz or len(quat_history_wxyz) < 1: print("Warning: Missing quaternion history for GD plot."); return
     iterations = np.arange(len(quat_history_wxyz)); quats = np.array(quat_history_wxyz); start_quat = quats[0]; angular_diff_deg = []
     if SCIPY_AVAILABLE and _calculate_geodesic_distance_func is not None: angular_diff_deg = [_calculate_geodesic_distance_func(start_quat, q) for q in quats]; angular_diff_deg = [d if not np.isnan(d) else 0 for d in angular_diff_deg]
     else: print("Warning: Cannot calculate angular distance for GD plot (SciPy missing).")
     fig, axs = plt.subplots(1, 2, figsize=(14, 5)); fig.suptitle("Gradient Descent Orientation Progress"); labels = ['w', 'x', 'y', 'z']
     for i in range(4): axs[0].plot(iterations, quats[:, i], label=labels[i])
     axs[0].set_xlabel("Iteration"); axs[0].set_ylabel("Quaternion Component Value"); axs[0].set_title("Quaternion Components vs. Iteration"); axs[0].grid(True); axs[0].legend()
     if angular_diff_deg: axs[1].plot(iterations, angular_diff_deg, marker='.', linestyle='-'); axs[1].set_xlabel("Iteration"); axs[1].set_ylabel("Angular Distance from Start (deg)"); axs[1].set_title("Orientation Change vs. Iteration"); axs[1].grid(True)
     else: axs[1].text(0.5, 0.5, "Angular Distance\nNot Available", ha='center', va='center', transform=axs[1].transAxes)
     plt.tight_layout(rect=[0, 0, 1, 0.95])
     # plt.show()


print("visualizer.py loaded (with GMM Opt plotting, Log Map plotting, validity checks, and local import fix).")
