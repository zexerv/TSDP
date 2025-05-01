# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import config
from ik_solver import forward_kinematics
# --- NEW: Add scipy for rotation conversions ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not found for visualizer. Install with 'pip install scipy'.")
    SCIPY_AVAILABLE = False
# --- END NEW ---


print("Loading visualizer.py...")

# --- C-Space Path Plotting ---
def plot_cspace_path(path_np, title="C-Space Path"):
    """ Plots the joint angles over the path sequence. """
    if path_np is None or path_np.shape[0] < 2:
        print("No C-space path to plot.")
        return

    num_points = path_np.shape[0]
    num_joints = path_np.shape[1]
    time_steps = np.arange(num_points)

    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Handle single joint case
    fig.suptitle(title, fontsize=14)

    for i in range(num_joints):
        axes[i].plot(time_steps, np.degrees(path_np[:, i]), marker='.', markersize=3, linestyle='-')
        axes[i].set_ylabel(f'Joint {i+1} (deg)')
        axes[i].grid(True)
        axes[i].axhline(np.degrees(config.JOINT_LIMITS_MIN[i]), color='r', linestyle='--', linewidth=1, alpha=0.7)
        axes[i].axhline(np.degrees(config.JOINT_LIMITS_MAX[i]), color='r', linestyle='--', linewidth=1, alpha=0.7)

    axes[-1].set_xlabel('Path Point Index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show(block=False)


# --- 3D Scene Visualization ---
class Visualizer3D:
    # ... (__init__, _plot_coordinate_frame, _plot_robot_links, plot_robot_config, plot_task_space_set, plot_trajectory, plot_gd_progress, add_legend, show) ...
    # (No changes needed in these methods from previous version)
    def __init__(self, title="Robot Visualization"):
        """ Sets up the 3D plotting environment. """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X base (m)"); self.ax.set_ylabel("Y base (m)"); self.ax.set_zlabel("Z base (m)")
        self.ax.set_title(title, fontsize=14)
        lim = config.VIS_PLOT_LIMITS
        self.ax.set_xlim(lim[0]); self.ax.set_ylim(lim[1]); self.ax.set_zlim(lim[2])
        self.ax.set_aspect('auto')
        self._plot_coordinate_frame(np.identity(4), length=config.VIS_BASE_FRAME_SIZE, labels=['X₀', 'Y₀', 'Z₀'])
        self.ax.view_init(elev=25., azim=-70)
        plt.tight_layout()
        self._legend_handles = {}
        self.colorbar = None

    def _plot_coordinate_frame(self, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], text_offset=1.1, linestyle='-', color=None, linewidth=1):
        origin = pose_matrix[:3, 3]; R_mat = pose_matrix[:3, :3]; colors = color if color else ['r', 'g', 'b']
        for i in range(3):
            axis_vector = R_mat[:, i]; current_color = colors[i] if isinstance(colors, list) else colors
            self.ax.quiver(origin[0], origin[1], origin[2], axis_vector[0], axis_vector[1], axis_vector[2], length=length, color=current_color, arrow_length_ratio=0.3, normalize=False, linestyle=linestyle, linewidth=linewidth)
            if labels and isinstance(colors, list): self.ax.text(origin[0] + axis_vector[0] * length * text_offset, origin[1] + axis_vector[1] * length * text_offset, origin[2] + axis_vector[2] * length * text_offset, labels[i], color=colors[i], fontsize=9)

    def _plot_robot_links(self, joint_positions, color='k', linewidth=3, style='-', alpha=1.0):
        if not joint_positions or len(joint_positions) < 2: return None
        jp = np.array(joint_positions)
        handle = self.ax.plot(jp[:,0], jp[:,1], jp[:,2], marker='o', markersize=4, linestyle=style, linewidth=linewidth, color=color, alpha=alpha)
        return handle[0]

    def plot_robot_config(self, q, color='black', linewidth=3, style='-', alpha=1.0, label=None, plot_tcp_frame=True):
        try:
            joint_pos, T_tcp, _ = forward_kinematics(q)
            if joint_pos and T_tcp is not None:
                handle = self._plot_robot_links(joint_pos, color=color, linewidth=linewidth, style=style, alpha=alpha)
                if handle and label and label not in self._legend_handles: self._legend_handles[label] = handle
                if plot_tcp_frame: self._plot_coordinate_frame(T_tcp, length=config.VIS_TCP_FRAME_SIZE, labels=None, color=color, linestyle=style, linewidth=linewidth*0.75)
            else: print(f"Warning: FK failed for config q={np.degrees(q)}, cannot plot.")
        except Exception as e: print(f"Error plotting robot config q={np.degrees(q)}: {e}")

    def plot_task_space_set(self, set_data, color='cyan', alpha=0.15, label=None):
        if set_data is None or 'pos_bounds' not in set_data: return
        pos_bounds = np.array(set_data['pos_bounds'])
        xmin, xmax = pos_bounds[0, 0], pos_bounds[0, 1]; ymin, ymax = pos_bounds[1, 0], pos_bounds[1, 1]; zmin, zmax = pos_bounds[2, 0], pos_bounds[2, 1]
        verts = [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin), (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)]
        faces = [[verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]], [verts[0], verts[1], verts[5], verts[4]], [verts[2], verts[3], verts[7], verts[6]], [verts[0], verts[3], verts[7], verts[4]], [verts[1], verts[2], verts[6], verts[5]]]
        face_collection = Poly3DCollection(faces, linewidths=1, edgecolors=color, facecolors=color, alpha=alpha)
        self.ax.add_collection3d(face_collection)
        if label and label not in self._legend_handles:
             proxy = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha*2); self._legend_handles[label] = proxy

    def plot_trajectory(self, path_np, num_snapshots=10, color='purple', style='--', label_prefix='Path', plot_tcp_frames=False):
        if path_np is None or path_np.shape[0] < 2: print("No trajectory path to visualize in 3D."); return
        path_len = path_np.shape[0]; indices = np.linspace(0, path_len - 1, min(num_snapshots, path_len), dtype=int)
        for i, idx in enumerate(indices):
            q = path_np[idx]; alpha_val = 0.1 + 0.9 * (i / (len(indices)-1)) if len(indices) > 1 else 1.0
            snapshot_label = None
            if i == 0: snapshot_label = f"{label_prefix} Start"
            elif i == len(indices) -1: snapshot_label = f"{label_prefix} End"
            tcp_frame = plot_tcp_frames or (i == 0 or i == len(indices)-1)
            self.plot_robot_config(q, color=color, linewidth=2, style=style, alpha=alpha_val, label=snapshot_label, plot_tcp_frame=tcp_frame)

    def plot_gd_progress(self, gd_pose_history_xyz, gd_cost_history, label='GD Path'):
        if not gd_pose_history_xyz or len(gd_pose_history_xyz) < 1: print("Not enough GD history points to plot 3D progress."); return
        poses = np.array(gd_pose_history_xyz); costs = np.array(gd_cost_history); valid_costs = costs[np.isfinite(costs)]
        if len(valid_costs) == 0: norm = mcolors.Normalize(vmin=0, vmax=1)
        else: vmin, vmax = np.min(valid_costs), np.max(valid_costs); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
        cmap = cm.viridis_r
        if len(poses) > 1: gd_line = self.ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], marker='.', markersize=5, linestyle='-', color='gray', alpha=0.7, label=label)[0]
        else: gd_line = None # No line for single point
        sc = self.ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c=costs, cmap=cmap, norm=norm, s=50, edgecolor='k', linewidth=0.5, alpha=0.9, zorder=5)
        if self.colorbar is None:
            try: self.colorbar = self.fig.colorbar(sc, ax=self.ax, shrink=0.6, aspect=20, label='Path Cost (Lower is better)')
            except Exception as e: print(f"Warning: Could not create colorbar: {e}")
        if gd_line and label and label not in self._legend_handles: self._legend_handles[label] = gd_line

    def add_legend(self, **kwargs):
        if not self._legend_handles: return
        valid_handles = {label: handle for label, handle in self._legend_handles.items() if handle is not None}
        self.ax.legend(valid_handles.values(), valid_handles.keys(), **kwargs)

    def show(self): plt.show()


# --- 2D Gradient Descent Visualization ---
def plot_gd_progress_2d(gd_pose_history_xyz, gd_cost_history, gd_gradient_history_7d, set_data):
    """ Plots the 2D projections (xy, yz, zx) of the GD path and position gradients. """
    if not gd_pose_history_xyz or len(gd_pose_history_xyz) < 1: print("Not enough GD history points to plot 2D progress."); return

    poses = np.array(gd_pose_history_xyz); costs = np.array(gd_cost_history)
    gradients_pos = np.array(gd_gradient_history_7d)[:, :3] if gd_gradient_history_7d else None # Extract position gradient

    plot_gradients = False
    if gradients_pos is not None and len(gradients_pos) == len(poses):
        plot_gradients = True; valid_grad_indices = np.linalg.norm(gradients_pos, axis=1) > 1e-9
        if not np.all(valid_grad_indices): print(f"Plotting position gradients for {np.sum(valid_grad_indices)} non-zero steps.")
    elif gradients_pos is not None: print(f"Warning: Mismatch pos history ({len(poses)})/grad history ({len(gradients_pos)}). Skipping grad plot.")

    valid_costs = costs[np.isfinite(costs)]
    if len(valid_costs) == 0: norm = mcolors.Normalize(vmin=0, vmax=1)
    else: vmin, vmax = np.min(valid_costs), np.max(valid_costs); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
    cmap = cm.viridis_r

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)); fig.suptitle('Gradient Descent Position Progress (2D Projections)', fontsize=16)

    def plot_projection(ax, dim1_idx, dim2_idx, dim1_label, dim2_label):
        sc = ax.scatter(poses[:, dim1_idx], poses[:, dim2_idx], c=costs, cmap=cmap, norm=norm, s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
        if len(poses) > 1: ax.plot(poses[:, dim1_idx], poses[:, dim2_idx], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
        legend_elements = {}
        if plot_gradients and np.sum(valid_grad_indices) > 0:
            valid_grads = gradients_pos[valid_grad_indices]; valid_poses = poses[valid_grad_indices]
            max_grad_comp = np.max(np.abs(valid_grads[:, [dim1_idx, dim2_idx]])) + 1e-6
            data_range = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]) + 1e-6
            grad_scale = 0.05 * data_range / max_grad_comp
            qvr = ax.quiver(valid_poses[:, dim1_idx], valid_poses[:, dim2_idx], -valid_grads[:, dim1_idx], -valid_grads[:, dim2_idx], color='red', scale_units='xy', angles='xy', scale=1/grad_scale, width=0.004, alpha=0.7, zorder=4, label='Pos Descent (-Grad)')
            legend_elements['Pos Descent (-Grad)'] = qvr
        pos_bounds = np.array(set_data['pos_bounds'])
        bound_line, = ax.plot([pos_bounds[dim1_idx, 0], pos_bounds[dim1_idx, 1], pos_bounds[dim1_idx, 1], pos_bounds[dim1_idx, 0], pos_bounds[dim1_idx, 0]], [pos_bounds[dim2_idx, 0], pos_bounds[dim2_idx, 0], pos_bounds[dim2_idx, 1], pos_bounds[dim2_idx, 1], pos_bounds[dim2_idx, 0]], color='blue', linestyle='--', alpha=0.5, label='Set Bounds', zorder=1)
        legend_elements['Set Bounds'] = bound_line
        ax.set_xlabel(f'{dim1_label} (m)'); ax.set_ylabel(f'{dim2_label} (m)'); ax.set_title(f'{dim1_label}-{dim2_label} Projection'); ax.grid(True); ax.set_aspect('equal', adjustable='box')
        ax.legend(handles=legend_elements.values(), labels=legend_elements.keys(), loc='best', fontsize=8)
        return sc

    sc_xy = plot_projection(axes[0], 0, 1, 'X', 'Y'); plot_projection(axes[1], 1, 2, 'Y', 'Z'); plot_projection(axes[2], 2, 0, 'Z', 'X')
    fig.colorbar(sc_xy, ax=axes.ravel().tolist(), shrink=0.7, aspect=20, label='Path Cost (Lower is better)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show(block=False)


# --- NEW: Quaternion Plotting ---
def plot_gd_orientation_progress(quat_history_wxyz, cost_history):
    """
    Plots the 2D projections (xy, zw) of the quaternion path during GD.

    Args:
        quat_history_wxyz (list): List of 4D numpy arrays (quaternions w,x,y,z).
        cost_history (list): List of corresponding path costs.
    """
    if not quat_history_wxyz or len(quat_history_wxyz) < 1:
        print("Not enough GD history points to plot orientation progress.")
        return

    quats = np.array(quat_history_wxyz) # Shape (N, 4) -> w, x, y, z
    costs = np.array(cost_history)

    # Normalize costs for colormap
    valid_costs = costs[np.isfinite(costs)]
    if len(valid_costs) == 0: norm = mcolors.Normalize(vmin=0, vmax=1)
    else: vmin, vmax = np.min(valid_costs), np.max(valid_costs); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
    cmap = cm.viridis_r

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Gradient Descent Orientation Progress (Quaternion Projections)', fontsize=16)

    # Plot XY projection
    ax_xy = axes[0]
    sc_xy = ax_xy.scatter(quats[:, 1], quats[:, 2], c=costs, cmap=cmap, norm=norm, s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
    if len(quats) > 1: ax_xy.plot(quats[:, 1], quats[:, 2], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
    ax_xy.set_xlabel('Quaternion X'); ax_xy.set_ylabel('Quaternion Y'); ax_xy.set_title('Quaternion XY Projection')
    ax_xy.grid(True); ax_xy.set_aspect('equal', adjustable='box')
    # Set limits for quaternion components (usually -1 to 1)
    ax_xy.set_xlim(-1.05, 1.05); ax_xy.set_ylim(-1.05, 1.05)


    # Plot ZW projection
    ax_zw = axes[1]
    sc_zw = ax_zw.scatter(quats[:, 3], quats[:, 0], c=costs, cmap=cmap, norm=norm, s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
    if len(quats) > 1: ax_zw.plot(quats[:, 3], quats[:, 0], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
    ax_zw.set_xlabel('Quaternion Z'); ax_zw.set_ylabel('Quaternion W'); ax_zw.set_title('Quaternion ZW Projection')
    ax_zw.grid(True); ax_zw.set_aspect('equal', adjustable='box')
    ax_zw.set_xlim(-1.05, 1.05); ax_zw.set_ylim(-1.05, 1.05)


    # Add a single colorbar
    fig.colorbar(sc_xy, ax=axes.ravel().tolist(), shrink=0.8, aspect=15, label='Path Cost (Lower is better)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)


print("visualizer.py loaded successfully (with orientation plot).")

