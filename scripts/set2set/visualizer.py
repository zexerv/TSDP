# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For plotting boxes
import matplotlib.cm as cm # For colormaps
import matplotlib.colors as mcolors # For color normalization
import config # To access configuration parameters like plot limits

# --- Attempt to import Forward Kinematics ---
# Assuming it's in ik_solver.py, adjust if necessary
try:
    from ik_solver import forward_kinematics
except ImportError:
    print("ERROR: visualizer.py cannot import forward_kinematics. Robot plotting will fail.")
    # Define a dummy function to avoid NameError later, but plots will be incorrect
    def forward_kinematics(q):
        print("ERROR: forward_kinematics is not available!")
        return None, None, None

# --- SciPy Import Check (needed for rotations if used) ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: (visualizer.py) scipy not found. Install with 'pip install scipy'. Orientation features may be limited.")
    SCIPY_AVAILABLE = False
    # Define a dummy R class if scipy is not available and code relies on it
    class R:
        @staticmethod
        def from_matrix(matrix): raise ImportError("scipy not available")
        @staticmethod
        def from_quat(quat): raise ImportError("scipy not available")
        def as_quat(self): raise ImportError("scipy not available")
        def as_matrix(self): raise ImportError("scipy not available")

print("Loading visualizer.py...")

# --- C-Space Path Plotting ---
def plot_cspace_path(path_np, title="C-Space Path"):
    """ Plots the joint angles over the path sequence. """
    if path_np is None or path_np.ndim != 2 or path_np.shape[0] < 2 or path_np.shape[1] != config.NUM_JOINTS:
        print(f"Invalid or insufficient C-space path data provided (Shape: {path_np.shape if path_np is not None else 'None'}). Cannot plot.")
        return

    num_points = path_np.shape[0]
    num_joints = path_np.shape[1]
    time_steps = np.arange(num_points)

    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)
    if num_joints == 1: axes = [axes] # Handle single joint case properly

    fig.suptitle(title, fontsize=14)

    for i in range(num_joints):
        axes[i].plot(time_steps, np.degrees(path_np[:, i]), marker='.', markersize=3, linestyle='-')
        axes[i].set_ylabel(f'Joint {i+1} (deg)')
        axes[i].grid(True)
        # Plot joint limits if configured correctly
        if hasattr(config, 'JOINT_LIMITS_MIN') and hasattr(config, 'JOINT_LIMITS_MAX') and len(config.JOINT_LIMITS_MIN) == num_joints:
           axes[i].axhline(np.degrees(config.JOINT_LIMITS_MIN[i]), color='r', linestyle='--', linewidth=1, alpha=0.7, label='Limits' if i==0 else "")
           axes[i].axhline(np.degrees(config.JOINT_LIMITS_MAX[i]), color='r', linestyle='--', linewidth=1, alpha=0.7)

    # Add legend only once if limits were plotted
    if hasattr(config, 'JOINT_LIMITS_MIN'):
        handles, labels = axes[0].get_legend_handles_labels()
        if handles: # Check if 'Limits' label was added
             fig.legend(handles, labels, loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Path Point Index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show(block=False) # Show plot but allow script to continue


# --- 3D Scene Visualization Class ---
class Visualizer3D:
    """ Handles the creation and population of the 3D visualization scene. """
    def __init__(self, title="Robot Visualization"):
        """ Sets up the 3D plotting environment. """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X base (m)", labelpad=10)
        self.ax.set_ylabel("Y base (m)", labelpad=10)
        self.ax.set_zlabel("Z base (m)", labelpad=10)
        self.ax.set_title(title, fontsize=14, pad=20)

        # Set plot limits from config
        try:
            lim = config.VIS_PLOT_LIMITS
            self.ax.set_xlim(lim[0])
            self.ax.set_ylim(lim[1])
            self.ax.set_zlim(lim[2])
        except (AttributeError, IndexError, TypeError):
            print("Warning: VIS_PLOT_LIMITS not configured correctly in config.py. Using default limits.")
            self.ax.set_xlim([-1, 1]); self.ax.set_ylim([-1, 1]); self.ax.set_zlim([-0.5, 1.5])

        self.ax.set_aspect('auto') # Use 'auto' aspect ratio for better view, 'equal' can distort badly
        # Plot base coordinate frame
        self._plot_coordinate_frame(np.identity(4), length=config.VIS_BASE_FRAME_SIZE, labels=['X₀', 'Y₀', 'Z₀'])
        self.ax.view_init(elev=25., azim=-70) # Adjust viewing angle as needed
        plt.tight_layout()
        self._legend_handles = {} # Dictionary to store handles for legend
        self.colorbar = None # Store colorbar instance to prevent duplicates

    def _plot_coordinate_frame(self, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], text_offset=1.1, linestyle='-', color=None, linewidth=1):
        """ Plots a 3D coordinate frame at the given pose. """
        if pose_matrix is None: return
        try:
            origin = pose_matrix[:3, 3]; R_mat = pose_matrix[:3, :3]; colors = color if color else ['r', 'g', 'b']
            for i in range(3):
                axis_vector = R_mat[:, i]; current_color = colors[i] if isinstance(colors, list) else colors
                self.ax.quiver(origin[0], origin[1], origin[2], axis_vector[0], axis_vector[1], axis_vector[2],
                               length=length, color=current_color, arrow_length_ratio=0.3, normalize=False,
                               linestyle=linestyle, linewidth=linewidth)
                # Add text labels if requested and using default colors
                if labels and isinstance(colors, list):
                     self.ax.text(origin[0] + axis_vector[0] * length * text_offset,
                                  origin[1] + axis_vector[1] * length * text_offset,
                                  origin[2] + axis_vector[2] * length * text_offset,
                                  labels[i], color=colors[i], fontsize=9, ha='center', va='center')
        except Exception as e:
             print(f"Warning: Failed to plot coordinate frame: {e}")


    def _plot_robot_links(self, joint_positions, color='k', linewidth=3, style='-', alpha=1.0):
        """ Plots the robot links given joint positions from FK. Returns the plot handle. """
        if not joint_positions or len(joint_positions) < 2: return None
        try:
            jp = np.array(joint_positions)
            # Plot links as lines, joints as markers
            handle = self.ax.plot(jp[:,0], jp[:,1], jp[:,2], marker='o', markersize=4, linestyle=style,
                                  linewidth=linewidth, color=color, alpha=alpha)
            return handle[0] # Return the line handle for the legend
        except Exception as e:
            print(f"Warning: Failed to plot robot links: {e}")
            return None

    def plot_robot_config(self, q, color='black', linewidth=3, style='-', alpha=1.0, label=None, plot_tcp_frame=True):
        """ Plots the robot in a specific configuration q. """
        if q is None:
             print("Warning: plot_robot_config received None configuration.")
             return
        try:
            # Ensure FK function is available and works
            joint_pos, T_tcp, _ = forward_kinematics(q) # Assuming FK returns these
            if joint_pos and T_tcp is not None:
                handle = self._plot_robot_links(joint_pos, color=color, linewidth=linewidth, style=style, alpha=alpha)
                # Store handle for legend only if label is provided and not already stored
                if handle and label and label not in self._legend_handles:
                     self._legend_handles[label] = handle
                # Plot TCP frame if requested
                if plot_tcp_frame:
                    self._plot_coordinate_frame(T_tcp, length=config.VIS_TCP_FRAME_SIZE, labels=None, # No labels for TCP
                                                  color=color, linestyle=style, linewidth=linewidth*0.75)
            # else: print(f"Warning: FK failed for config q={np.degrees(q)}, cannot plot.") # Less verbose
        except Exception as e:
            print(f"Error plotting robot config q={np.degrees(q)}: {e}")

    def plot_task_space_set(self, set_data, color='cyan', alpha=0.15, label=None):
         """ Plots the task space set bounds as a semi-transparent box. """
         if set_data is None or 'pos_bounds' not in set_data:
              # print("Warning: plot_task_space_set received invalid set_data.") # Less verbose
              return
         try:
             pos_bounds = np.array(set_data['pos_bounds'])
             xmin, xmax = pos_bounds[0, 0], pos_bounds[0, 1]; ymin, ymax = pos_bounds[1, 0], pos_bounds[1, 1]; zmin, zmax = pos_bounds[2, 0], pos_bounds[2, 1]
             # Define vertices of the box
             verts = [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
                      (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)]
             # Define faces using vertices (ensure correct order for outward normals if needed, though not critical for display)
             faces = [[verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]],
                      [verts[0], verts[1], verts[5], verts[4]], [verts[2], verts[3], verts[7], verts[6]],
                      [verts[0], verts[3], verts[7], verts[4]], [verts[1], verts[2], verts[6], verts[5]]]
             face_collection = Poly3DCollection(faces, linewidths=1, edgecolors=mcolors.to_rgba(color, alpha=alpha*1.5), facecolors=mcolors.to_rgba(color, alpha=alpha))
             self.ax.add_collection3d(face_collection)
             # Create a proxy artist for the legend
             if label and label not in self._legend_handles:
                  # Use a Rectangle proxy; actual handle is the collection which doesn't work well in legends
                  proxy = plt.Rectangle((0, 0), 1, 1, fc=mcolors.to_rgba(color, alpha=alpha*1.5)) # Make alpha slightly higher for legend
                  self._legend_handles[label] = proxy
         except Exception as e:
              print(f"Warning: Failed to plot task space set '{label}': {e}")


    def plot_trajectory(self, path_np, num_snapshots=10, color='purple', style='--', label_prefix='Path', plot_tcp_frames=False):
        """ Plots snapshots of the robot along a C-space trajectory. """
        if path_np is None or path_np.ndim != 2 or path_np.shape[0] < 2:
            print("No valid trajectory path to visualize in 3D.")
            return
        path_len = path_np.shape[0]
        # Ensure num_snapshots doesn't exceed path length
        num_snapshots = min(num_snapshots, path_len)
        # Ensure at least start and end are plotted if num_snapshots >= 2
        if num_snapshots < 2 and path_len >= 2: num_snapshots = 2
        if num_snapshots == 1 and path_len == 1: indices = [0]
        elif num_snapshots > 0 : indices = np.linspace(0, path_len - 1, num_snapshots, dtype=int)
        else: indices = []

        # print(f"Plotting {len(indices)} snapshots for trajectory...") # Debug
        for i, idx in enumerate(indices):
            q = path_np[idx]
            # Fade snapshots from start to end for visual clarity
            alpha_val = 0.15 + 0.85 * (i / (len(indices)-1)) if len(indices) > 1 else 1.0
            snapshot_label = None
            # Label only the first and last snapshot for the legend
            if len(indices) > 1:
                 if i == 0: snapshot_label = f"{label_prefix} Start"
                 elif i == len(indices) -1: snapshot_label = f"{label_prefix} End"
            elif len(indices) == 1: # If only one snapshot, label it
                 snapshot_label = f"{label_prefix}"

            # Plot TCP frame only for start/end unless specifically requested
            tcp_frame = plot_tcp_frames or (i == 0 or i == len(indices)-1)
            self.plot_robot_config(q, color=color, linewidth=2, style=style, alpha=alpha_val, label=snapshot_label, plot_tcp_frame=tcp_frame)

    def plot_gd_progress(self, gd_pose_history_xyz, gd_cost_history, label='GD Path'):
        """ Plots the GD path in 3D, colored by cost, and marks the best point found. """
        if not isinstance(gd_pose_history_xyz, (list, np.ndarray)) or len(gd_pose_history_xyz) < 1:
            # print("Not enough GD history points to plot 3D progress.") # Less verbose
            return
        if not isinstance(gd_cost_history, (list, np.ndarray)) or len(gd_cost_history) != len(gd_pose_history_xyz):
             print("Warning: Mismatch GD pose/cost history length. Cannot plot GD progress.")
             return

        poses = np.array(gd_pose_history_xyz)
        costs = np.array(gd_cost_history)
        # Ensure costs are finite for normalization and finding minimum
        finite_mask = np.isfinite(costs)
        valid_costs = costs[finite_mask]
        valid_poses = poses[finite_mask]

        best_cost_idx_in_valid = -1 # Initialize as not found

        if len(valid_costs) == 0:
            # print("Warning: No finite costs in GD history to determine color range or best point.") # Less verbose
            norm = mcolors.Normalize(vmin=0, vmax=1) # Default norm
        else:
            vmin, vmax = np.min(valid_costs), np.max(valid_costs)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
            # Find index of minimum cost *within the valid costs*
            best_cost_idx_in_valid = np.argmin(valid_costs)

        # Use reversed Viridis: Yellow=Low Cost (Good), Purple=High Cost (Bad)
        cmap = cm.viridis_r

        # Plot the path sequence using ALL points (even if cost was inf) for continuity
        if len(poses) > 1:
            gd_line = self.ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2, label=label)[0]
            if label and label not in self._legend_handles: self._legend_handles[label] = gd_line

        # Plot points colored by cost (only plot points with finite cost)
        sc = None
        if len(valid_costs) > 0:
             sc = self.ax.scatter(valid_poses[:, 0], valid_poses[:, 1], valid_poses[:, 2],
                                  c=valid_costs, cmap=cmap, norm=norm, s=50, edgecolor='k',
                                  linewidth=0.5, alpha=0.9, zorder=5)
             # Add colorbar if not already present and if we plotted scatter points
             if self.colorbar is None:
                 try:
                     self.colorbar = self.fig.colorbar(sc, ax=self.ax, shrink=0.6, aspect=20, label='Path Cost (Yellow=Low)')
                 except Exception as e: print(f"Warning: Could not create colorbar: {e}")
        # else: # Optionally plot points with inf cost differently
            # inf_poses = poses[~finite_mask]
            # if len(inf_poses) > 0:
            #    self.ax.scatter(inf_poses[:, 0], inf_poses[:, 1], inf_poses[:, 2], color='red', marker='x', s=50, label='GD Points (Inf Cost)')

        # Mark the best point found (among finite costs)
        if best_cost_idx_in_valid != -1:
            best_pose = valid_poses[best_cost_idx_in_valid] # Get corresponding pose
            best_marker = self.ax.scatter(best_pose[0], best_pose[1], best_pose[2], marker='*', s=250, color='red', edgecolor='black', zorder=6, label='GD Best Found')
            if 'GD Best Found' not in self._legend_handles: self._legend_handles['GD Best Found'] = best_marker

    def plot_pso_final_swarm(self, final_particle_positions, best_position_xyz):
        """ Plots the final positions of all PSO particles and the best one found. """
        if final_particle_positions is None or final_particle_positions.ndim != 2 or final_particle_positions.shape[0] == 0:
            # print("No valid PSO final swarm positions to plot.") # Less verbose
            return
        if final_particle_positions.shape[1] < 3:
             print("Error: PSO particle positions do not have at least 3 dimensions (XYZ). Cannot plot.")
             return

        try:
            # Extract position part (first 3 columns)
            swarm_pos_xyz = final_particle_positions[:, :3]

            # Plot all final particle positions as small grey dots
            swarm_scatter = self.ax.scatter(swarm_pos_xyz[:, 0], swarm_pos_xyz[:, 1], swarm_pos_xyz[:, 2],
                                            marker='.', s=30, color='grey', alpha=0.6, zorder=3, label='PSO Final Particles')
            if 'PSO Final Particles' not in self._legend_handles:
                self._legend_handles['PSO Final Particles'] = swarm_scatter

            # Plot the best position found by PSO with a distinct marker
            if best_position_xyz is not None:
                # Ensure best_position_xyz is a 1D array/list of 3 elements
                if np.asarray(best_position_xyz).shape == (3,):
                     best_marker = self.ax.scatter(best_position_xyz[0], best_position_xyz[1], best_position_xyz[2],
                                                  marker='o', s=150, color='limegreen', edgecolor='black', zorder=6, label='PSO Best Found')
                     if 'PSO Best Found' not in self._legend_handles:
                          self._legend_handles['PSO Best Found'] = best_marker
                else: print("Warning: Invalid shape for PSO best_position_xyz.")
            else: print("Warning: Best PSO position not provided for plotting.")

        except Exception as e:
             print(f"Error plotting PSO final swarm: {e}")

    def add_legend(self, **kwargs):
        """ Adds a legend to the plot using stored handles. """
        if not self._legend_handles: return
        # Filter out None handles if any occurred
        valid_handles = {label: handle for label, handle in self._legend_handles.items() if handle is not None}
        if valid_handles: # Only show legend if there are valid handles
             try:
                  self.ax.legend(valid_handles.values(), valid_handles.keys(), **kwargs)
             except Exception as e:
                  print(f"Warning: Failed to add legend: {e}")

    def show(self):
        """ Displays the plot. """
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")


# --- 2D Gradient Descent Visualization ---
def plot_gd_progress_2d(gd_pose_history_xyz, gd_cost_history, gd_gradient_history_7d, set_data):
    """ Plots the 2D projections (xy, yz, zx) of the GD path and position gradients. """
    if not isinstance(gd_pose_history_xyz, (list, np.ndarray)) or len(gd_pose_history_xyz) < 1:
        # print("Not enough GD history points to plot 2D progress.") # Less verbose
        return
    if not isinstance(gd_cost_history, (list, np.ndarray)) or len(gd_cost_history) != len(gd_pose_history_xyz):
         print("Warning: Mismatch GD pose/cost history length. Cannot plot 2D GD progress.")
         return

    poses = np.array(gd_pose_history_xyz)
    costs = np.array(gd_cost_history)

    # Extract position gradients safely
    gradients_pos = None
    plot_gradients = False
    if gd_gradient_history_7d and isinstance(gd_gradient_history_7d, (list, np.ndarray)) and len(gd_gradient_history_7d) == len(poses):
         grad_array = np.asarray(gd_gradient_history_7d) # Convert list of arrays/lists
         if grad_array.ndim == 2 and grad_array.shape[1] >= 3:
             gradients_pos = grad_array[:, :3]
             plot_gradients = True
             valid_grad_indices = np.linalg.norm(gradients_pos, axis=1) > 1e-9
             # if not np.all(valid_grad_indices): print(f"Plotting position gradients for {np.sum(valid_grad_indices)} non-zero steps.") # Less verbose
         # else: print("Warning: Gradient history has unexpected shape. Skipping gradient plot.") # Less verbose
    # elif gd_gradient_history_7d: print(f"Warning: Mismatch pos history ({len(poses)})/grad history ({len(gd_gradient_history_7d)}). Skipping grad plot.") # Less verbose

    # Normalize costs for colormap
    finite_mask = np.isfinite(costs)
    valid_costs = costs[finite_mask]
    if len(valid_costs) == 0: norm = mcolors.Normalize(vmin=0, vmax=1)
    else: vmin, vmax = np.min(valid_costs), np.max(valid_costs); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
    cmap = cm.viridis_r # Yellow=Low

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Gradient Descent Position Progress (2D Projections)', fontsize=16)

    # Internal plotting function for each projection
    def plot_projection(ax, dim1_idx, dim2_idx, dim1_label, dim2_label):
        sc = None
        if len(valid_costs) > 0:
             valid_poses = poses[finite_mask]
             sc = ax.scatter(valid_poses[:, dim1_idx], valid_poses[:, dim2_idx], c=valid_costs, cmap=cmap, norm=norm,
                             s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
        if len(poses) > 1: ax.plot(poses[:, dim1_idx], poses[:, dim2_idx], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
        legend_elements = {}
        if plot_gradients and np.sum(valid_grad_indices) > 0:
            # (Plotting gradients as before...)
            valid_grads_pos = gradients_pos[valid_grad_indices]; valid_poses_for_grads = poses[valid_grad_indices]
            max_grad_comp = np.max(np.abs(valid_grads_pos[:, [dim1_idx, dim2_idx]])) + 1e-6
            data_range = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]) + 1e-6
            grad_scale = 0.05 * data_range / max_grad_comp
            qvr = ax.quiver(valid_poses_for_grads[:, dim1_idx], valid_poses_for_grads[:, dim2_idx], -valid_grads_pos[:, dim1_idx], -valid_grads_pos[:, dim2_idx],
                            color='red', scale_units='xy', angles='xy', scale=1/grad_scale, width=0.004, alpha=0.7, zorder=4, label='Pos Descent (-Grad)')
            legend_elements['Pos Descent (-Grad)'] = qvr

        if set_data and 'pos_bounds' in set_data:
            pos_bounds = np.array(set_data['pos_bounds'])
            bound_line, = ax.plot([pos_bounds[dim1_idx, 0], pos_bounds[dim1_idx, 1], pos_bounds[dim1_idx, 1], pos_bounds[dim1_idx, 0], pos_bounds[dim1_idx, 0]],
                                   [pos_bounds[dim2_idx, 0], pos_bounds[dim2_idx, 0], pos_bounds[dim2_idx, 1], pos_bounds[dim2_idx, 1], pos_bounds[dim2_idx, 0]],
                                   color='blue', linestyle='--', alpha=0.5, label='Set Bounds', zorder=1)
            legend_elements['Set Bounds'] = bound_line

        ax.set_xlabel(f'{dim1_label} (m)'); ax.set_ylabel(f'{dim2_label} (m)'); ax.set_title(f'{dim1_label}-{dim2_label} Projection'); ax.grid(True); ax.set_aspect('equal', adjustable='box')
        if legend_elements: ax.legend(handles=legend_elements.values(), labels=legend_elements.keys(), loc='best', fontsize=8)
        return sc

    # Plot projections and add colorbar
    sc_xy = plot_projection(axes[0], 0, 1, 'X', 'Y')
    plot_projection(axes[1], 1, 2, 'Y', 'Z')
    plot_projection(axes[2], 2, 0, 'Z', 'X')
    if sc_xy: # Add colorbar only if scatter points were plotted
        try: fig.colorbar(sc_xy, ax=axes.ravel().tolist(), shrink=0.7, aspect=20, label='Path Cost (Yellow=Low)')
        except Exception as e: print(f"Warning: Could not create 2D colorbar: {e}")

    try: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except ValueError: print("Warning: tight_layout failed for 2D GD plot.") # Sometimes fails with aspect('equal')
    plt.show(block=False)


# --- Quaternion Plotting (for GD) ---
def plot_gd_orientation_progress(quat_history_wxyz, cost_history):
    """ Plots the 2D projections (xy, zw) of the quaternion path during GD. """
    if not isinstance(quat_history_wxyz, (list, np.ndarray)) or len(quat_history_wxyz) < 1:
        # print("Not enough GD history points to plot orientation progress.") # Less verbose
        return
    if not isinstance(cost_history, (list, np.ndarray)) or len(cost_history) != len(quat_history_wxyz):
         print("Warning: Mismatch GD quaternion/cost history length. Cannot plot orientation progress.")
         return

    quats = np.array(quat_history_wxyz)
    # Ensure quats has the correct shape (N, 4)
    if quats.ndim != 2 or quats.shape[1] != 4:
         print(f"Warning: Invalid shape for quaternion history ({quats.shape}). Cannot plot.")
         return
    costs = np.array(cost_history)

    # Normalize costs for colormap
    finite_mask = np.isfinite(costs)
    valid_costs = costs[finite_mask]
    if len(valid_costs) == 0: norm = mcolors.Normalize(vmin=0, vmax=1)
    else: vmin, vmax = np.min(valid_costs), np.max(valid_costs); norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmin != vmax else vmax + 0.1)
    cmap = cm.viridis_r # Yellow=Low

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Gradient Descent Orientation Progress (Quaternion Projections)', fontsize=16)

    # Plot XY projection (qx vs qy)
    ax_xy = axes[0]
    sc_xy = None
    if len(valid_costs) > 0:
         valid_quats = quats[finite_mask]
         sc_xy = ax_xy.scatter(valid_quats[:, 1], valid_quats[:, 2], c=valid_costs, cmap=cmap, norm=norm,
                               s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
    if len(quats) > 1: ax_xy.plot(quats[:, 1], quats[:, 2], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
    ax_xy.set_xlabel('Quaternion X'); ax_xy.set_ylabel('Quaternion Y'); ax_xy.set_title('Quaternion XY Projection')
    ax_xy.grid(True); ax_xy.set_aspect('equal', adjustable='box')
    ax_xy.set_xlim(-1.05, 1.05); ax_xy.set_ylim(-1.05, 1.05)

    # Plot ZW projection (qz vs qw)
    ax_zw = axes[1]
    if len(valid_costs) > 0:
         valid_quats = quats[finite_mask]
         sc_zw = ax_zw.scatter(valid_quats[:, 3], valid_quats[:, 0], c=valid_costs, cmap=cmap, norm=norm,
                               s=40, edgecolor='k', linewidth=0.5, alpha=0.8, zorder=3)
    if len(quats) > 1: ax_zw.plot(quats[:, 3], quats[:, 0], marker=None, linestyle='-', color='gray', alpha=0.5, zorder=2)
    ax_zw.set_xlabel('Quaternion Z'); ax_zw.set_ylabel('Quaternion W'); ax_zw.set_title('Quaternion ZW Projection')
    ax_zw.grid(True); ax_zw.set_aspect('equal', adjustable='box')
    ax_zw.set_xlim(-1.05, 1.05); ax_zw.set_ylim(-1.05, 1.05)

    # Add a single colorbar if points were plotted
    if sc_xy:
        try: fig.colorbar(sc_xy, ax=axes.ravel().tolist(), shrink=0.8, aspect=15, label='Path Cost (Yellow=Low)')
        except Exception as e: print(f"Warning: Could not create orientation colorbar: {e}")

    try: plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    except ValueError: print("Warning: tight_layout failed for orientation plot.")
    plt.show(block=False)


# --- Final module print statement ---
print("visualizer.py loaded successfully (with PSO swarm plot and GD best marker).")