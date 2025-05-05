# main.py
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import config # Reads updated config (probabilistic only)
import utils # Reads updated utils (probabilistic only)
from planner import OMPLPlanner
import visualizer
# Import specific IK/FK needed
try:
    from ik_solver import forward_kinematics
    IK_SOLVER_AVAILABLE = utils.IK_SOLVER_AVAILABLE # Check utils flag
except ImportError:
    print("WARNING (main.py): Could not import forward_kinematics from ik_solver.")
    IK_SOLVER_AVAILABLE = False
import traceback # Import for better error reporting
import itertools # For generating branch tuples
import random # For selecting colors
import inspect # For debugging imports

# --- Import Optimization Function and Check PySwarms ---
PY_SWARMS_AVAILABLE = False # Define globally, default to False
run_gmm_waypoint_optimization = None # Define placeholder
try:
    import optimization
    print("INFO (main.py): Successfully imported 'optimization' module.")
    if hasattr(optimization, 'PY_SWARMS_AVAILABLE'): PY_SWARMS_AVAILABLE = optimization.PY_SWARMS_AVAILABLE
    if config.ENABLE_GMM_OPTIMIZATION:
        if hasattr(optimization, 'run_gmm_waypoint_optimization') and callable(getattr(optimization, 'run_gmm_waypoint_optimization')):
            run_gmm_waypoint_optimization = optimization.run_gmm_waypoint_optimization
            print("INFO (main.py): Successfully found and imported 'run_gmm_waypoint_optimization' function.")
        else:
            print("ERROR (main.py): 'optimization' module imported BUT 'run_gmm_waypoint_optimization' function not found or not callable.")
            print("Available attributes in optimization module:")
            try: print([name for name, obj in inspect.getmembers(optimization) if callable(obj) or not name.startswith("__")])
            except Exception as e_inspect: print(f"Could not inspect optimization module attributes: {e_inspect}")
            def run_gmm_waypoint_optimization(*args, **kwargs): return None, None, float('inf'), {'algorithm': 'GMM_OPT_FUNC_MISSING'}
    else:
        def run_gmm_waypoint_optimization(*args, **kwargs): return None, None, float('inf'), {'algorithm': 'GMM_OPT_DISABLED'}
        print("INFO (main.py): GMM waypoint optimization disabled in config.")
except ImportError:
    print("ERROR (main.py): Failed to import optimization.py. GMM optimization unavailable.")
    def run_gmm_waypoint_optimization(*args, **kwargs): return None, None, float('inf'), {'algorithm': 'GMM_OPT_IMPORT_FAILED'}
    PY_SWARMS_AVAILABLE = False

# --- SciPy Rotation Import Check ---
if utils.SCIPY_AVAILABLE:
    try: from scipy.spatial.transform import Rotation as R; print("INFO (main.py): Successfully imported scipy.spatial.transform.Rotation")
    except ImportError: print("ERROR (main.py): utils.SCIPY_AVAILABLE was True, but failed to import Rotation here.")
else: print("WARNING (main.py): scipy not found. Transformation/Quaternion features might be limited.")

print("Starting main.py execution (Probabilistic Branch Following + GMM Optimization Approach)...")

# --- REMOVED: find_optimal_target_in_set ---

# --- Main Scenario Function ---
def run_planning_scenario():
    """Defines and runs the multi-segment motion planning scenario using
       probabilistic boundary waypoints, IK branch following, and optional GMM optimization."""

    # --- 0. Load Common Configurations & Data ---
    print("\n" + "="*40); print("=== 0. Loading Common Configuration Files & Data ==="); print("="*40)
    H_I_B = None; H_B_I = None; interface_transforms = None; H_B_D = None
    global_interface_id = config.INTERFACE_ID
    boundary_stats_list = None
    try:
        interface_transforms = utils.load_interface_transforms(config.INTERFACE_TRANSFORMS_YAML_PATH)
        if interface_transforms is None: raise ValueError(f"Interface Transforms YAML invalid")
        H_B_D = utils.load_environment_config(config.ENVIRONMENT_YAML_PATH)
        if H_B_D is None: raise ValueError(f"Environment Config invalid")
        if not global_interface_id: raise ValueError(f"INTERFACE_ID not set")
        if global_interface_id not in interface_transforms: raise ValueError(f"Interface ID '{global_interface_id}' not found")
        H_D_I = interface_transforms[global_interface_id]; H_B_I = H_B_D @ H_D_I
        print("Calculated H_B_I (Base to Interface Transform).")
        try: H_I_B = np.linalg.inv(H_B_I); print("Calculated H_I_B (Interface to Base Transform).")
        except np.linalg.LinAlgError: print("FATAL: Cannot calculate H_I_B"); H_I_B = None
        if H_B_I is None: raise ValueError("H_B_I calculation failed.")
        boundary_stats_list = utils.load_yaml_file(config.PROBABILISTIC_STATS_YAML_PATH)
        if boundary_stats_list is None or not isinstance(boundary_stats_list, list) or not boundary_stats_list: raise ValueError(f"Probabilistic Stats YAML invalid")
        num_boundaries = len(boundary_stats_list); print(f"Loaded Probabilistic Boundary Stats with {num_boundaries} boundaries.")
    except Exception as e: print(f"FATAL: Error during configuration/data loading: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 1. Initialize Robot Start State ---
    print("\n" + "="*40); print("=== 1. Initializing Robot Start State ==="); print("="*40)
    q_init_rad = np.radians([0, -90, 0, -90, 0, 0]); q_init_norm, T_init_B, _ = utils.initialize_robot_cspace(q_init_rad)
    if q_init_norm is None: print("FATAL: Failed start state init."); sys.exit(1)
    print(f"Start Config (q0, deg): {np.round(np.degrees(q_init_norm), 1)}")

    # --- Instantiate Planner ---
    print("\nInstantiating OMPL Planner..."); ompl_planner = OMPLPlanner()
    if ompl_planner.si is None: print("FATAL: OMPL Planner setup failed."); sys.exit(1)
    print("Planner instantiated successfully.")

    # --- 2. IK Branch Following Planning Loop ---
    print("\n" + "="*40); print("=== 2. Running Probabilistic Planning with IK Branch Following ==="); print("="*40)
    total_start_time = time.time(); all_branch_results = {}
    ik_branch_tuples = list(itertools.product([0, 1], repeat=3))
    print(f"Attempting to plan for {len(ik_branch_tuples)} IK branches...")
    if config.ENABLE_GMM_OPTIMIZATION: print("GMM Waypoint Optimization: ENABLED")
    else: print("GMM Waypoint Optimization: DISABLED")

    for branch_tuple in ik_branch_tuples:
        print("\n" + "-"*30); print(f"--- Attempting Branch: {branch_tuple} ---")
        branch_start_time = time.time()
        branch_path_segments = []; branch_q_goals = [q_init_norm]; branch_T_goals = [T_init_B]
        branch_total_cost = 0.0; branch_successful = True; q_current_branch = q_init_norm
        branch_optimization_infos = []

        for i in range(num_boundaries):
            boundary_data = boundary_stats_list[i]; boundary_index = boundary_data.get('time_index', i)
            print(f"\n  Segment {i+1} (Target: Boundary {i+1} @ {boundary_index}, Branch {branch_tuple})")
            gmm_params = boundary_data.get('segment_gmm_params')
            if gmm_params is None: print(f"  ERROR: Missing GMM params. Branch failed."); branch_successful = False; break

            q_goal_i_branch = None; T_ik_check_branch = None
            segment_cost = float('inf'); opt_info_boundary = {}

            if config.ENABLE_GMM_OPTIMIZATION:
                if not callable(run_gmm_waypoint_optimization): print(f"  ERROR: run_gmm_waypoint_optimization not available. Branch failed."); opt_info_boundary = {'algorithm': 'GMM_OPT_FUNC_MISSING'}; branch_successful = False; break
                print(f"  Running GMM optimization for waypoint...")
                z_opt, q_opt_branch, cost_opt, opt_info_boundary = run_gmm_waypoint_optimization(gmm_params, branch_tuple, q_current_branch, ompl_planner, H_B_I)
                opt_info_boundary['boundary_data'] = boundary_data

                if z_opt is not None and q_opt_branch is not None and np.isfinite(cost_opt):
                    print(f"  GMM Optimization successful. Min Cost (incl GMM term): {cost_opt:.4f}")
                    q_goal_i_branch = q_opt_branch
                    T_ik_check_branch = None # Initialize before try block
                    try: _, T_ik_check_branch, _ = utils.forward_kinematics(q_goal_i_branch)
                    except Exception as e_fk: print(f"  ERROR: FK call failed for optimized q. Branch failed. {e_fk}"); branch_successful = False; break
                    if T_ik_check_branch is None: print(f"  ERROR: FK returned None for optimized q. Branch failed."); branch_successful = False; break
                    path_seg_opt, segment_cost = ompl_planner.plan(q_current_branch, q_goal_i_branch, time_limit=config.INTER_WAYPOINT_PLANNING_TIME)
                    if path_seg_opt is None or not np.isfinite(segment_cost): print(f"  ERROR: Final planning step failed. Branch failed."); branch_successful = False; break
                    branch_path_segments.append(path_seg_opt)
                else: print(f"  ERROR: GMM optimization failed. Branch failed. Status: {opt_info_boundary.get('algorithm','UNKNOWN')}"); branch_successful = False; break
            else:
                T_target_B_i = utils.extract_target_pose_from_gmm(gmm_params, config.PROB_WAYPOINT_METHOD, H_B_I)
                if T_target_B_i is None: print(f"  ERROR: Failed to extract GMM target pose. Branch failed."); branch_successful = False; break
                q_goal_i_branch, T_ik_check_branch = utils.get_ik_solution_for_branch(T_target_B_i, branch_tuple, q_current_branch)
                if q_goal_i_branch is None: print(f"  ERROR: No valid IK solution for branch. Branch failed."); branch_successful = False; break
                path_seg_simple, path_cost_simple = ompl_planner.plan(q_current_branch, q_goal_i_branch, time_limit=config.INTER_WAYPOINT_PLANNING_TIME)
                if path_seg_simple is None: print(f"  ERROR: OMPL planning failed (simple). Branch failed."); branch_successful = False; break
                if not np.isfinite(path_cost_simple): print(f"  ERROR: OMPL non-finite cost (simple). Branch failed."); branch_successful = False; break
                segment_cost = path_cost_simple
                opt_info_boundary = {'algorithm': f'PROB_SIMPLE_{config.PROB_WAYPOINT_METHOD}', 'target_pose_B': T_target_B_i, 'boundary_data': boundary_data}
                branch_path_segments.append(path_seg_simple)

            if branch_successful:
                branch_q_goals.append(q_goal_i_branch); branch_T_goals.append(T_ik_check_branch)
                branch_total_cost += segment_cost; branch_optimization_infos.append(opt_info_boundary)
                q_current_branch = q_goal_i_branch
            else: branch_optimization_infos.append(opt_info_boundary); break

        if branch_successful:
            print(f"--- Branch {branch_tuple} SUCCEEDED (Accumulated Plan Cost: {branch_total_cost:.4f}, Time: {time.time() - branch_start_time:.2f}s) ---")
            all_branch_results[branch_tuple] = {'path_segments': branch_path_segments, 'cost': branch_total_cost, 'q_goals': branch_q_goals, 'T_goals': branch_T_goals, 'opt_infos': branch_optimization_infos}
        else: print(f"--- Branch {branch_tuple} FAILED (Time: {time.time() - branch_start_time:.2f}s) ---")

    # --- 3. Select Best Branch ---
    # (Keep existing logic - unchanged)
    print("\n" + "="*40); print("=== 3. Selecting Best Branch ==="); print("="*40)
    best_branch_tuple = None; min_overall_cost = float('inf'); planning_successful = False
    if not all_branch_results: print("FATAL: No IK branches resulted in a successful full path plan."); all_path_segments = []; all_goal_configs_q = [q_init_norm]; all_goal_poses_T = [T_init_B]; all_optimization_info = [{'algorithm': 'PROB_BRANCH_ALL_FAILED'}]
    else:
        planning_successful = True
        for branch, result in all_branch_results.items():
            if np.isfinite(result['cost']): print(f"  Branch {branch}: Cost = {result['cost']:.4f}");
            if result['cost'] < min_overall_cost: min_overall_cost = result['cost']; best_branch_tuple = branch
            else: print(f"  Branch {branch}: Cost = {result['cost']} (INVALID - Ignored)")
        if best_branch_tuple is None: print("FATAL: All successful branches had non-finite costs."); planning_successful = False; all_path_segments = []; all_goal_configs_q = [q_init_norm]; all_goal_poses_T = [T_init_B]; all_optimization_info = [{'algorithm': 'PROB_BRANCH_ALL_INF_COST'}]
        else: print(f"\nSelected Best Branch: {best_branch_tuple} (Cost: {min_overall_cost:.4f})"); best_result = all_branch_results[best_branch_tuple]; all_path_segments = best_result['path_segments']; all_goal_configs_q = best_result['q_goals']; all_goal_poses_T = best_result['T_goals']; all_optimization_info = best_result['opt_infos']
    print(f"\nTotal Branch Exploration & Selection time: {time.time() - total_start_time:.2f} sec")

    # --- 4. Process Full Path & Calculate Segment Boundaries (from best branch) ---
    # (Keep existing logic - unchanged)
    full_path_np = None; segment_boundary_indices = [0]
    if not all_path_segments: print("\nNo path segments available from best branch.")
    else:
        print("\nConcatenating path segments from best branch...")
        path_list_to_stack = []; current_index = 0
        first_seg = all_path_segments[0]
        if first_seg is not None and len(first_seg) > 0: path_list_to_stack.append(first_seg); current_index = len(first_seg); segment_boundary_indices.append(current_index)
        for i in range(1, len(all_path_segments)):
            seg_path = all_path_segments[i]
            if seg_path is None or len(seg_path) == 0: segment_boundary_indices.append(current_index); continue
            if path_list_to_stack and len(seg_path) > 0:
                 prev_last_point = path_list_to_stack[-1][-1]
                 if np.allclose(seg_path[0], prev_last_point, atol=1e-6):
                      if len(seg_path) > 1: path_list_to_stack.append(seg_path[1:]); current_index += (len(seg_path) - 1)
                 else: path_list_to_stack.append(seg_path); current_index += len(seg_path)
            elif seg_path is not None and len(seg_path) > 0: path_list_to_stack.append(seg_path); current_index += len(seg_path)
            segment_boundary_indices.append(current_index)
        if path_list_to_stack:
            try:
                full_path_np = np.vstack(path_list_to_stack)
                if segment_boundary_indices and full_path_np is not None and segment_boundary_indices[-1] != len(full_path_np): segment_boundary_indices[-1] = len(full_path_np)
                print(f"Full path generated ({full_path_np.shape[0]} waypoints). Boundaries: {segment_boundary_indices}")
            except ValueError as e: print(f"Error concatenating path segments: {e}"); full_path_np = None
        else: print("Error concatenating path: No valid segments to stack."); full_path_np = None

    # --- Extract Segment Statistics (Dummy for Viz) ---
    segment_stats_list = [{} for _ in range(len(all_path_segments))]

    # --- 5. Final Results Summary (from best branch) ---
    # (Keep existing logic - unchanged)
    print("\n" + "="*40); print(f"=== Planning Result (Best Branch: {best_branch_tuple}) ==="); print("="*40)
    if full_path_np is not None: print(f"Waypoints in Full Path: {full_path_np.shape[0]}")
    if all_goal_configs_q: print(f"Final Config q* (deg): {np.round(np.degrees(all_goal_configs_q[-1]), 2)}")
    if all_goal_poses_T: print(f"Final Pose T* (Base):\n{np.round(all_goal_poses_T[-1], 3)}")

    # --- 5b. Final Long-Term Path Refinement (OMPL Smoothing) ---
    # (Keep existing logic - unchanged)
    final_refined_path_np = None
    if planning_successful and len(all_goal_configs_q) > 1:
        print("\n" + "="*40); print(f"=== 5b. Performing Final Path Refinement (OMPL Smoothing) ==="); print(f"=== Planning Time Limit Per Segment: {config.INTER_WAYPOINT_PLANNING_TIME}s ==="); print("="*40)
        refined_segments = []; total_refined_cost = 0; refinement_successful = True
        for i in range(len(all_goal_configs_q) - 1):
            q_start_refined = all_goal_configs_q[i]; q_goal_refined = all_goal_configs_q[i+1]
            print(f"Refining segment {i+1} (q{i}* -> q{i+1}*)...")
            refined_path_seg, refined_cost_seg = ompl_planner.plan(q_start_refined, q_goal_refined, time_limit=config.INTER_WAYPOINT_PLANNING_TIME)
            if refined_path_seg is None: print(f"ERROR: Refinement FAILED segment {i+1}. Aborting."); refinement_successful = False; break
            if not np.isfinite(refined_cost_seg): print(f"ERROR: Refinement segment {i+1} non-finite cost. Aborting."); refinement_successful = False; break
            print(f"  Segment {i+1} refinement OK (Cost: {refined_cost_seg:.4f})."); refined_segments.append(refined_path_seg); total_refined_cost += refined_cost_seg
        if refinement_successful and refined_segments:
            print("\nConcatenating refined path segments...")
            refined_path_list = []
            if refined_segments[0] is not None and len(refined_segments[0]) > 0: refined_path_list.append(refined_segments[0])
            for j in range(1, len(refined_segments)):
                 current_seg = refined_segments[j]
                 if not refined_path_list:
                      if current_seg is not None and len(current_seg)>0: refined_path_list.append(current_seg); continue
                 prev_last_point = refined_path_list[-1][-1]
                 if current_seg is not None and len(current_seg) > 0:
                      if len(current_seg) > 1 and np.allclose(current_seg[0], prev_last_point, atol=1e-6): refined_path_list.append(current_seg[1:])
                      elif not np.allclose(current_seg[0], prev_last_point, atol=1e-6): refined_path_list.append(current_seg)
            if refined_path_list:
                try: final_refined_path_np = np.vstack(refined_path_list); print(f"Final refined path generated ({final_refined_path_np.shape[0]} waypoints, Total Refined Cost: {total_refined_cost:.4f}).")
                except ValueError as e: print(f"Error concatenating refined segments: {e}"); final_refined_path_np = None
            else: print("Warning: Refinement loop completed but no valid segments generated."); final_refined_path_np = None
        else: print("Final path refinement incomplete or failed."); final_refined_path_np = None
    # ... (rest of skipping logic) ...

    # --- 6. Visualization ---
    # (Keep existing logic - uses updated visualizer)
    print("\n" + "="*40); print("=== 6. Visualization ==="); print("="*40)
    path_to_visualize_cspace = final_refined_path_np if final_refined_path_np is not None else full_path_np
    approach_label = f"ProbBranch{best_branch_tuple}" if best_branch_tuple else "ProbBranchFAIL"
    refinement_label = "Refined" if final_refined_path_np is not None else "Initial"
    path_label_suffix = f"({approach_label}_{refinement_label})"
    print(f"Visualizing Paths (All Successful Branches + Best Highlighted): {path_label_suffix}")
    # 6.1 C-Space Path
    if path_to_visualize_cspace is not None: visualizer.plot_cspace_path(path_to_visualize_cspace, title=f"Best Branch C-Space Path {path_label_suffix}")
    else: print("Skipping C-Space plot: No best path available.")
    # 6.2 3D Scene
    vis3d = visualizer.Visualizer3D(f"Robot Motion - All Branches {path_label_suffix}")
    vis3d.plot_robot_config(q_init_norm, color='black', linewidth=3, style='-', label='Initial Pose (q0)')
    distinct_colors = plt.cm.tab10.colors; num_colors = len(distinct_colors); color_idx = 0; plotted_branches = 0
    for branch, result in all_branch_results.items():
        if not np.isfinite(result['cost']): continue
        branch_label = f"Branch {branch}"; branch_color = distinct_colors[color_idx % num_colors]; color_idx += 1
        # Robust Concatenation for Plotting
        branch_full_path = None; branch_seg_list = []
        current_branch_segments = result.get('path_segments', [])
        if current_branch_segments:
            first_seg = current_branch_segments[0]
            if first_seg is not None and len(first_seg) > 0: branch_seg_list.append(first_seg)
            for i in range(1, len(current_branch_segments)):
                seg_path = current_branch_segments[i]
                if branch_seg_list and seg_path is not None and len(seg_path) > 0:
                    prev_last_point = branch_seg_list[-1][-1]
                    if np.allclose(seg_path[0], prev_last_point, atol=1e-6):
                        if len(seg_path) > 1: branch_seg_list.append(seg_path[1:])
                    else: branch_seg_list.append(seg_path)
                elif seg_path is not None and len(seg_path) > 0: branch_seg_list.append(seg_path)
        if branch_seg_list:
            try: branch_full_path = np.vstack(branch_seg_list)
            except ValueError: branch_full_path = None
        if branch_full_path is not None:
            plotted_branches += 1; is_best = (branch == best_branch_tuple)
            line_width = 3.0 if is_best else 1.5; line_style = '-' if is_best else '--'; zorder = 10 if is_best else 5
            print(f"  Plotting 3D trajectory for Branch {branch} (Length: {len(branch_full_path)})...")
            path_plot_label = f"{branch_label} TCP Path"
            vis3d.plot_trajectory(branch_full_path, num_snapshots=0, color=branch_color, style=line_style, label_prefix=path_plot_label, plot_tcp_frames=False, linewidth=line_width)
            for i in range(1, len(result['q_goals'])):
                q_wp = result['q_goals'][i]; wp_label = ""
                if i == 1: wp_label = f"{branch_label} WP Start"
                elif i == len(result['q_goals']) - 1: wp_label = f"{branch_label} WP End"
                vis3d.plot_tcp_waypoint(q_wp, color=branch_color, marker='.', size=30, label=wp_label)
        else: print(f"  Skipping plotting for Branch {branch} - path concatenation failed.")
    if best_branch_tuple and all_goal_configs_q:
         vis3d.plot_tcp_waypoint(all_goal_configs_q[-1], color='red', marker='*', size=150, label=f'Best Branch Final Goal (q*)')
         vis3d.plot_robot_config(all_goal_configs_q[-1], color='purple', linewidth=2, style='-', label=f'Best Branch Final Config')
    elif plotted_branches == 0: vis3d.ax.text2D(0.5, 0.5, "No successful branches found to plot", ha='center', va='center', transform=vis3d.ax.transAxes, color='red')
    # 6.3 Task Space Components
    print("\nCalculating TCP Trajectory & Pose Components in Interface Frame (Best Branch)...")
    tcp_pos_path_interface_np = None; tcp_rot_matrices_interface_list = []; segment_start_end_pose_components = []
    path_for_interface_viz = final_refined_path_np if final_refined_path_np is not None else full_path_np
    if path_for_interface_viz is not None and H_I_B is not None:
        final_segment_boundary_indices = [0]
        if best_branch_tuple and best_branch_tuple in all_branch_results and len(all_branch_results[best_branch_tuple]['path_segments']) > 0:
             accum_len = 0; path_indices = [len(seg) if seg is not None else 0 for seg in all_branch_results[best_branch_tuple]['path_segments']]
             if path_indices[0] > 0: accum_len = path_indices[0]; final_segment_boundary_indices.append(accum_len)
             for length in path_indices[1:]:
                  if length > 1: accum_len += (length - 1)
                  final_segment_boundary_indices.append(accum_len)
             if final_segment_boundary_indices and final_segment_boundary_indices[-1] != len(path_for_interface_viz): final_segment_boundary_indices[-1] = len(path_for_interface_viz)
        tcp_path_interface_pos = []; fk_errors = 0; transform_errors = 0; current_segment_start_components = None
        if IK_SOLVER_AVAILABLE:
            for idx, q_waypoint in enumerate(path_for_interface_viz):
                try:
                    _, T_B_TCP, _ = forward_kinematics(q_waypoint)
                    if T_B_TCP is None: fk_errors += 1; continue
                    T_I_TCP = H_I_B @ T_B_TCP; pos_I = T_I_TCP[:3, 3]; rot_I = T_I_TCP[:3, :3]
                    tcp_path_interface_pos.append(pos_I); tcp_rot_matrices_interface_list.append(rot_I)
                    current_components_flat_rot = np.concatenate((pos_I, rot_I.flatten()))
                    if idx in final_segment_boundary_indices[:-1]: current_segment_start_components = current_components_flat_rot
                    if idx + 1 in final_segment_boundary_indices[1:]:
                        try: # <--- NESTED TRY BLOCK
                            # Find the segment index corresponding to this end boundary index
                            seg_idx = final_segment_boundary_indices.index(idx + 1) - 1
                            # Get the start waypoint index for this segment
                            start_idx_val = final_segment_boundary_indices[seg_idx]
                            # Append start/end component info IF start components were captured
                            if current_segment_start_components is not None:
                                segment_start_end_pose_components.append({
                                    'segment_index': seg_idx,
                                    'start_idx': start_idx_val,
                                    'end_idx': idx, # Current index is the end of the segment
                                    'start_components': current_segment_start_components,
                                    'end_components': current_components_flat_rot
                                })
                        except ValueError:
                             # This can happen if idx+1 is not found in the boundary list,
                             # which might indicate an issue with boundary calculation.
                             # print(f"Debug: Could not find boundary index {idx+1} in list.") # Optional Debug
                             pass # Silently ignore if index not found

                except Exception as e_tf: transform_errors += 1; continue
            if fk_errors > 0: print(f"Warn: FK failed {fk_errors} times (IF viz).")
            if transform_errors > 0: print(f"Warn: Transform failed {transform_errors} times (IF viz).")
        else: print("Skipping IF component calculation: FK unavailable.")
        if tcp_path_interface_pos:
            tcp_pos_path_interface_np = np.array(tcp_path_interface_pos)
            print(f"Calculated TCP pos trajectory in IF ({tcp_pos_path_interface_np.shape[0]} points).")
            visualizer.plot_task_space_trajectory(tcp_pos_path_interface_np, title=f"TCP Trajectory in IF '{global_interface_id}' {path_label_suffix}", interface_frame_id=global_interface_id)
            if tcp_rot_matrices_interface_list: visualizer.plot_interface_frame_pose_components(tcp_pos_path_interface_np, tcp_rot_matrices_interface_list, final_segment_boundary_indices, segment_stats_list, segment_start_end_pose_components, title=f"TCP Pose Components (Pos + LogMap) vs Waypoint Index (IF: '{global_interface_id}') {path_label_suffix}")
            else: print("Could not plot pose components: Rotation matrix list empty.")
        else: print("Could not generate TCP trajectory/components in IF.")
    # ... (rest of skipping logic) ...

    # --- NEW: Plot GMM Optimization Details for Best Branch ---
    if config.ENABLE_GMM_OPTIMIZATION and best_branch_tuple and best_branch_tuple in all_branch_results:
        print("\nPlotting GMM Optimization Details for Best Branch...")
        best_branch_opt_infos = all_branch_results[best_branch_tuple].get('opt_infos', [])
        for boundary_idx, opt_info_boundary in enumerate(best_branch_opt_infos):
             if boundary_idx < len(boundary_stats_list):
                 gmm_params_boundary = boundary_stats_list[boundary_idx].get('segment_gmm_params')
                 if gmm_params_boundary:
                     if hasattr(visualizer, 'plot_gmm_optimization_details'):
                         visualizer.plot_gmm_optimization_details(opt_info=opt_info_boundary, gmm_params=gmm_params_boundary, branch_tuple=best_branch_tuple, boundary_index=boundary_idx, H_B_I=H_B_I)
                     else: print(f"Warning: visualizer.plot_gmm_optimization_details function not found.")
                 else: print(f"Warning: Could not find GMM params for boundary {boundary_idx} to plot opt details.")
             else: print(f"Warning: Boundary index {boundary_idx} out of range for plotting opt details.")
    elif config.ENABLE_GMM_OPTIMIZATION: print("\nSkipping GMM Optimization plot: No successful best branch found.")
    else: print("\nSkipping GMM Optimization plot: Optimization disabled.")

    # Add legend and show plots
    vis3d.add_legend(loc='best')
    if plt.get_fignums(): print("\nDisplaying all plots... Close plot windows to exit."); plt.show()
    else: print("No plots were generated.")

    # --- 7. Save Final Trajectory (Best Branch) ---
    # (Keep existing logic - unchanged)
    print("\n" + "="*40); print("=== 7. Saving Final Trajectory (Best Branch) ==="); print("="*40)
    path_to_save = final_refined_path_np if final_refined_path_np is not None else full_path_np
    save_label_suffix = f"({approach_label}_{refinement_label})"
    if path_to_save is not None:
        trajectory_filename = "planned_trajectory.npy"
        try: np.save(trajectory_filename, path_to_save); print(f"Saved final trajectory {save_label_suffix} ({path_to_save.shape[0]} waypoints) to: {trajectory_filename}")
        except Exception as e: print(f"ERROR saving trajectory: {e}")
    else: print("No final trajectory generated to save.")

    print("\nmain.py finished.")


# --- Entry Point ---
if __name__ == "__main__":
    # Dependency checks
    if not utils.SCIPY_AVAILABLE: print("\nFATAL: Scipy required. Install with 'pip install scipy'."); sys.exit(1)
    if not utils.IK_SOLVER_AVAILABLE: print("\nFATAL: IK Solver functions required but not found."); sys.exit(1)
    if config.ENABLE_GMM_OPTIMIZATION and config.GMM_OPT_ALGORITHM == "PSO" and not PY_SWARMS_AVAILABLE:
         print("\nFATAL: GMM PSO Optimization enabled, but pyswarms not found. Install with 'pip install pyswarms'."); sys.exit(1)
    # Run main scenario
    run_planning_scenario()
