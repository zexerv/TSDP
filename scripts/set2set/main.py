# main.py
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import config
import utils
from planner import OMPLPlanner
import visualizer
# Import optimization functions and SciPy check/class
from optimization import run_optimization, evaluate_target_pose, SCIPY_AVAILABLE
from ik_solver import forward_kinematics # For verification if needed

# Import Rotation if SciPy is available
if SCIPY_AVAILABLE:
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        print("Warning: (main.py) SCIPY_AVAILABLE was True, but failed to import Rotation.")
        SCIPY_AVAILABLE = False

print("Starting main.py execution...")


# --- Helper Function for a Single Segment ---
def find_optimal_target_in_set(q_segment_start, segment_index, set_definition, planner):
    """
    Performs sampling and refinement for a single planning segment.

    Args:
        q_segment_start (np.ndarray): Start configuration for this segment.
        segment_index (int): The index of the current segment (0 for S1, 1 for S2, etc.).
        set_definition (dict): Configuration dictionary for the target set (from config.TARGET_SET_DEFINITIONS).
        planner (OMPLPlanner): The instantiated OMPL planner.

    Returns:
        tuple: (path_segment, q_goal_final, T_goal_final, opt_info)
               Returns (None, None, None, None) on failure.
    """
    set_name = set_definition.get('name', f'Set{segment_index+1}')
    num_samples = set_definition.get('num_samples', 10) # Default samples if not specified
    sampling_plan_time_limit = set_definition.get('segment_planning_time_limit', 0.03) # Default time

    print("\n" + "="*40)
    print(f"=== Segment {segment_index + 1}: Finding Target in {set_name} from q{segment_index}* ===")
    print(f"=== Start Config (q{segment_index}*, deg): {np.round(np.degrees(q_segment_start), 1)}")
    print("="*40)

    # 1. Define Target Set Si
    print(f"Defining {set_name}...")
    base_q_deg = set_definition.get('base_q_deg')
    if base_q_deg is None: print(f"ERROR: 'base_q_deg' not defined for {set_name}"); return None, None, None, None
    base_q_rad = np.radians(base_q_deg)

    # Note: Perturbing based on a fixed base_q might lead sets far from the actual segment start.
    # Consider perturbing relative to q_segment_start FK pose if needed later.
    q_center_si = utils.generate_perturbed_config(base_q_rad,
                                                set_definition.get('perturb_min_deg', -10),
                                                set_definition.get('perturb_max_deg', 10))
    if q_center_si is None: q_center_si = base_q_rad # Use base if perturbation fails

    current_set_data = utils.define_task_space_set(q_center_si,
                                                 set_definition.get('pos_interval_m', [0.05]*3),
                                                 set_definition.get('rot_interval_rad', 0.1))
    if current_set_data is None: print(f"FATAL: Failed to define {set_name}."); return None, None, None, None
    print(f"Defined {set_name} around q_center (deg): {np.round(np.degrees(current_set_data['q_center']), 1)}")
    current_set_data['color'] = set_definition.get('color', 'gray') # Store color for vis


    # 2. Sampling Loop within Si
    print(f"\nSampling {set_name} ({num_samples} samples, Plan Time Limit: {sampling_plan_time_limit}s)...")
    best_cost_sampling = float('inf')
    best_T_target_sampling = None
    best_q_goal_sampling = None
    best_path_np_sampling = None
    best_joint_pos_goal_sampling = None # Store joint positions if needed by visualizer
    successful_plans = 0
    ik_failures = 0
    sampling_failures = 0
    start_sampling_time = time.time()

    for i in range(num_samples):
        T_target_i = utils.sample_pose_from_set(current_set_data)
        if T_target_i is None: sampling_failures += 1; continue

        # Use set center as hint initially
        # Pass q_segment_start as hint? Maybe better for closeness? Let's use set center for now.
        q_goal_i, _, joint_pos_goal_i = utils.initialize_robot_taskspace(
            T_target_i, current_config_hint=current_set_data.get('q_center'))

        if q_goal_i is None: ik_failures += 1; continue

        # Plan from the start of THIS segment (q_segment_start)
        path_np_i, path_cost_i = planner.plan(q_segment_start, q_goal_i, time_limit=sampling_plan_time_limit)

        if path_np_i is not None and np.isfinite(path_cost_i):
            successful_plans += 1
            if path_cost_i < best_cost_sampling:
                # print(f"  Sample {i+1}: New best sampling cost for {set_name}: {path_cost_i:.4f}") # Less verbose
                best_cost_sampling = path_cost_i
                best_path_np_sampling = path_np_i
                best_q_goal_sampling = q_goal_i
                best_T_target_sampling = T_target_i
                best_joint_pos_goal_sampling = joint_pos_goal_i
        # else: planning failed

    print(f"Sampling finished ({time.time() - start_sampling_time:.2f} sec). Best cost: {best_cost_sampling:.4f}. Success rate: {successful_plans}/{num_samples - sampling_failures - ik_failures} planning attempts.")

    # Initialize segment results with sampling best
    seg_final_cost = best_cost_sampling
    seg_final_T_target = best_T_target_sampling
    seg_final_q_goal = best_q_goal_sampling
    seg_final_path_np = best_path_np_sampling
    seg_final_joint_pos_goal = best_joint_pos_goal_sampling # If needed
    seg_optimization_info = {'algorithm': 'SamplingOnly'} # Default info

    # 3. Refinement Step (GD or PSO)
    if seg_final_path_np is not None and np.isfinite(seg_final_cost):
        print(f"\nAttempting refinement for {set_name} using '{config.OPTIMIZATION_ALGORITHM}' algorithm...")
        # Call the global run_optimization dispatcher
        refined_T, refined_cost, seg_optimization_info = run_optimization(
            seg_final_T_target,     # Initial guess T from sampling
            seg_final_cost,         # Initial guess cost
            q_segment_start,        # Start config q FOR THIS SEGMENT
            planner,                # Planner instance
            current_set_data        # Target set info FOR THIS SEGMENT
        )

        ran_algorithm = seg_optimization_info.get('algorithm', 'UNKNOWN')
        print(f"Refinement Algorithm Run for {set_name}: {ran_algorithm}")

        refinement_succeeded = 'FAILED' not in ran_algorithm and 'DISABLED' not in ran_algorithm and 'UNKNOWN' not in ran_algorithm
        found_better_solution = refinement_succeeded and np.isfinite(refined_cost) and refined_cost < best_cost_sampling

        if found_better_solution:
            print(f"{ran_algorithm} found better solution for {set_name} (Cost: {refined_cost:.5f} < Sampling: {best_cost_sampling:.5f})")
            final_best_cost_candidate = refined_cost
            final_best_T_target_candidate = refined_T

            # Re-evaluate the refined pose with the FINAL planning time for consistency
            print(f"Re-evaluating final {ran_algorithm} pose for {set_name} (using {config.FINAL_EVAL_PLANNING_TIME_LIMIT}s limit)...")
            final_path_np_refined, final_cost_check = evaluate_target_pose(
                 final_best_T_target_candidate, q_segment_start, planner, # Plan from segment start
                 hint_q=seg_final_q_goal, # Hint with sampling goal for this segment
                 eval_time_limit=config.FINAL_EVAL_PLANNING_TIME_LIMIT # Use final eval time
            )
            final_q_goal_refined, _, final_joint_pos_goal_refined = utils.initialize_robot_taskspace(
                final_best_T_target_candidate, current_config_hint=seg_final_q_goal
            )

            # Accept ONLY if final evaluation is successful
            if final_path_np_refined is not None and final_q_goal_refined is not None and np.isfinite(final_cost_check):
                print(f"Successfully updated {set_name} results with {ran_algorithm} refinement.")
                seg_final_path_np = final_path_np_refined
                seg_final_q_goal = final_q_goal_refined
                seg_final_joint_pos_goal = final_joint_pos_goal_refined
                seg_final_T_target = final_best_T_target_candidate
                seg_final_cost = final_cost_check # Use cost from final consistent evaluation
                print(f"Final {set_name} Cost (Re-evaluated): {seg_final_cost:.5f}")
            else:
                print(f"Warning: Failed final re-evaluation for {ran_algorithm}/{set_name} result. Keeping sampling result.")
                # Revert to sampling best if final evaluation fails
        else:
             print(f"{ran_algorithm} did not provide better result or failed/disabled for {set_name}. Keeping sampling result.")
             # Keep sampling results

    elif seg_final_path_np is None or not np.isfinite(seg_final_cost):
         print(f"Skipping Optimization Step for {set_name}: no valid initial path found by sampling.")
         return None, None, None, None # Indicate segment failure

    # 4. Return results for this segment
    if seg_final_path_np is None:
         print(f"ERROR: Failed to find any valid path for segment {segment_index+1}.")
         return None, None, None, seg_optimization_info # Return failure but include info

    print(f"\nFinished Segment {segment_index + 1}. Final Cost: {seg_final_cost:.4f}")
    print(f"Optimal Goal q{segment_index + 1}* (deg): {np.round(np.degrees(seg_final_q_goal), 1)}")
    print(f"Optimal Target T{segment_index + 1}*:\n{np.round(seg_final_T_target[:3,:], 3)}")

    # Add set data to optimization info for plotting later
    seg_optimization_info['set_data'] = current_set_data

    return seg_final_path_np, seg_final_q_goal, seg_final_T_target, seg_optimization_info


# --- Main Scenario Function ---
def run_planning_scenario():
    """Defines and runs the multi-segment motion planning scenario."""

    # --- 1. Initialize Robot Start State ---
    print("\n" + "="*40)
    print("=== 1. Initializing Robot Start State ===")
    print("="*40)
    q_init_rad = np.radians([0, -90, 0, -90, 0, 0]) # Example start
    # --- Randomize Start ---
    # q_init_min_deg = np.array([-180, -180, -150, -180, -180, -180])
    # q_init_max_deg = np.array([ 180,    0,  150,    0,  180,  180])
    # q_init_deg_random = np.random.uniform(low=q_init_min_deg, high=q_init_max_deg, size=config.NUM_JOINTS)
    # q_init_rad = np.radians(q_init_deg_random)
    # print(f"Generated Random Start Config (degrees): {np.round(q_init_deg_random, 2)}")
    # --- End Randomize ---

    q_init_norm, T_init, joint_pos_init = utils.initialize_robot_cspace(q_init_rad)
    if q_init_norm is None: print("FATAL: Failed to initialize start state."); sys.exit(1)
    print(f"Start Config (q_init, deg): {np.round(np.degrees(q_init_norm), 1)}")


    # --- Instantiate Planner ---
    print("\nInstantiating OMPL Planner...")
    ompl_planner = OMPLPlanner() # Assuming this class exists and works
    if ompl_planner.si is None: print("FATAL: OMPL Planner setup failed."); sys.exit(1)
    print("Planner instantiated successfully.")


    # --- 2. Multi-Segment Planning Loop ---
    num_segments = config.NUM_PLANNING_SEGMENTS
    if num_segments <= 0 or len(config.TARGET_SET_DEFINITIONS) < num_segments:
        print(f"FATAL: Invalid NUM_PLANNING_SEGMENTS ({num_segments}) or insufficient TARGET_SET_DEFINITIONS.")
        sys.exit(1)

    q_current = q_init_norm # Start from initial configuration
    all_path_segments = []
    all_goal_configs_q = [q_init_norm] # List of q0*, q1*, q2*, ...
    all_goal_poses_T = [T_init]        # List of T0, T1*, T2*, ...
    all_optimization_info = []       # List of info dicts from each segment

    total_start_time = time.time()

    for i in range(num_segments):
        segment_start_time = time.time()
        set_definition = config.TARGET_SET_DEFINITIONS[i]

        path_seg, q_goal_i, T_goal_i, opt_info_i = find_optimal_target_in_set(
            q_current, i, set_definition, ompl_planner
        )

        if path_seg is None:
            print(f"\nFATAL: Failed to plan segment {i+1}. Stopping.")
            # Optionally visualize the state up to failure here
            break # Exit the loop on failure

        # Store results
        all_path_segments.append(path_seg)
        all_goal_configs_q.append(q_goal_i)
        all_goal_poses_T.append(T_goal_i)
        all_optimization_info.append(opt_info_i)

        # Update start for next segment
        q_current = q_goal_i
        print(f"Segment {i+1} processing time: {time.time() - segment_start_time:.2f} sec")

    print(f"\nTotal multi-segment planning time: {time.time() - total_start_time:.2f} sec")

    # --- 3. Process Full Path ---
    full_path_np = None
    if not all_path_segments:
        print("\nNo path segments were successfully planned.")
    else:
        print("\nConcatenating path segments...")
        # Concatenate segments, removing duplicate start/end points
        path_list_to_stack = [all_path_segments[0]] # Start with the first segment
        for j in range(1, len(all_path_segments)):
             # Skip the first point of subsequent segments as it's the end of the previous one
             if len(all_path_segments[j]) > 1:
                  path_list_to_stack.append(all_path_segments[j][1:])
             elif len(all_path_segments[j]) == 1: # Segment only had one point? Add it.
                  path_list_to_stack.append(all_path_segments[j])

        if path_list_to_stack:
             full_path_np = np.vstack(path_list_to_stack)
             print(f"Full path generated with {full_path_np.shape[0]} waypoints.")
        else: print("Error during path concatenation.")


    # --- 4. Final Results Summary ---
    print("\n" + "="*40)
    print(f"=== Final Multi-Segment Result ({len(all_path_segments)} successful segments) ===")
    print("="*40)
    if full_path_np is None:
        print("FATAL: No valid final path generated.")
        sys.exit(1)

    # Calculate final cost (e.g., total length or sum of segment costs?)
    # For now, just report stats
    print(f"Total Waypoints: {full_path_np.shape[0]}")
    print(f"Final Configuration q{num_segments}* (deg): {np.round(np.degrees(all_goal_configs_q[-1]), 2)}")
    print(f"Final Target Pose T{num_segments}*:\n{np.round(all_goal_poses_T[-1], 3)}")
    if SCIPY_AVAILABLE:
        try:
            final_quat = R.from_matrix(all_goal_poses_T[-1][:3,:3]).as_quat()[[3,0,1,2]]
            print(f"Final Target Quat T{num_segments}* (wxyz): {np.round(final_quat, 4)}")
        except Exception as e: print(f"Could not calculate final quaternion: {e}")


    # --- 5. Visualization ---
    print("\n" + "="*40)
    print("=== 5. Visualization (Full Path & Sets) ===")
    print("="*40)

    # 5.1 C-Space Path (Full)
    visualizer.plot_cspace_path(full_path_np, title=f"Full C-Space Path ({num_segments} Segments)")

    # 5.2 3D Scene (Full)
    vis3d = visualizer.Visualizer3D(f"Multi-Segment Robot Motion ({num_segments} Segments)")
    # Plot Start Pose
    vis3d.plot_robot_config(all_goal_configs_q[0], color='black', linewidth=3, style='-', label='Initial Pose (q0*)')

    # Plot Target Sets and Intermediate Goals
    all_set_data_for_plot = []
    for i, opt_info in enumerate(all_optimization_info):
        set_data = opt_info.get('set_data')
        if set_data:
            all_set_data_for_plot.append(set_data) # Collect set data
            set_name = set_data.get('name', f'Set{i+1}')
            set_color = set_data.get('color', 'grey')
            vis3d.plot_task_space_set(set_data, color=set_color, alpha=0.1, label=f'{set_name} Bounds')

            # Plot intermediate goal pose q(i+1)*
            goal_q = all_goal_configs_q[i+1]
            vis3d.plot_robot_config(goal_q, color=set_color, linewidth=3, style='-.', alpha=0.8, label=f'Goal {set_name} (q{i+1}*)', plot_tcp_frame=True)

    # Plot Full Trajectory (maybe fewer snapshots?)
    if full_path_np is not None:
        total_snapshots = config.VIS_3D_SNAPSHOTS # Use global setting or adjust
        vis3d.plot_trajectory(full_path_np, num_snapshots=total_snapshots, color='purple', style='-', label_prefix='Full Path', plot_tcp_frames=False) # Don't plot TCP for every snapshot

    # Plot Optimization History (e.g., for the LAST segment if available)
    last_opt_info = all_optimization_info[-1] if all_optimization_info else None
    if last_opt_info:
         last_algo = last_opt_info.get('algorithm', 'UNKNOWN')
         print(f"Plotting optimization details for last segment ({last_algo})...")
         # Plot GD History
         if last_algo == 'GD' and last_opt_info.get('history'):
              hist_tuple = last_opt_info['history']
              if len(hist_tuple) == 4:
                   pose_hist, _, cost_hist, grad_hist = hist_tuple
                   if pose_hist and cost_hist: # Check if not empty
                        vis3d.plot_gd_progress(pose_hist, cost_hist, label=f'GD Path (Seg {num_segments})')
                        # Plot 2D GD plots if needed (pass history vars)
                        if grad_hist and last_opt_info.get('set_data'):
                            visualizer.plot_gd_progress_2d(pose_hist, cost_hist, grad_hist, last_opt_info['set_data'])
                        if last_opt_info.get('history')[1] and cost_hist: # Quat history
                             visualizer.plot_gd_orientation_progress(last_opt_info['history'][1], cost_hist)

         # Plot PSO Swarm
         elif 'final_swarm_positions' in last_opt_info and last_opt_info['final_swarm_positions'] is not None:
              swarm_pos = last_opt_info['final_swarm_positions']
              # Get the best position corresponding to the last goal T
              best_pos_xyz = all_goal_poses_T[-1][:3,3] if all_goal_poses_T else None
              if best_pos_xyz is not None:
                   vis3d.plot_pso_final_swarm(swarm_pos, best_pos_xyz)

    # Add legend and show
    vis3d.add_legend(loc='best')
    if plt.get_fignums():
        print("Displaying plots...")
        plt.show()
    else:
        print("No plots generated.")

    print("\nmain.py finished.")


if __name__ == "__main__":
    run_planning_scenario()