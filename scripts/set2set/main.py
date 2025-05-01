# main.py
import numpy as np
import sys
import time
import matplotlib.pyplot as plt # Make sure plt is imported
import config
import utils
from planner import OMPLPlanner
import visualizer # Import the module
import optimization # Import the optimization module
from ik_solver import forward_kinematics # For verification if needed

print("Starting main.py execution...")

# --- Configuration loaded from config.py ---
NUM_SAMPLES_S1 = config.NUM_SAMPLES_S1
ENABLE_GD_REFINEMENT = config.ENABLE_GD_REFINEMENT
# GD parameters are used within optimization.py

def run_planning_scenario():
    """Defines and runs the motion planning scenario."""

    # --- 1. Initialize Robot Start State ---
    print("\n" + "="*40)
    print("=== 1. Initializing Robot Start State ===")
    print("="*40)
    q_init_rad = np.radians([42, -73, 102, -116, -91, 1])
    q_init_norm, T_init, joint_pos_init = utils.initialize_robot_cspace(q_init_rad)
    if q_init_norm is None: sys.exit(1)
    print(f"Start Config (q_init, normalized deg): {np.round(np.degrees(q_init_norm), 2)}")

    # --- 2. Define Target Sets ---
    print("\n" + "="*40)
    print("=== 2. Defining Task Space Target Set (S1) ===")
    print("="*40)
    q_base_s1 = np.radians([60, -90, 110, -100, -90, 0])
    q_center_s1 = utils.generate_perturbed_config(q_base_s1, config.PERTURB_MIN_DEG, config.PERTURB_MAX_DEG)
    pos_interval_s1 = config.DEFAULT_POS_INTERVAL
    rot_interval_s1 = config.DEFAULT_ROT_INTERVAL
    set1_data = utils.define_task_space_set(q_center_s1, pos_interval_s1, rot_interval_s1)
    if set1_data is None: sys.exit(1)


    # --- 3. Optimization Loop (Random Sampling): Find Best Target in S1 ---
    print("\n" + "="*40)
    print(f"=== 3. Optimization Loop (Random Sampling): Searching for Best Target in S1 (Samples={NUM_SAMPLES_S1}) ===")
    print("="*40)
    best_cost_sampling = float('inf')
    best_T_target_sampling = None
    best_q_goal_sampling = None
    best_path_np_sampling = None
    best_joint_pos_goal_sampling = None
    successful_plans = 0
    ik_failures = 0
    sampling_failures = 0

    # Instantiate the planner ONCE
    ompl_planner = None
    if config.OMPL_AVAILABLE:
        ompl_planner = OMPLPlanner()
        if ompl_planner.si is None:
            print("FATAL: OMPL Planner setup failed. Exiting.")
            sys.exit(1)
    else:
         print("FATAL: OMPL is not available. Exiting.")
         sys.exit(1)

    start_loop_time = time.time()
    for i in range(NUM_SAMPLES_S1):
        T_target_i = utils.sample_pose_from_set(set1_data)
        if T_target_i is None:
            sampling_failures += 1; continue

        q_goal_i, _, joint_pos_goal_i = utils.initialize_robot_taskspace(
            T_target_i, current_config_hint=set1_data['q_center'])
        if q_goal_i is None:
            ik_failures += 1; continue

        path_np_i, path_cost_i = ompl_planner.plan(q_init_norm, q_goal_i, time_limit=config.PLANNING_TIME_LIMIT)

        if path_np_i is not None and path_cost_i < best_cost_sampling:
            print(f"  Sample {i+1}: Found new best sampling solution! Cost: {path_cost_i:.4f} (Improvement: {best_cost_sampling - path_cost_i:.4f})")
            best_cost_sampling = path_cost_i; best_path_np_sampling = path_np_i
            best_q_goal_sampling = q_goal_i; best_T_target_sampling = T_target_i
            best_joint_pos_goal_sampling = joint_pos_goal_i
            successful_plans += 1
        elif path_np_i is not None:
             successful_plans += 1

    end_loop_time = time.time()
    print("\n" + "="*40)
    print(f"=== Random Sampling Finished ({end_loop_time - start_loop_time:.2f} sec) ===")
    print(f"  Valid paths found: {successful_plans}/{NUM_SAMPLES_S1 - sampling_failures - ik_failures}")
    print(f"  (Total Samples: {NUM_SAMPLES_S1}, Sampling Failures: {sampling_failures}, IK Failures: {ik_failures})")
    print(f"  Best Sampling Cost: {best_cost_sampling:.4f}")
    print("=" * 40)

    # Initialize final results with the best from sampling
    final_best_cost = best_cost_sampling
    final_best_T_target = best_T_target_sampling
    final_best_q_goal = best_q_goal_sampling
    final_best_path_np = best_path_np_sampling
    final_best_joint_pos_goal = best_joint_pos_goal_sampling
    # Initialize history lists for GD results
    gd_pose_history_xyz, gd_quat_history_wxyz, gd_cost_history, gd_gradient_history_7d = None, None, None, None

    # --- 3.5 Gradient Descent Refinement ---
    if final_best_path_np is not None and ENABLE_GD_REFINEMENT:
        # Call the updated GD function which now handles 7D
        gd_refined_T, gd_refined_cost, gd_pose_history_xyz, gd_quat_history_wxyz, gd_cost_history, gd_gradient_history_7d = optimization.refine_target_pose_with_gd(
            final_best_T_target, final_best_cost, q_init_norm, ompl_planner, set1_data
        )

        # Check if GD found a *valid* and *better* solution
        if np.isfinite(gd_refined_cost) and len(gd_cost_history) > 0 and gd_refined_cost < gd_cost_history[0]: # Compare to re-evaluated cost
            print("GD found a better solution than re-evaluated sampling start point!")
            final_best_cost = gd_refined_cost
            final_best_T_target = gd_refined_T
            # Re-run IK and plan one last time for the final GD pose using default time
            print("Re-evaluating final GD pose for path and q_goal (using default time)...")
            final_path_np_gd, final_cost_check = optimization.evaluate_target_pose(
                final_best_T_target, q_init_norm, ompl_planner,
                eval_time_limit=config.PLANNING_TIME_LIMIT # Use default time
            )
            final_q_goal_gd, _, final_joint_pos_goal_gd = utils.initialize_robot_taskspace(final_best_T_target)

            if final_path_np_gd is not None and final_q_goal_gd is not None:
                final_best_path_np = final_path_np_gd
                final_best_q_goal = final_q_goal_gd
                final_best_joint_pos_goal = final_joint_pos_goal_gd
                final_best_cost = final_cost_check # Update cost to match final plan
                print(f"Successfully updated final results with GD refinement. Final Cost: {final_best_cost:.5f}")
            else:
                print("Warning: Failed to get final path/q_goal for GD result. Keeping sampling result.")
                final_best_cost = best_cost_sampling
                final_best_T_target = best_T_target_sampling
        else:
            print("GD did not find a better solution or failed.")
            # Keep the original sampling result as the final best if GD fails/doesn't improve
            final_best_cost = best_cost_sampling
            final_best_T_target = best_T_target_sampling
            final_best_q_goal = best_q_goal_sampling
            final_best_path_np = best_path_np_sampling
            final_best_joint_pos_goal = best_joint_pos_goal_sampling

    elif not ENABLE_GD_REFINEMENT:
         print("\nGD Refinement step was disabled.")
    elif final_best_path_np is None:
         print("\nSkipping GD Refinement: no initial path found by sampling.")


    # --- Final Results Summary ---
    print("\n" + "="*40)
    print(f"=== Final Optimal Result for Segment 1 ===")
    print("="*40)
    if final_best_path_np is None:
         print("FATAL: No valid path found after all optimization steps.")
         vis3d = visualizer.Visualizer3D("Planning Failed: q_init -> S1")
         vis3d.plot_robot_config(q_init_norm, color='black', linewidth=3, style='-', label='Initial Pose')
         vis3d.plot_task_space_set(set1_data, color='blue', alpha=0.1, label='Set 1 Bounds')
         vis3d.add_legend(loc='upper left')
         vis3d.show()
         sys.exit(1)
    print(f"Optimal Path Cost (Length): {final_best_cost:.4f}")
    print(f"Optimal Goal Config q1* (deg): {np.round(np.degrees(final_best_q_goal), 2)}")
    print(f"Optimal Target Pose p1*:\n{np.round(final_best_T_target, 3)}")
    if optimization.SCIPY_AVAILABLE: # Print final quaternion if possible
        final_quat = optimization.R.from_matrix(final_best_T_target[:3,:3]).as_quat()[[3,0,1,2]]
        print(f"Optimal Target Quat p1* (wxyz): {np.round(final_quat, 4)}")


    # --- 4. Visualization ---
    print("\n" + "="*40)
    print("=== 4. Visualization (Final Best Path + GD Progress) ===")
    print("="*40)

    # 4.1 C-Space Path
    visualizer.plot_cspace_path(final_best_path_np, title=f"Final Best Path (q_init -> q1*) - {final_best_path_np.shape[0]} points")

    # 4.2 3D Scene
    vis3d = visualizer.Visualizer3D("Robot Motion Planning: Final Path q_init -> S1* & GD")
    vis3d.plot_robot_config(q_init_norm, color='black', linewidth=3, style='-', label='Initial Pose')
    if final_best_q_goal is not None and final_best_joint_pos_goal is not None:
        vis3d.plot_robot_config(final_best_q_goal, color='lime', linewidth=4, style='-', label='Final Optimal Pose (p1*)')
    vis3d.plot_task_space_set(set1_data, color='blue', alpha=0.1, label='Set 1 Bounds')
    vis3d.plot_trajectory(final_best_path_np, num_snapshots=config.VIS_3D_SNAPSHOTS, color='purple', style='-', label_prefix='Final Path')
    if gd_pose_history_xyz and gd_cost_history: # Check if position history exists
        vis3d.plot_gd_progress(gd_pose_history_xyz, gd_cost_history, label='GD Target Path (3D)')
    vis3d.add_legend(loc='best')

    # 4.3 2D Position GD Projections
    if gd_pose_history_xyz and gd_cost_history and gd_gradient_history_7d:
         visualizer.plot_gd_progress_2d(gd_pose_history_xyz, gd_cost_history, gd_gradient_history_7d, set1_data)

    # 4.4 2D Orientation GD Projections (NEW)
    if gd_quat_history_wxyz and gd_cost_history:
         visualizer.plot_gd_orientation_progress(gd_quat_history_wxyz, gd_cost_history)


    # Show all plots
    plt.show()

    # --- 5. Placeholder for Next Steps ---
    print("\n" + "="*40)
    print("=== Planning Segment 1 (Optimized) Done ===")
    print("="*40)
    q1_star = final_best_path_np[-1]
    print(f"Optimal configuration q1* (deg): {np.round(np.degrees(q1_star), 2)}")
    print("\nNext Steps:")
    print("1. Implement Collision Checking in `planner.SimpleUR5eStateValidityChecker`.")
    print("2. Define and integrate custom OMPL Optimization Objectives (Manipulability, Smoothness).")
    print("3. Define Set 2 (S2).")
    print("4. Implement the optimization loop for the second segment (q1* -> S2) to find p2* and q2*.")
    print("5. Combine paths and visualize the full q_init -> q1* -> q2* trajectory.")

if __name__ == "__main__":
    run_planning_scenario()
    print("\nmain.py finished.")

