# optimization.py
import numpy as np
import time
import config
import utils
# --- NEW: Add scipy for rotation conversions ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not found. Install it with 'pip install scipy'. Orientation optimization will be disabled.")
    SCIPY_AVAILABLE = False
# --- END NEW ---


# --- Configuration loaded from config.py ---
ENABLE_GD_REFINEMENT = config.ENABLE_GD_REFINEMENT and SCIPY_AVAILABLE # Disable if scipy missing
GD_ITERATIONS = config.GD_ITERATIONS
GD_STEP_SIZE = config.GD_STEP_SIZE # Use same step size for pos and quat for now
GD_TOLERANCE = config.GD_TOLERANCE
GD_EPSILON = config.GD_EPSILON # Use same epsilon for pos and quat for now
GD_EVAL_PLANNING_TIME_LIMIT = config.GD_EVAL_PLANNING_TIME_LIMIT


# --- Helper Function ---

def evaluate_target_pose(T_target, q_start, planner, hint_q=None, eval_time_limit=None):
    """
    Helper function internal to optimization: performs IK and Planning.
    (No changes needed here from previous version)
    """
    q_goal, _, _ = utils.initialize_robot_taskspace(T_target, current_config_hint=hint_q)
    if q_goal is None:
        return None, float('inf')
    path_np, path_cost = planner.plan(q_start, q_goal, time_limit=eval_time_limit)
    if path_np is None:
        return None, float('inf')
    return path_np, path_cost

# --- Quaternion Helper ---
def normalize_quat(q):
    """Normalizes a quaternion [w, x, y, z]."""
    norm = np.linalg.norm(q)
    if norm < 1e-9: # Avoid division by zero
        return np.array([1.0, 0.0, 0.0, 0.0]) # Return identity quaternion
    return q / norm

# --- Gradient Descent Function (7D: Position + Quaternion Orientation) ---

def refine_target_pose_with_gd(initial_T, initial_cost_from_sampling, q_start, planner, set_data):
    """
    Refines the target pose (position and orientation) using Gradient Descent.
    Uses quaternions for orientation.

    Args:
        initial_T (np.ndarray): Starting 4x4 target pose from sampling.
        initial_cost_from_sampling (float): Path cost associated with initial_T *as found during sampling*.
        q_start (np.ndarray): The starting C-space configuration for planning.
        planner (OMPLPlanner): The planner instance.
        set_data (dict): Data defining the target set (for bounds).

    Returns:
        tuple: (best_T_found, best_cost_found, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d)
               Returns initial values if GD fails or finds no improvement.
    """
    print("\n" + "-"*40)
    print("--- Starting Gradient Descent Refinement (Position + Orientation) ---")
    print(f"Cost from Sampling: {initial_cost_from_sampling:.5f}")
    print(f"Max Iterations: {GD_ITERATIONS}, Step Size: {GD_STEP_SIZE}, Epsilon: {GD_EPSILON}")
    print("-" * 40)

    # Initialize return history lists
    pose_history_xyz = []
    quat_history_wxyz = [] # Store quaternion history [w, x, y, z]
    cost_history = []
    gradient_history_7d = [] # Store 7D gradients [gx, gy, gz, gw, gqx, gqy, gqz]

    if not ENABLE_GD_REFINEMENT:
        print("GD refinement disabled (or scipy unavailable).")
        pose_history_xyz.append(initial_T[:3, 3])
        if SCIPY_AVAILABLE:
             initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Convert to w,x,y,z
        else:
             initial_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Default
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_from_sampling)
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_from_sampling, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

    # --- Verification Step ---
    initial_q_goal_for_gd, _, _ = utils.initialize_robot_taskspace(initial_T)
    ik_hint_for_gd = set_data.get('q_center') if initial_q_goal_for_gd is None else initial_q_goal_for_gd
    if initial_q_goal_for_gd is None: print("Warning: IK failed for initial GD pose, using set center as hint.")

    print("Verifying initial cost using GD's evaluation method...")
    _, initial_cost_re_evaluated = evaluate_target_pose(initial_T, q_start, planner,
                                                        hint_q=ik_hint_for_gd,
                                                        eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)
    print(f"Initial Cost (Re-evaluated with GD settings): {initial_cost_re_evaluated:.5f}")
    if not np.isfinite(initial_cost_re_evaluated):
        print("GD refinement skipped: Initial pose cost is non-finite with GD settings.")
        pose_history_xyz.append(initial_T[:3, 3])
        initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_re_evaluated)
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_re_evaluated, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d
    # --- End Verification Step ---

    current_T = np.copy(initial_T)
    current_cost = initial_cost_re_evaluated
    current_pos = current_T[:3, 3]
    # Convert initial orientation to quaternion [w, x, y, z]
    current_quat_wxyz = R.from_matrix(current_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Scipy quat is [x,y,z,w] -> convert
    ik_hint = ik_hint_for_gd

    pose_history_xyz.append(np.copy(current_pos))
    quat_history_wxyz.append(np.copy(current_quat_wxyz))
    cost_history.append(current_cost)
    gradient_history_7d.append(np.zeros(7)) # Placeholder

    pos_bounds = np.array(set_data['pos_bounds'])
    best_gd_cost = current_cost
    best_gd_T = np.copy(current_T)

    for iteration in range(GD_ITERATIONS):
        print(f"\nGD Iteration {iteration + 1}/{GD_ITERATIONS}")
        print(f"  Current Pos: {np.round(current_pos, 5)}, Quat (wxyz): {np.round(current_quat_wxyz, 4)}, Cost: {current_cost:.5f}")

        gradient_7d = np.zeros(7) # [gx, gy, gz, gw, gqx, gqy, gqz]
        valid_gradient = True

        print("    Calculating Gradient (7D):")
        # --- Gradient for Position (x, y, z) ---
        current_R = R.from_quat(current_quat_wxyz[[1, 2, 3, 0]]).as_matrix() # Convert back for T_plus
        for i in range(3):
            pos_plus = np.copy(current_pos); pos_plus[i] += GD_EPSILON
            T_plus = np.identity(4); T_plus[:3,:3] = current_R; T_plus[:3, 3] = pos_plus
            # print(f"      Pos Dim {i}: Perturbing pos to: {np.round(pos_plus, 5)}") # Verbose
            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)
            # print(f"      Pos Dim {i}: Evaluated cost_plus = {cost_plus:.5f}") # Verbose

            if not np.isfinite(cost_plus):
                 print(f"      Pos Dim {i}: Warning: Planning failed. Estimating high gradient.")
                 bound_center_i = np.mean(pos_bounds[i])
                 grad_component = np.sign(current_pos[i] - bound_center_i) * 1e6
            else:
                 if GD_EPSILON == 0: grad_component = 0.0
                 else: grad_component = (cost_plus - current_cost) / GD_EPSILON

            if not np.isfinite(grad_component):
                 print(f"      Pos Dim {i}: Warning: Non-finite gradient. Using zero.")
                 gradient_7d[i] = 0; valid_gradient = False
            else: gradient_7d[i] = grad_component

        # --- Gradient for Orientation (w, x, y, z quaternion components) ---
        for i in range(4): # Iterate through w, x, y, z (indices 0, 1, 2, 3)
            quat_plus_unnormalized = np.copy(current_quat_wxyz)
            quat_plus_unnormalized[i] += GD_EPSILON
            # *** Normalize the perturbed quaternion ***
            quat_plus = normalize_quat(quat_plus_unnormalized)

            # Convert normalized quaternion back to rotation matrix
            try:
                 R_plus = R.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix() # Scipy needs [x,y,z,w]
            except ValueError as e:
                 print(f"      Quat Dim {i}: Warning: Invalid quaternion after perturbation? {e}. Skipping.")
                 gradient_7d[3+i] = 0 # Assign zero gradient
                 valid_gradient = False
                 continue # Skip evaluation for this component

            # Create the perturbed pose matrix
            T_plus = np.identity(4); T_plus[:3,:3] = R_plus; T_plus[:3, 3] = current_pos # Use current pos
            # print(f"      Quat Dim {i}: Perturbing quat to (wxyz): {np.round(quat_plus, 4)}") # Verbose

            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)
            # print(f"      Quat Dim {i}: Evaluated cost_plus = {cost_plus:.5f}") # Verbose

            if not np.isfinite(cost_plus):
                 print(f"      Quat Dim {i}: Warning: Planning failed. Estimating high gradient.")
                 # Heuristic: push towards identity quaternion? Less clear than position bounds.
                 # Use a large fixed gradient for now if perturbation fails.
                 grad_component = 1e6 # Arbitrary large gradient
            else:
                 if GD_EPSILON == 0: grad_component = 0.0
                 else: grad_component = (cost_plus - current_cost) / GD_EPSILON

            if not np.isfinite(grad_component):
                 print(f"      Quat Dim {i}: Warning: Non-finite gradient. Using zero.")
                 gradient_7d[3+i] = 0; valid_gradient = False # Index 3 is w, 4 is x, etc.
            else: gradient_7d[3+i] = grad_component


        gradient_history_7d[-1] = np.copy(gradient_7d) # Store calculated 7D gradient

        gradient_norm = np.linalg.norm(gradient_7d)
        if not valid_gradient or np.any(np.isnan(gradient_7d)) or gradient_norm < 1e-9:
            print("  Gradient calculation failed or resulted in near-zero vector. Stopping GD.")
            gradient_history_7d.pop(); break

        print(f"  Raw Gradient (7D): {np.round(gradient_7d, 4)}, Norm: {gradient_norm:.4f}")

        # --- Update Step (Position + Quaternion) ---
        update_direction = -gradient_7d / gradient_norm if gradient_norm > 0 else np.zeros(7)

        # Update Position
        grad_pos = update_direction[:3]
        next_pos = current_pos + GD_STEP_SIZE * grad_pos # Use same step size for now
        next_pos_clamped = np.clip(next_pos, pos_bounds[:, 0], pos_bounds[:, 1])
        if not np.allclose(next_pos, next_pos_clamped):
             print(f"  Proposed Next Pos (Clamped): {np.round(next_pos_clamped, 5)}")
        next_pos = next_pos_clamped

        # Update Quaternion
        grad_quat = update_direction[3:] # w, x, y, z components
        next_quat_unnormalized = current_quat_wxyz + GD_STEP_SIZE * grad_quat # Use same step size
        # *** Normalize the updated quaternion ***
        next_quat_wxyz = normalize_quat(next_quat_unnormalized)
        # print(f"  Proposed Next Quat (wxyz): {np.round(next_quat_wxyz, 4)}") # Verbose

        # --- Evaluate the new combined pose ---
        try:
            next_R = R.from_quat(next_quat_wxyz[[1, 2, 3, 0]]).as_matrix() # Convert back to R
        except ValueError as e:
            print(f"  Warning: Invalid quaternion after update? {e}. Stopping GD.")
            gradient_history_7d.pop() # Remove last gradient placeholder
            break # Stop if quaternion update fails

        next_T = np.identity(4); next_T[:3,:3] = next_R; next_T[:3, 3] = next_pos
        _, next_cost = evaluate_target_pose(next_T, q_start, planner, hint_q=ik_hint,
                                            eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)
        print(f"  Evaluated Cost at Next Pose: {next_cost:.5f}")

        # --- Check for improvement ---
        cost_improvement = current_cost - next_cost
        if np.isfinite(next_cost) and next_cost < current_cost:
            print(f"  Cost improved by: {cost_improvement:.5f}")
            # Update current state
            current_pos = next_pos
            current_quat_wxyz = next_quat_wxyz
            current_T = next_T # Store the full matrix as well
            current_cost = next_cost

            if current_cost < best_gd_cost:
                 best_gd_cost = current_cost
                 best_gd_T = np.copy(current_T)

            # Update IK hint
            current_q_goal, _, _ = utils.initialize_robot_taskspace(current_T)
            if current_q_goal is not None: ik_hint = current_q_goal

            # Add new state to history
            pose_history_xyz.append(np.copy(current_pos))
            quat_history_wxyz.append(np.copy(current_quat_wxyz))
            cost_history.append(current_cost)
            gradient_history_7d.append(np.zeros(7)) # Placeholder for next gradient

            if cost_improvement < GD_TOLERANCE:
                print(f"  Convergence detected (Improvement < {GD_TOLERANCE}). Stopping GD.")
                gradient_history_7d.pop(); break
        else:
            print("  Cost did not improve or became non-finite. Stopping GD.")
            gradient_history_7d.pop(); break
    else: # Loop finished naturally
         if gradient_history_7d: gradient_history_7d.pop()

    print("\n" + "-"*40)
    print("--- Gradient Descent Finished ---")
    print(f"Best GD Pos: {np.round(best_gd_T[:3,3], 5)}, Quat (wxyz): {np.round(R.from_matrix(best_gd_T[:3,:3]).as_quat()[[3,0,1,2]], 4)}")
    print(f"Best GD Cost: {best_gd_cost:.5f}")
    print(f"Total Improvement vs Initial (Re-evaluated): {initial_cost_re_evaluated - best_gd_cost:.5f}")
    print("-" * 40)

    # Return the best T and cost found *during GD*, and the histories
    return best_gd_T, best_gd_cost, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

print("optimization.py loaded (7D GD with Quaternions).")
