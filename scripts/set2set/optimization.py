# optimization.py
import numpy as np
import time
import config # Make sure config is imported
import utils
# --- NEW: Add scipy for rotation conversions ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNING: scipy not found. Install it with 'pip install scipy'. Orientation optimization will be disabled.")
    SCIPY_AVAILABLE = False
    # Define a dummy R class if scipy is not available to avoid errors later if ENABLE_GD_REFINEMENT is True
    class R:
        @staticmethod
        def from_matrix(matrix): raise ImportError("scipy not available")
        @staticmethod
        def from_quat(quat): raise ImportError("scipy not available")
        def as_quat(self): raise ImportError("scipy not available")
        def as_matrix(self): raise ImportError("scipy not available")
# --- END NEW ---


# --- Configuration loaded from config.py ---
# Ensure these are correctly imported or defined based on your config.py structure
ENABLE_GD_REFINEMENT = config.ENABLE_GD_REFINEMENT and SCIPY_AVAILABLE
GD_ITERATIONS = config.GD_ITERATIONS
# GD_STEP_SIZE is no longer used, replaced by arrays below
GD_TOLERANCE = config.GD_TOLERANCE
GD_EPSILON = config.GD_EPSILON
GD_EVAL_PLANNING_TIME_LIMIT = config.GD_EVAL_PLANNING_TIME_LIMIT

# --- Import new config vars ---
GD_STEP_SIZES_POS = config.GD_STEP_SIZES_POS
GD_STEP_SIZES_ROT = config.GD_STEP_SIZES_ROT
GD_ENABLE_CLIPPING = config.GD_ENABLE_CLIPPING
GD_MAX_GRAD_NORM = config.GD_MAX_GRAD_NORM


# --- Helper Functions (ensure these exist as you had them) ---

def evaluate_target_pose(T_target, q_start, planner, hint_q=None, eval_time_limit=None):
    """
    Helper function internal to optimization: performs IK and Planning.
    (Assumes this function exists and works as before)
    """
    q_goal, _, _ = utils.initialize_robot_taskspace(T_target, current_config_hint=hint_q)
    if q_goal is None:
        # print("Debug: IK failed in evaluate_target_pose") # Optional debug
        return None, float('inf')
    # Use the specific eval time limit for GD steps
    eval_limit = eval_time_limit if eval_time_limit is not None else config.GD_EVAL_PLANNING_TIME_LIMIT
    path_np, path_cost = planner.plan(q_start, q_goal, time_limit=eval_limit) # Pass correct time limit
    if path_np is None:
        # print("Debug: Planning failed in evaluate_target_pose") # Optional debug
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
    Refines the target pose (position and orientation) using Gradient Descent
    with dimension-specific step sizes. Uses quaternions for orientation.

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
    print("--- Starting Gradient Descent Refinement (Pos + Quat, Dim-Specific Steps) ---")
    print(f"Cost from Sampling: {initial_cost_from_sampling:.5f}")
    print(f"Max Iterations: {GD_ITERATIONS}, Pos Steps: {np.round(GD_STEP_SIZES_POS, 4)}, Rot Steps: {np.round(GD_STEP_SIZES_ROT, 4)}, Epsilon: {GD_EPSILON}")
    print("-" * 40)

    # Initialize return history lists
    pose_history_xyz = []
    quat_history_wxyz = [] # Store quaternion history [w, x, y, z]
    cost_history = []
    gradient_history_7d = [] # Store 7D gradients [gx, gy, gz, gw, gqx, gqy, gqz]

    if not ENABLE_GD_REFINEMENT:
        print("GD refinement disabled (or scipy unavailable). Returning initial values.")
        pose_history_xyz.append(initial_T[:3, 3])
        try:
             initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Convert to w,x,y,z
        except Exception: # Handle case where R is dummy class
             initial_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Default
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_from_sampling) # Report the sampling cost if GD disabled
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_from_sampling, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

    # --- Verification Step: Evaluate initial pose with GD's short time limit ---
    initial_q_goal_for_gd, _, _ = utils.initialize_robot_taskspace(initial_T)
    ik_hint_for_gd = set_data.get('q_center') if initial_q_goal_for_gd is None else initial_q_goal_for_gd
    if initial_q_goal_for_gd is None: print("Warning: IK failed for initial GD pose, using set center as hint.")

    print("Verifying initial cost using GD's evaluation method...")
    _, initial_cost_re_evaluated = evaluate_target_pose(initial_T, q_start, planner,
                                                       hint_q=ik_hint_for_gd,
                                                       eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT) # Use specific limit
    print(f"Initial Cost (Re-evaluated with GD eval time limit): {initial_cost_re_evaluated:.5f}")

    if not np.isfinite(initial_cost_re_evaluated):
        print("GD refinement skipped: Initial pose cost is non-finite with GD settings.")
        pose_history_xyz.append(initial_T[:3, 3])
        initial_quat = R.from_matrix(initial_T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        quat_history_wxyz.append(initial_quat)
        cost_history.append(initial_cost_re_evaluated) # Store inf cost
        gradient_history_7d.append(np.zeros(7))
        return initial_T, initial_cost_re_evaluated, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d
    # --- End Verification Step ---

    # Initialize GD state
    current_T = np.copy(initial_T)
    current_cost = initial_cost_re_evaluated # Start GD from the re-evaluated cost
    current_pos = current_T[:3, 3]
    try:
        current_quat_wxyz = R.from_matrix(current_T[:3, :3]).as_quat()[[3, 0, 1, 2]] # Scipy quat is [x,y,z,w] -> convert
    except Exception as e:
         print(f"ERROR converting initial rotation to quaternion: {e}. Cannot start GD.")
         # Return initial state but indicate failure maybe? Or return the infinite cost.
         # Let's return the infinite cost to signal the problem clearly.
         pose_history_xyz.append(initial_T[:3, 3])
         quat_history_wxyz.append(np.array([1.0, 0.0, 0.0, 0.0]))
         cost_history.append(initial_cost_re_evaluated)
         gradient_history_7d.append(np.zeros(7))
         return initial_T, initial_cost_re_evaluated, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d


    ik_hint = ik_hint_for_gd # Use the hint determined earlier

    # Initialize history with the starting state
    pose_history_xyz.append(np.copy(current_pos))
    quat_history_wxyz.append(np.copy(current_quat_wxyz))
    cost_history.append(current_cost)
    gradient_history_7d.append(np.zeros(7)) # Placeholder for gradient *leading to* this state (zero)

    # Get bounds from set_data
    pos_bounds = np.array(set_data['pos_bounds']) # Assumes shape (3, 2) -> [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    # Track best state found *during GD*
    best_gd_cost = current_cost
    best_gd_T = np.copy(current_T)

    # --- GD Loop ---
    for iteration in range(GD_ITERATIONS):
        print(f"\nGD Iteration {iteration + 1}/{GD_ITERATIONS}")
        print(f"  Current Pos: {np.round(current_pos, 5)}, Quat (wxyz): {np.round(current_quat_wxyz, 4)}, Cost: {current_cost:.5f}")

        gradient_7d = np.zeros(7) # [gx, gy, gz, gw, gqx, gqy, gqz]
        valid_gradient = True # Flag to track if all gradient components are valid

        # print("     Calculating Gradient (7D):") # Less verbose
        # --- Gradient for Position (x, y, z) ---
        try:
             current_R_matrix = R.from_quat(current_quat_wxyz[[1, 2, 3, 0]]).as_matrix() # Convert back for T_plus
        except Exception as e:
             print(f"  ERROR: Could not convert current quaternion to matrix: {e}. Stopping GD.")
             gradient_history_7d.pop() # Remove last placeholder
             break

        for i in range(3): # x, y, z
            pos_plus = np.copy(current_pos); pos_plus[i] += GD_EPSILON
            T_plus = np.identity(4); T_plus[:3,:3] = current_R_matrix; T_plus[:3, 3] = pos_plus
            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)

            if not np.isfinite(cost_plus):
                print(f"      Pos Dim {i}: Warning - Planning failed for positive perturbation. Estimating high gradient.")
                # Heuristic: push back towards center of bounds
                bound_center_i = np.mean(pos_bounds[i])
                grad_component = np.sign(current_pos[i] - bound_center_i) * 1e6 # Large gradient away from perturbation
                # Or maybe try negative perturbation? For simplicity, just use high gradient.
            else:
                if GD_EPSILON == 0: grad_component = 0.0
                else: grad_component = (cost_plus - current_cost) / GD_EPSILON # Central difference might be better but needs 2 evals

            if not np.isfinite(grad_component):
                print(f"      Pos Dim {i}: Warning - Non-finite gradient component. Using zero.")
                gradient_7d[i] = 0.0
                valid_gradient = False # Mark gradient as potentially unreliable
            else: gradient_7d[i] = grad_component

        # --- Gradient for Orientation (w, x, y, z quaternion components) ---
        for i in range(4): # w, x, y, z (indices 0, 1, 2, 3 in current_quat_wxyz)
            quat_plus_unnormalized = np.copy(current_quat_wxyz)
            quat_plus_unnormalized[i] += GD_EPSILON
            quat_plus = normalize_quat(quat_plus_unnormalized) # Normalize perturbed quat

            try:
                R_plus = R.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix() # Scipy needs [x,y,z,w]
            except ValueError as e:
                print(f"      Quat Dim {i}: Warning - Invalid quaternion after perturbation? {e}. Using zero gradient.")
                gradient_7d[3+i] = 0.0
                valid_gradient = False
                continue # Skip evaluation for this component

            T_plus = np.identity(4); T_plus[:3,:3] = R_plus; T_plus[:3, 3] = current_pos # Use current pos
            _, cost_plus = evaluate_target_pose(T_plus, q_start, planner, hint_q=ik_hint,
                                                eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT)

            if not np.isfinite(cost_plus):
                print(f"      Quat Dim {i}: Warning - Planning failed for positive perturbation. Estimating high gradient.")
                # Heuristic: push towards identity? Less obvious than position. Use large fixed gradient.
                grad_component = 1e6 # Arbitrary large gradient penalty for invalid orientation region
            else:
                if GD_EPSILON == 0: grad_component = 0.0
                else: grad_component = (cost_plus - current_cost) / GD_EPSILON

            if not np.isfinite(grad_component):
                print(f"      Quat Dim {i}: Warning - Non-finite gradient component. Using zero.")
                gradient_7d[3+i] = 0.0
                valid_gradient = False
            else: gradient_7d[3+i] = grad_component


        # --- Store and Check Gradient ---
        # Store the gradient that *led* to the *next* state (calculated in this iteration)
        # The placeholder added earlier corresponds to the gradient leading to the *current* state.
        # So, update the *last* element of gradient_history_7d
        gradient_history_7d[-1] = np.copy(gradient_7d)

        gradient_norm = np.linalg.norm(gradient_7d)
        # Check validity *before* clipping
        if not valid_gradient or np.any(np.isnan(gradient_7d)) or gradient_norm < 1e-9:
            print(f"  Gradient calculation failed, invalid, or resulted in near-zero vector (Norm: {gradient_norm:.4e}). Stopping GD.")
            # No update occurs, loop breaks. History is already correct up to the previous state.
            break

        print(f"  Raw Gradient (7D): {np.round(gradient_7d, 4)}, Norm: {gradient_norm:.4f}")

        # --- Apply Gradient Clipping (Optional but recommended) ---
        clipped_gradient_7d = np.copy(gradient_7d) # Start with raw gradient
        if GD_ENABLE_CLIPPING and gradient_norm > GD_MAX_GRAD_NORM:
            scale = GD_MAX_GRAD_NORM / gradient_norm
            clipped_gradient_7d = gradient_7d * scale
            new_norm = np.linalg.norm(clipped_gradient_7d)
            print(f"  Gradient clipped (Scale: {scale:.4f}). New Norm: {new_norm:.4f}")
        # --- End Clipping ---

        # ***** MODIFIED UPDATE STEP using dimension-specific step sizes *****
        # Apply dimension-specific step sizes to the (potentially clipped) raw gradient

        # Position Update:
        grad_pos_effective = clipped_gradient_7d[:3] # Use clipped gradient
        delta_pos = -GD_STEP_SIZES_POS * grad_pos_effective # Element-wise multiplication
        next_pos = current_pos + delta_pos
        # Apply clamping
        next_pos_clamped = np.clip(next_pos, pos_bounds[:, 0], pos_bounds[:, 1])
        if not np.allclose(next_pos, next_pos_clamped):
            # print(f"  Position update clamped.") # Less verbose
            pass
        next_pos = next_pos_clamped
        # print(f"  Delta Pos: {np.round(delta_pos, 6)}") # Debug

        # Quaternion Update:
        grad_quat_effective = clipped_gradient_7d[3:] # Use clipped gradient
        delta_quat = -GD_STEP_SIZES_ROT * grad_quat_effective # Element-wise multiplication
        next_quat_unnormalized = current_quat_wxyz + delta_quat
        # Normalize the updated quaternion
        next_quat_wxyz = normalize_quat(next_quat_unnormalized)
        # print(f"  Delta Quat (Unnormalized): {np.round(delta_quat, 6)}") # Debug
        # ***** END OF MODIFIED UPDATE STEP *****


        # --- Evaluate the new combined pose ---
        try:
            next_R_matrix = R.from_quat(next_quat_wxyz[[1, 2, 3, 0]]).as_matrix() # Convert back to R
        except ValueError as e:
            print(f"  Warning: Invalid quaternion after update? {e}. Stopping GD.")
            # Don't add this failed state to history
            break # Stop if quaternion update fails

        next_T = np.identity(4); next_T[:3,:3] = next_R_matrix; next_T[:3, 3] = next_pos
        _, next_cost = evaluate_target_pose(next_T, q_start, planner, hint_q=ik_hint,
                                            eval_time_limit=GD_EVAL_PLANNING_TIME_LIMIT) # Use GD eval time
        print(f"  Evaluated Cost at Next Pose: {next_cost:.5f}")

        # --- Check for improvement and update state ---
        cost_improvement = current_cost - next_cost # Positive if cost decreased

        # Only update if the cost is finite AND has decreased
        if np.isfinite(next_cost) and next_cost < current_cost :
            print(f"  Cost improved by: {cost_improvement:.5f}")
            # Update current state for the next iteration
            current_pos = next_pos
            current_quat_wxyz = next_quat_wxyz
            current_T = next_T
            current_cost = next_cost

            # Update the best-found state if this one is better
            if current_cost < best_gd_cost:
                 best_gd_cost = current_cost
                 best_gd_T = np.copy(current_T)
                 # print("    Updated best GD state.") # Optional debug

            # Update IK hint based on the new pose
            current_q_goal, _, _ = utils.initialize_robot_taskspace(current_T)
            if current_q_goal is not None:
                 ik_hint = current_q_goal # Use the new config as hint if IK succeeded

            # Add the *new* state to history
            pose_history_xyz.append(np.copy(current_pos))
            quat_history_wxyz.append(np.copy(current_quat_wxyz))
            cost_history.append(current_cost)
            # Add placeholder for the gradient calculated in the *next* iteration
            gradient_history_7d.append(np.zeros(7))

            # Check for convergence based on small improvement
            if cost_improvement < GD_TOLERANCE:
                print(f"  Convergence detected (Improvement {cost_improvement:.5e} < Tolerance {GD_TOLERANCE}). Stopping GD.")
                gradient_history_7d.pop() # Remove last placeholder as loop is ending
                break

        else: # Cost did not improve or became non-finite
            if not np.isfinite(next_cost):
                print("  Cost became non-finite. Stopping GD.")
            else:
                print(f"  Cost did not improve (Improvement: {cost_improvement:.5f}). Stopping GD.")
            # Don't add the non-improving state to history.
            # The last placeholder in gradient_history_7d corresponds to the last *successful* step.
            # Remove the placeholder added before the failed step evaluation.
            gradient_history_7d.pop()
            break # Stop GD

    # End of GD loop (either break or max iterations)
    else: # Loop finished naturally (max iterations)
         # Remove the final placeholder added in the last iteration
         if gradient_history_7d: gradient_history_7d.pop()
         print(f"\nGD reached maximum iterations ({GD_ITERATIONS}).")


    # --- Final Results ---
    print("\n" + "-"*40)
    print("--- Gradient Descent Finished ---")
    if len(cost_history) > 1: # Check if GD actually ran
        try:
            final_best_quat = R.from_matrix(best_gd_T[:3,:3]).as_quat()[[3,0,1,2]]
            print(f"Best GD Pos: {np.round(best_gd_T[:3,3], 5)}, Quat (wxyz): {np.round(final_best_quat, 4)}")
        except Exception: # Handle case R is dummy
            print(f"Best GD Pos: {np.round(best_gd_T[:3,3], 5)}")
        print(f"Best GD Cost: {best_gd_cost:.5f}")
        print(f"Total Improvement vs Initial (Re-evaluated): {initial_cost_re_evaluated - best_gd_cost:.5f}")
    else: # GD didn't run or failed immediately
        print("GD did not run or failed on the first step.")
        print(f"Initial Re-evaluated Cost: {initial_cost_re_evaluated:.5f}")
    print(f"Total Iterations performed: {len(cost_history) - 1}") # -1 because history includes initial state
    print("-" * 40)

    # Return the best T and cost found *during GD*, and the histories
    return best_gd_T, best_gd_cost, pose_history_xyz, quat_history_wxyz, cost_history, gradient_history_7d

# --- End of refine_target_pose_with_gd function ---

# Optional: Add final print statement for the module
print("optimization.py loaded (7D GD with Quaternions, Dim-Specific Steps).")