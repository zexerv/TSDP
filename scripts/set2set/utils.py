# utils.py
import numpy as np
import random # Make sure random is imported
import config
from ik_solver import forward_kinematics, inverse_kinematics, normalize_angles, calculate_jacobian
import traceback

print("Loading utils.py...")

# --- Robot State Initialization ---


def check_joint_limits(q):
    """Checks if a configuration q is within joint limits."""
    return np.all(q >= config.JOINT_LIMITS_MIN) and np.all(q <= config.JOINT_LIMITS_MAX)

def calculate_manipulability_cost(q, epsilon=config.MANIPULABILITY_EPSILON, max_cost=config.MAX_MANIPULABILITY_COST):
    """Calculates inverse manipulability cost. Returns large value on error."""
    try:
        J = calculate_jacobian(q)
        if J is None or J.shape[0] != 6: return max_cost
        JJT = J @ J.T
        det_JJT = np.linalg.det(JJT)
        if det_JJT < 1e-12: w = 0.0 # Treat near-zero as singular
        else: w = math.sqrt(det_JJT)
        cost = 1.0 / (w + epsilon)
        return min(cost, max_cost) # Clamp cost
    except Exception: # Catch LinAlgError or others
        return max_cost

def calculate_joint_limit_cost(q, epsilon=config.IK_SELECT_JLIM_EPSILON):
    """Calculates a cost based on proximity to joint limits."""
    cost = 0.0
    for i in range(config.NUM_JOINTS):
        q_i = q[i]
        q_min = config.JOINT_LIMITS_MIN[i]
        q_max = config.JOINT_LIMITS_MAX[i]
        # Add cost for lower limit proximity
        cost += 1.0 / (abs(q_i - q_min) + epsilon)
        # Add cost for upper limit proximity
        cost += 1.0 / (abs(q_max - q_i) + epsilon)
        # Optional: Normalize by range? For now, raw inverse distance.
    return cost

def calculate_cspace_distance_sq_cost(q1, q2):
    """Calculates squared Euclidean distance between two configurations."""
    # Check for None inputs first
    if q1 is None or q2 is None:
        # Decide how to handle missing reference: return 0, large cost, or error?
        # Returning 0 means the distance component has no effect if hint is missing.
        # print("Warning: Cannot calculate C-space distance - None input.") # Optional debug
        return 0.0

    # --- Convert to NumPy arrays ---
    try:
        q1_np = np.asarray(q1) # Use asarray to avoid copy if already array
        q2_np = np.asarray(q2)
    except Exception as e:
        print(f"Error converting inputs to NumPy arrays in distance calc: {e}")
        return float('inf') # Return high cost on conversion error

    # Check shapes after conversion
    if q1_np.shape != q2_np.shape or q1_np.ndim != 1:
         # print(f"Warning: Cannot calculate C-space distance - shape mismatch {q1_np.shape} vs {q2_np.shape}.") # Debug
         return float('inf') # Return high cost for shape mismatch


    # --- Perform subtraction on NumPy arrays ---
    diff = q1_np - q2_np

    # Optional: Handle angle wrap-around difference? More complex, requires knowing which indices are angles.
    # Sticking to standard Euclidean distance for now:
    return np.dot(diff, diff) # Squared distance

# def calculate_cspace_distance_sq_cost(q1, q2):
#     """Calculates squared Euclidean distance between two configurations."""
#     if q1 is None or q2 is None or len(q1) != len(q2):
#         # print("Warning: Cannot calculate C-space distance - invalid inputs.") # Debug
#         return 0.0 # Or maybe infinity if you want to penalize missing hint? 0 seems safer.
#     diff = q1 - q2
#     # Optional: Handle angle wrap-around difference? Simple Euclidean for now.
#     return np.dot(diff, diff) # Squared distance

def initialize_robot_cspace(initial_joint_angles):
    """
    Initializes and validates the robot state from a C-space configuration.
    Performs Forward Kinematics (FK).

    Args:
        initial_joint_angles (list or np.array): List of 6 joint angles (radians).

    Returns:
        tuple: (q_normalized, T0_TCP, joint_positions) or (None, None, None) on failure.
               q_normalized: Normalized joint angles (numpy array).
               T0_TCP: 4x4 pose matrix of the TCP.
               joint_positions: List of 3D coordinates for base, joints, and TCP.
    """
    # print("-" * 20)
    # print(f"Attempting initialization from C-space...") # Less verbose
    try:
        q_init_c = np.array(initial_joint_angles, dtype=float)
        if q_init_c.shape != (config.NUM_JOINTS,):
            print(f"Error: Invalid C-space input. Expected {config.NUM_JOINTS} angles, got {len(q_init_c)}.")
            return None, None, None
        # print(f"Input C-space (deg): {np.round(np.degrees(q_init_c), 2)}") # Less verbose

        # Normalize angles (important before FK if solver expects specific ranges)
        q_init_c_norm = np.array(normalize_angles(q_init_c))
        if not np.allclose(q_init_c, q_init_c_norm, atol=1e-6):
             print(f"Info: Angles normalized. Initial (deg): {np.round(np.degrees(q_init_c), 1)}, Normalized (deg): {np.round(np.degrees(q_init_c_norm), 1)}")


        # Check against joint limits (optional but good practice)
        if not (np.all(q_init_c_norm >= config.JOINT_LIMITS_MIN) and np.all(q_init_c_norm <= config.JOINT_LIMITS_MAX)):
             print(f"Warning: Initial C-space config (normalized, deg) {np.round(np.degrees(q_init_c_norm), 1)} potentially outside defined limits.")
             # Decide if this should be an error or just a warning
             # return None, None, None # Uncomment to make it an error

        joint_positions, T0_TCP, _ = forward_kinematics(q_init_c_norm)
        # print("FK successful.") # Less verbose
        # print(f"Resulting Task Space Pose (TCP):\n{np.round(T0_TCP, 3)}")
        # print("-" * 20)
        return q_init_c_norm, T0_TCP, joint_positions
    except Exception as e:
        print(f"Error during C-space initialization (FK): {e}")
        # traceback.print_exc() # Uncomment for detailed debug
        # print("-" * 20)
        return None, None, None

def initialize_robot_taskspace(T_target, current_config_hint=None):
    """
    Initializes the robot to a target end-effector pose T_target.
    Calculates IK, finds all solutions, and selects the 'best' one based
    on a combined cost function (manipulability, joint limits, distance).

    Args:
        T_target (np.ndarray): The desired 4x4 end-effector pose matrix.
        current_config_hint (np.ndarray, optional): A nearby C-space configuration
                                                     used as reference for distance cost.

    Returns:
        tuple: (q_best, T_final, joint_positions_best)
               q_best (np.ndarray): The selected best joint configuration (radians).
               T_final (np.ndarray): The actual 4x4 pose achieved by q_best (from FK).
               joint_positions_best (list): List of 3D joint positions for q_best.
               Returns (None, None, None) if no valid IK solution is found.
    """
    print(f"Seeking best IK solution for T_target:\n{np.round(T_target[:3,:], 3)}") # Debug

    # 1. Get all IK solutions
    # Assuming inverse_kinematics returns a list of tuples like [(angles1, type1), (angles2, type2), ...]
    # Or just a list of angle arrays: [angles1, angles2, ...]
    # Adjust parsing based on what your ik_solver returns
    raw_solutions_info = inverse_kinematics(T_target) # Get list of solutions

    if not raw_solutions_info:
        print("  IK Warning: No solutions found.")
        return None, None, None

    best_q_so_far = None
    min_total_cost = float('inf')
    found_valid = False

    # Determine reference configuration for distance cost
    q_ref = current_config_hint # Use the hint if provided

    # 2. Iterate through solutions and evaluate cost
    num_solutions = len(raw_solutions_info)
    # print(f"  IK Found {num_solutions} raw solution(s). Evaluating costs...") # Debug
    evaluated_costs = [] # For debugging

    for i, sol_info in enumerate(raw_solutions_info):
        # --- Adjust parsing based on ik_solver return format ---
        if isinstance(sol_info, tuple):
             q_raw = sol_info[0] # Assuming first element is the angle array
        else:
             q_raw = sol_info # Assuming it directly returns angle arrays
        # --- End parsing adjustment ---

        q_norm = normalize_angles(q_raw) # Normalize angles to consistent range

        # Basic validity check: Joint Limits
        if not check_joint_limits(q_norm):
            # print(f"  Solution {i+1}: Invalid (Joint Limits)") # Debug
            continue

        # Calculate individual cost components
        cost_manip = calculate_manipulability_cost(q_norm)
        cost_jlim = calculate_joint_limit_cost(q_norm)
        cost_dist = calculate_cspace_distance_sq_cost(q_norm, q_ref)

        # Check if costs are valid
        if not (np.isfinite(cost_manip) and np.isfinite(cost_jlim) and np.isfinite(cost_dist)):
             # print(f"  Solution {i+1}: Invalid cost components (Manip={cost_manip:.2e}, JLim={cost_jlim:.2e}, Dist={cost_dist:.2e})") # Debug
             continue

        # Calculate total weighted cost
        total_cost = (config.IK_SELECT_WEIGHT_MANIP_COST * cost_manip +
                      config.IK_SELECT_WEIGHT_JLIM_COST * cost_jlim +
                      config.IK_SELECT_WEIGHT_DIST_COST * cost_dist)

        evaluated_costs.append(total_cost) # Store for debugging

        # Check if this solution is the best so far
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_q_so_far = q_norm
            found_valid = True
            # print(f"  Solution {i+1}: New best found! Total Cost = {total_cost:.4f} (M:{cost_manip:.2f}, J:{cost_jlim:.2f}, D:{cost_dist:.2f})") # Debug

    # print(f"  Evaluated Costs: {np.round(evaluated_costs, 3)}") # Debug summary

    # 3. Return the best solution found
    if found_valid and best_q_so_far is not None:
        print(f"  Selected best IK solution with cost {min_total_cost:.4f}")
        # Perform FK on the selected best configuration
        try:
            joint_positions_best, T_final_check, _ = forward_kinematics(best_q_so_far)
            # Optional: Check if T_final_check is close to T_target as a sanity check
            # pose_diff = np.linalg.norm(T_final_check[:3,3] - T_target[:3,3])
            # if pose_diff > 1e-3: print(f"Warning: FK of selected IK solution differs from target by {pose_diff:.4f}m")

            return best_q_so_far, T_final_check, joint_positions_best
        except Exception as e:
             print(f"Error during final FK for best IK solution: {e}")
             return None, None, None # Failed FK
    else:
        print("  IK Warning: No valid solution found after evaluating costs.")
#         return None, None, None
# def initialize_robot_taskspace(target_pose_matrix, current_config_hint=None):
#     """
#     Finds a C-space configuration for a target Task Space pose using Inverse Kinematics (IK).

#     Args:
#         target_pose_matrix (np.ndarray): Target 4x4 pose matrix.
#         current_config_hint (np.array, optional): Current joint config (radians)
#                                                    to help select the 'closest' IK solution. Defaults to None.

#     Returns:
#         tuple: (q_solution, T_target, joint_positions_sol) or (None, None, None) on failure.
#                q_solution: Selected joint angles for the target pose (numpy array).
#                T_target: The input target_pose_matrix.
#                joint_positions_sol: List of 3D coordinates corresponding to q_solution.
#     """
#     # print("-" * 20)
#     # print(f"Attempting initialization from Task Space...") # Less verbose
#     try:
#         if not isinstance(target_pose_matrix, np.ndarray) or target_pose_matrix.shape != (4, 4):
#             print("Error: Invalid Task Space input. Expected a 4x4 NumPy array.")
#             return None, None, None

#         # print(f"Target Task Space Pose (TCP):\n{np.round(target_pose_matrix, 3)}") # Less verbose
#         ik_solutions_data = inverse_kinematics(target_pose_matrix) # Data might include angles + flags

#         if not ik_solutions_data:
#             # print("IK Warning: No solutions found for the target pose.") # Can be verbose in loops
#             return None, None, None

#         # print(f"IK successful. Found {len(ik_solutions_data)} potential solution(s).") # Less verbose

#         # --- Solution Selection ---
#         valid_solutions = []
#         for sol_data in ik_solutions_data:
#             q_sol = np.array(sol_data[0], dtype=float) # Assuming angles are the first element
#             q_sol_norm = normalize_angles(q_sol)

#             # Basic check: ensure solution is within joint limits
#             if np.all(q_sol_norm >= config.JOINT_LIMITS_MIN) and np.all(q_sol_norm <= config.JOINT_LIMITS_MAX):
#                  valid_solutions.append(q_sol_norm)
#             # else:
#             #    print(f"Debug: IK Solution {np.degrees(q_sol_norm)} rejected (out of limits).")

#         if not valid_solutions:
#              # print("IK Warning: All solutions were outside joint limits.") # Can be verbose
#              return None, None, None

#         # TODO: Implement more sophisticated "fittest" selection logic here.
#         #       Examples: closest to hint, highest manipulability, etc.
#         if current_config_hint is not None:
#             # Select solution closest to the hint configuration
#              best_sol = min(valid_solutions, key=lambda sol: np.linalg.norm(sol - current_config_hint))
#              # print(f"Selected IK solution closest to hint (diff: {np.linalg.norm(best_sol - current_config_hint):.2f})") # Verbose
#         else:
#             # If no hint, just pick the first valid one
#             best_sol = valid_solutions[0]
#             # print("Selected first valid IK solution (no hint provided).") # Verbose

#         selected_solution_angles = np.array(best_sol)
#         # print(f"Selected C-space solution (deg): {np.round(np.degrees(selected_solution_angles), 2)}") # Less verbose

#         # --- Verification ---
#         joint_pos_check, T0_TCP_check, _ = forward_kinematics(selected_solution_angles)
#         if not np.allclose(T0_TCP_check, target_pose_matrix, atol=1e-4): # Tolerance might need adjustment
#              print("\nVerification Warning: FK of the selected IK solution deviates significantly from the target pose.")
#              # print(" T_target:\n", np.round(target_pose_matrix, 4))
#              # print(" T_check:\n", np.round(T0_TCP_check, 4))
#              # print(" Difference:\n", np.round(target_pose_matrix - T0_TCP_check, 4))
#              # Decide if this is fatal
#              # return None, None, None
#         # else:
#              # print("Verification: FK of selected IK solution matches target pose.") # Less verbose

#         # print("-" * 20)
#         return selected_solution_angles, target_pose_matrix, joint_pos_check
#     except Exception as e:
#         print(f"Error during Task Space initialization (IK or subsequent FK): {e}")
#         traceback.print_exc()
#         # print("-" * 20)
#         return None, None, None


# --- Set Definition ---

def generate_perturbed_config(base_q, min_deg, max_deg):
    """Generates a perturbed C-space configuration by adding random noise."""
    perturbation_rad = np.radians(np.random.uniform(min_deg, max_deg, size=base_q.shape))
    perturbed_q = base_q + perturbation_rad
    return np.array(normalize_angles(perturbed_q))

def define_task_space_set(center_q, pos_interval_xyz, rot_interval_r):
    """
    Defines a task space set (hypercube for position, simple range for rotation)
    based on a center C-space configuration.

    Args:
        center_q (np.array): Center configuration in C-space (radians).
        pos_interval_xyz (list/tuple): Size of interval [dx, dy, dz] around center position (meters).
        rot_interval_r (float): Simple tolerance for rotation matrix elements (unitless).
                                 Note: This is a *very* simplified representation of orientation bounds.

    Returns:
        dict: Dictionary containing set data ('q_center', 'T_center', 'pos_bounds', 'rot_bounds')
              or None on FK failure.
    """
    try:
        q_center_norm, T_center, _ = initialize_robot_cspace(center_q) # Use init to ensure validity check
        if T_center is None:
            print(f"Error: FK failed for the provided center configuration: {np.degrees(center_q)}")
            return None

        center_pos = T_center[:3, 3]
        center_rot_flat = T_center[:3, :3].flatten() # Flatten 3x3 rotation matrix

        # Position bounds [ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
        pos_delta = np.array(pos_interval_xyz) / 2.0 # Interval is total width
        pos_bounds = np.array([center_pos - pos_delta, center_pos + pos_delta]).T # Shape (3, 2)
        # Ensure min is first, max is second for each dimension
        pos_bounds = np.sort(pos_bounds, axis=1)

        # Rotation bounds (simplified: bounds on each element of flattened matrix)
        # WARNING: This doesn't guarantee valid rotation matrices within the bounds.
        rot_delta = rot_interval_r
        rot_bounds = np.array([center_rot_flat - rot_delta, center_rot_flat + rot_delta]).T # Shape (9, 2)
        rot_bounds = np.sort(rot_bounds, axis=1)

        set_data = {
            'q_center': q_center_norm,      # Store the potentially normalized center q
            'T_center': T_center,
            'pos_bounds': pos_bounds.tolist(),    # List of [min, max] for x, y, z
            'rot_bounds': rot_bounds.tolist()     # List of [min, max] for R11, R12, ..., R33
        }
        print(f"Defined Task Space Set around q_center (deg): {np.round(np.degrees(q_center_norm),1)}")
        return set_data
    except Exception as e:
        print(f"Error defining task space set: {e}")
        return None


def sample_pose_from_set(set_data):
    """
    Samples a target pose within the defined task space set bounds.
    Currently samples position uniformly within the bounds and uses the
    center pose's orientation.

    Args:
        set_data (dict): Dictionary containing set information
                         (must have 'pos_bounds' and 'T_center').

    Returns:
        np.ndarray: A 4x4 sampled target pose matrix, or None if input is invalid.
    """
    if set_data is None or 'pos_bounds' not in set_data or 'T_center' not in set_data:
        print("Error: Invalid set_data provided for sampling.")
        return None

    pos_bounds = np.array(set_data['pos_bounds']) # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    T_center = set_data['T_center']

    try:
        # Sample position uniformly within the bounds
        sampled_x = random.uniform(pos_bounds[0, 0], pos_bounds[0, 1])
        sampled_y = random.uniform(pos_bounds[1, 0], pos_bounds[1, 1])
        sampled_z = random.uniform(pos_bounds[2, 0], pos_bounds[2, 1])
        sampled_pos = np.array([sampled_x, sampled_y, sampled_z])

        # Create the new pose matrix
        T_sampled = np.copy(T_center)
        T_sampled[:3, 3] = sampled_pos # Update position part

        # TODO: Implement orientation sampling within bounds if needed.
        # For now, we use the orientation from T_center.
        # A more robust method would sample axis-angle or quaternion variations.

        # print(f"Sampled Pos: {np.round(sampled_pos, 3)}") # Debug print if needed
        return T_sampled

    except Exception as e:
        print(f"Error during pose sampling: {e}")
        return None


print("utils.py loaded successfully (with sample_pose_from_set).")
