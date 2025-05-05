# utils.py
import numpy as np
import random
import config # Reads configuration parameters
# Attempt to import from ik_solver, handle if it fails
try:
    # Ensure all necessary functions are imported
    from ik_solver import forward_kinematics, inverse_kinematics, normalize_angles, calculate_jacobian
    IK_SOLVER_AVAILABLE = True
except ImportError:
    print("WARNING (utils.py): Could not import functions from ik_solver.py. Related features will fail.")
    IK_SOLVER_AVAILABLE = False
    # Define dummy functions if ik_solver is missing to avoid NameErrors later
    def forward_kinematics(*args, **kwargs): print("ERROR: forward_kinematics unavailable."); return None, None, None
    def inverse_kinematics(*args, **kwargs): print("ERROR: inverse_kinematics unavailable."); return []
    def normalize_angles(q): print("ERROR: normalize_angles unavailable."); return q # Return input as fallback
    def calculate_jacobian(*args, **kwargs): print("ERROR: calculate_jacobian unavailable."); return None

import traceback # For printing detailed error messages
import yaml # For loading configuration files
from pathlib import Path # For handling file paths
import math # For mathematical operations like sqrt, acos, sin
from typing import Optional, Tuple, List, Dict, Any # Added more specific types

# --- SciPy/TF Import for Transformations ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
    print("INFO (utils.py): Using scipy.spatial.transform for rotations.")

    def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
        """Converts 3x3 or 4x4 matrix to quaternion [w, x, y, z] using scipy."""
        if not SCIPY_AVAILABLE: raise RuntimeError("Scipy not available.")
        if matrix.shape != (3, 3) and matrix.shape != (4, 4):
             raise ValueError(f"Input matrix must be 3x3 or 4x4, got {matrix.shape}")
        try:
            quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        except Exception as e:
            print(f"ERROR in matrix_to_quat_wxyz: {e}. Returning identity.")
            traceback.print_exc()
            return np.array([1.0, 0.0, 0.0, 0.0])

    def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        """Converts quaternion [w, x, y, z] to 3x3 rotation matrix using scipy."""
        if not SCIPY_AVAILABLE: raise RuntimeError("Scipy not available.")
        quat_wxyz = np.asarray(quat_wxyz)
        if quat_wxyz.shape != (4,): raise ValueError("Input quaternion must have 4 elements.")
        try:
            norm = np.linalg.norm(quat_wxyz)
            if norm < 1e-9: return np.identity(3)
            quat_normalized = quat_wxyz / norm
            quat_xyzw = quat_normalized[[1, 2, 3, 0]]
            return R.from_quat(quat_xyzw).as_matrix()
        except Exception as e:
            print(f"ERROR in quat_wxyz_to_matrix: {e}. Returning identity.")
            traceback.print_exc()
            return np.identity(3)

    def matrix_from_pose_dict(pose_dict: dict) -> Optional[np.ndarray]:
        """Creates 4x4 matrix from pose dict {'position': [x,y,z], 'quaternion': [w,x,y,z]}."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for matrix_from_pose_dict."); return None
        try:
            pos = np.array(pose_dict['position'])
            quat_wxyz = np.array(pose_dict['quaternion']) # Assuming wxyz
            if len(pos) != 3 or len(quat_wxyz) != 4: raise ValueError("Invalid pose dict structure.")
            rot_matrix = quat_wxyz_to_matrix(quat_wxyz)
            matrix = np.identity(4)
            matrix[:3, :3] = rot_matrix
            matrix[:3, 3] = pos
            return matrix
        except (KeyError, ValueError, TypeError) as e:
             print(f"ERROR creating matrix from pose dict: {e}"); return None
        except Exception as e:
             print(f"ERROR creating matrix from pose dict (unexpected): {e}"); traceback.print_exc(); return None

    # --- Orientation Math Utilities using Scipy ---
    def log_map_so3(R_mat: np.ndarray) -> Optional[np.ndarray]:
        """Maps SO(3) -> so(3) represented as R^3 vector (axis * angle)."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for log_map_so3."); return None
        if R_mat.shape != (3, 3): raise ValueError("Input must be a 3x3 rotation matrix.")
        try:
            if np.allclose(R_mat, np.identity(3)): return np.zeros(3)
            # Use scipy's as_rotvec which directly computes axis * angle
            rotvec = R.from_matrix(R_mat).as_rotvec()
            return rotvec
        except Exception as e:
            print(f"ERROR calculating log map: {e}"); return None

    def exp_map_so3(v_log: np.ndarray) -> Optional[np.ndarray]:
        """Maps so(3) represented as R^3 vector -> SO(3) (3x3 rotation matrix)."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for exp_map_so3."); return None
        v_log = np.asarray(v_log)
        if v_log.shape != (3,): raise ValueError("Input must be a 3D axis-angle vector.")
        try:
            if np.allclose(v_log, np.zeros(3)): return np.identity(3)
            R_mat = R.from_rotvec(v_log).as_matrix()
            return R_mat
        except Exception as e:
            print(f"ERROR calculating exp map: {e}"); return None

    def quat_to_log_map(q_wxyz: np.ndarray) -> Optional[np.ndarray]:
        """Converts quaternion [w, x, y, z] to log map vector v_log."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for quat_to_log_map."); return None
        try:
            R_mat = quat_wxyz_to_matrix(q_wxyz) # Handles normalization
            if R_mat is None: return None
            return log_map_so3(R_mat)
        except Exception as e: print(f"ERROR in quat_to_log_map: {e}"); return None

    def log_map_to_quat(v_log: np.ndarray) -> Optional[np.ndarray]:
        """Converts log map vector v_log to quaternion [w, x, y, z]."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for log_map_to_quat."); return None
        try:
            R_mat = exp_map_so3(v_log)
            if R_mat is None: return None
            return matrix_to_quat_wxyz(R_mat)
        except Exception as e: print(f"ERROR in log_map_to_quat: {e}"); return None

except ImportError:
    print("CRITICAL WARNING (utils.py): scipy not found. Install with 'pip install scipy'. Transformation functions will fail.")
    SCIPY_AVAILABLE = False
    # Define dummy functions
    def matrix_to_quat_wxyz(matrix): raise NotImplementedError("Scipy required")
    def quat_wxyz_to_matrix(quat_wxyz): raise NotImplementedError("Scipy required")
    def matrix_from_pose_dict(pose_dict): raise NotImplementedError("Scipy required")
    def log_map_so3(R_mat): raise NotImplementedError("Scipy required")
    def exp_map_so3(v_log): raise NotImplementedError("Scipy required")
    def quat_to_log_map(q_wxyz): raise NotImplementedError("Scipy required")
    def log_map_to_quat(v_log): raise NotImplementedError("Scipy required")


print("Loading utils.py...")


# --- Angular Difference Helper ---
def calculate_angular_difference(R1: np.ndarray, R2: np.ndarray) -> Optional[float]:
    """Calculates angle of rotation from R1 to R2 (radians)."""
    if not SCIPY_AVAILABLE: print("Warning: Scipy recommended for robust angular diff calc.");
    if R1.shape != (3,3) or R2.shape != (3,3): return None
    try:
        R_rel = R1.T @ R2
        trace_R_rel = np.trace(R_rel)
        cos_angle_arg = np.clip((trace_R_rel - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle_arg)
        return angle_rad
    except Exception as e: print(f"Warning: Could not calculate angular difference: {e}"); return None

# --- YAML Loading Helpers ---
def load_yaml_file(filepath: str) -> Optional[Any]:
    """Loads a YAML file safely, returning None on error."""
    path_obj = Path(filepath)
    if not path_obj.is_file(): print(f"ERROR: YAML file not found: {filepath}"); return None
    try:
        with open(path_obj, 'r') as f: data = yaml.safe_load(f)
        if data is None: print(f"Warning: YAML file is empty: {filepath}"); return {} # Return empty dict/list?
        return data
    except yaml.YAMLError as e: print(f"ERROR parsing YAML file {filepath}: {e}"); return None
    except Exception as e: print(f"ERROR reading file {filepath}: {e}"); return None

def load_interface_transforms(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """ Loads interface transforms (H_D_I) from YAML."""
    config_data = load_yaml_file(filepath)
    if config_data is None or not isinstance(config_data, dict):
        print(f"ERROR: Invalid format or failed to load interface transforms from {filepath}"); return None
    transforms = {}; loaded_count = 0; error_count = 0
    for interface_id_key, data in config_data.items():
        try:
            if isinstance(data, dict) and 'T_aruco_interface' in data:
                H_D_I = np.array(data['T_aruco_interface'], dtype=float)
                if H_D_I.shape == (4, 4): transforms[interface_id_key] = H_D_I; loaded_count += 1
                else: print(f"Warning: Invalid matrix shape for {interface_id_key}"); error_count += 1
        except Exception as e: print(f"ERROR processing interface {interface_id_key}: {e}"); error_count += 1
    print(f"Loaded {loaded_count} interface transforms from {filepath} ({error_count} errors).")
    if loaded_count == 0 and error_count > 0: return None
    return transforms

def load_environment_config(filepath: str) -> Optional[np.ndarray]:
    """ Loads environment config YAML and extracts H_B_D. """
    config_data = load_yaml_file(filepath)
    if config_data is None or not isinstance(config_data, dict):
         print(f"ERROR: Invalid format or failed to load environment config from {filepath}"); return None
    H_B_D = None
    try:
        device_data = config_data.get('aruco_device', config_data.get('aruco_device_pose'))
        if device_data is None: raise KeyError("Could not find 'aruco_device' key.")
        pose_data = device_data.get('pose')
        if pose_data is None: raise KeyError("Missing 'pose' key.")
        H_B_D = matrix_from_pose_dict(pose_data) # Uses scipy
        if H_B_D is None: raise ValueError("Failed to convert pose dict to matrix.")
        print(f"Loaded device pose (H_B_D) from {filepath}.")
        return H_B_D
    except (KeyError, ValueError, TypeError, NotImplementedError) as e:
        print(f"ERROR extracting device pose from {filepath}: {e}"); return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading environment config {filepath}: {e}"); traceback.print_exc(); return None

# --- REMOVED: define_set_from_segment_data ---
# This function was specific to the traditional approach using tube_stats

# --- Helper Function for Probabilistic Approach ---
def extract_target_pose_from_gmm(gmm_params: dict, method: str, H_B_I: np.ndarray) -> Optional[np.ndarray]:
    """
    Extracts a single target 4x4 pose matrix (in Base frame) from GMM parameters
    representing a distribution at a boundary (relative to Interface frame).
    """
    if not SCIPY_AVAILABLE: print("ERROR: Scipy required for extract_target_pose_from_gmm."); return None
    if H_B_I is None or H_B_I.shape != (4,4): print("ERROR: Invalid H_B_I transform provided."); return None

    try:
        weights = np.array(gmm_params['weights'])
        means = np.array(gmm_params['means']) # Shape (K, 6)
        covariances = gmm_params.get('covariances') # Optional
        n_components = len(weights)

        if means.shape != (n_components, 6):
             raise ValueError(f"GMM means shape {means.shape} inconsistent with weights length {n_components} or dim != 6.")
        if covariances is not None:
            covariances = np.array(covariances) # Convert list of lists to numpy array
            if covariances.shape != (n_components, 6, 6):
                 print(f"Warning: GMM covariances shape {covariances.shape} inconsistent. Ignoring for sampling.")
                 covariances = None

        p_I = None; v_log_I = None # Position and Log map vector in Interface frame

        if method == 'mean_of_best_component':
            if n_components == 0: raise ValueError("GMM has no components.")
            best_k = np.argmax(weights); mean_k = means[best_k]
            p_I = mean_k[:3]; v_log_I = mean_k[3:]
            # print(f"  Using mean of best GMM component {best_k} (weight {weights[best_k]:.3f})") # Debug

        elif method == 'sample':
            if n_components == 0: raise ValueError("GMM has no components.")
            k = np.random.choice(n_components, p=weights) # Sample component index
            mean_k = means[k]
            if covariances is not None:
                # print(f"  Sampling from GMM component {k} (weight {weights[k]:.3f})") # Debug
                z_sampled = np.random.multivariate_normal(mean_k, covariances[k])
            else:
                # print(f"  Using mean of randomly selected GMM component {k} (weight {weights[k]:.3f}) (Covariances missing/invalid)") # Debug
                z_sampled = mean_k # Fallback to mean
            p_I = z_sampled[:3]; v_log_I = z_sampled[3:]

        elif method == 'overall_mean':
            if n_components == 0: raise ValueError("GMM has no components.")
            # print(f"  Calculating overall weighted mean (Note: orientation is approximate)") # Debug
            overall_mean = np.sum(weights[:, np.newaxis] * means, axis=0)
            p_I = overall_mean[:3]; v_log_I = overall_mean[3:]

        else: raise ValueError(f"Unknown waypoint extraction method: {method}")

        # Convert log map vector (relative to I) to rotation matrix R_I
        R_I = exp_map_so3(v_log_I)
        if R_I is None: raise ValueError("Failed to convert log map vector to rotation matrix.")

        # Construct pose matrix in Interface frame
        T_I_target = np.identity(4)
        T_I_target[:3, :3] = R_I
        T_I_target[:3, 3] = p_I

        # Transform to Base frame
        T_B_target = H_B_I @ T_I_target
        return T_B_target

    except KeyError as e: print(f"ERROR: Missing key {e} in gmm_params."); return None
    except ValueError as e: print(f"ERROR processing GMM data: {e}"); return None
    except Exception as e: print(f"ERROR during target pose extraction: {e}"); traceback.print_exc(); return None


# --- Robot State Initialization and IK Selection ---
def check_joint_limits(q: np.ndarray) -> bool:
    """Checks if a configuration q is within joint limits defined in config."""
    return np.all(q >= config.JOINT_LIMITS_MIN) and np.all(q <= config.JOINT_LIMITS_MAX)

def calculate_manipulability_cost(q: np.ndarray, epsilon=config.MANIPULABILITY_EPSILON, max_cost=config.MAX_MANIPULABILITY_COST) -> float:
    """ Calculates inverse manipulability cost = 1 / (sqrt(det(JJ^T)) + eps). """
    if not IK_SOLVER_AVAILABLE: return max_cost
    try: J = calculate_jacobian(q)
    except NameError: return max_cost
    if J is None or J.shape[0] != 6 or J.shape[1] != config.NUM_JOINTS: return max_cost
    try: JJT = J @ J.T; det_JJT = np.linalg.det(JJT)
    except np.linalg.LinAlgError: return max_cost
    if det_JJT < epsilon**2: w = 0.0
    else: w = math.sqrt(det_JJT)
    cost = 1.0 / (w + epsilon); return min(cost, max_cost)
    # except Exception as e: print(f"Warning: Unexpected error during manipulability cost calc: {e}"); return max_cost

def calculate_joint_limit_cost(q: np.ndarray, epsilon=config.IK_SELECT_JLIM_EPSILON) -> float:
    """ Calculates a cost based on proximity to joint limits. """
    cost = 0.0; q_np = np.asarray(q); q_min = config.JOINT_LIMITS_MIN; q_max = config.JOINT_LIMITS_MAX
    dist_to_min = np.abs(q_np - q_min); dist_to_max = np.abs(q_max - q_np)
    cost = np.sum(1.0 / (dist_to_min + epsilon)) + np.sum(1.0 / (dist_to_max + epsilon)); return cost

def calculate_cspace_distance_sq_cost(q1: Optional[np.ndarray], q2: Optional[np.ndarray]) -> float:
    """ Calculates squared Euclidean distance between two C-space configurations. """
    if q1 is None or q2 is None: return 0.0
    try:
        q1_np = np.asarray(q1); q2_np = np.asarray(q2)
        if q1_np.shape != q2_np.shape or q1_np.ndim != 1 or q1_np.shape[0] != config.NUM_JOINTS: return float('inf')
        diff = q1_np - q2_np; return np.dot(diff, diff)
    except Exception: return float('inf')

def initialize_robot_cspace(initial_joint_angles: List[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """ Initializes robot from C-space config, performs FK. """
    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver (for FK) not available."); return None, None, None
    try:
        q_init_c = np.array(initial_joint_angles, dtype=float)
        if q_init_c.shape != (config.NUM_JOINTS,): return None, None, None
        q_init_c_norm = np.array(normalize_angles(q_init_c))
        if not check_joint_limits(q_init_c_norm): print(f"WARNING: Initial C-space config outside limits.")
        joint_positions, T0_TCP, _ = forward_kinematics(q_init_c_norm)
        if T0_TCP is None: raise ValueError("Forward Kinematics returned None")
        return q_init_c_norm, T0_TCP, joint_positions
    except Exception as e: print(f"ERROR during C-space initialization (FK): {e}"); traceback.print_exc(); return None, None, None

# This function selects the 'best' overall IK solution based on cost.
# It's kept for initializing the start state or potential fallbacks.
# The branch-following logic will use the new get_ik_solution_for_branch.
def initialize_robot_taskspace(T_target_B: np.ndarray, current_config_hint: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """ Finds the 'best' C-space config for a target Task Space pose (Base frame) using IK cost evaluation. """
    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver not available."); return None, None, None
    try: raw_solutions_info = inverse_kinematics(T_target_B)
    except Exception as e: print(f"ERROR calling IK: {e}"); return None, None, None
    if not raw_solutions_info: return None, None, None

    best_q_so_far = None; min_total_cost = float('inf'); found_valid = False; q_ref = current_config_hint
    for i, sol_info in enumerate(raw_solutions_info):
        try: # Parse solution format (assuming list or tuple with angles first)
            if isinstance(sol_info, (tuple, list)) and len(sol_info) > 0 and isinstance(sol_info[0], (list, np.ndarray)): q_raw = sol_info[0]
            elif isinstance(sol_info, (list, np.ndarray)): q_raw = sol_info
            else: continue
            q_norm = np.array(normalize_angles(q_raw))
            if not check_joint_limits(q_norm): continue
            cost_manip = calculate_manipulability_cost(q_norm)
            cost_jlim = calculate_joint_limit_cost(q_norm)
            cost_dist = calculate_cspace_distance_sq_cost(q_norm, q_ref)
            if not (np.isfinite(cost_manip) and np.isfinite(cost_jlim) and np.isfinite(cost_dist)): continue
            total_cost = (config.IK_SELECT_WEIGHT_MANIP_COST * cost_manip + config.IK_SELECT_WEIGHT_JLIM_COST * cost_jlim + config.IK_SELECT_WEIGHT_DIST_COST * cost_dist)
            if total_cost < min_total_cost: min_total_cost = total_cost; best_q_so_far = q_norm; found_valid = True
        except Exception as e_eval: print(f"ERROR evaluating IK sol {i+1}: {e_eval}")

    if found_valid and best_q_so_far is not None:
        try: # Final FK check
            joint_positions_best, T_final_check, _ = forward_kinematics(best_q_so_far)
            if T_final_check is None: raise ValueError("FK failed for best IK solution")
            # pose_diff = np.linalg.norm(T_final_check[:3,3] - T_target_B[:3,3]) # Optional check
            # if pose_diff > 1e-3: print(f"WARNING: FK vs target pose diff {pose_diff:.4f}m")
            return best_q_so_far, T_final_check, joint_positions_best
        except Exception as e_fk: print(f"ERROR during final FK for best IK: {e_fk}"); return None, None, None
    else: return None, None, None

# Inside utils.py

# *** NEW HELPER FUNCTION for Branch Following ***
def get_ik_solution_for_branch(T_target_B: np.ndarray,
                               branch_tuple: Tuple[int, int, int],
                               current_config_hint: Optional[np.ndarray] = None # Hint currently unused but kept
                               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Finds the specific IK solution corresponding to the given branch tuple.
    Performs FK check and basic validation.
    """
    # --- DEBUG PRINT ---
    print(f"DEBUG [get_ik_branch]: Target Branch={branch_tuple}, Target Pose Pos={np.round(T_target_B[:3,3], 5)}")
    # --- END DEBUG ---

    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver not available."); return None, None
    try:
        # Get all raw solutions and their branch indices
        raw_solutions_info = inverse_kinematics(T_target_B) # Returns list of (angles, t1, t5, t3)
        # --- DEBUG PRINT ---
        print(f"DEBUG [get_ik_branch]: IK Solver returned {len(raw_solutions_info)} raw solutions.")
        # Optional: Print all raw solutions for comparison across runs
        # for idx, dbg_sol in enumerate(raw_solutions_info):
        #     if isinstance(dbg_sol, (tuple, list)) and len(dbg_sol) == 4:
        #          print(f"  Raw Sol {idx}: Branch={dbg_sol[1:]}, Angles={np.round(np.degrees(dbg_sol[0]), 3)}")
        # --- END DEBUG ---
    except Exception as e:
        print(f"ERROR calling inverse_kinematics in get_ik_solution_for_branch: {e}")
        return None, None

    found_q = None
    # Iterate through solutions to find the one matching the branch tuple
    for sol_idx, sol_info in enumerate(raw_solutions_info):
        try:
            # Check format: expects (angles, t1_idx, t5_idx, t3_idx)
            if isinstance(sol_info, (tuple, list)) and len(sol_info) == 4:
                q_raw, t1_idx, t5_idx, t3_idx = sol_info
                current_branch_tuple = (t1_idx, t5_idx, t3_idx)

                # Check if this solution matches the desired branch
                if current_branch_tuple == branch_tuple:
                    # --- DEBUG PRINT ---
                    print(f"DEBUG [get_ik_branch]: Found matching raw solution {sol_idx} for branch {branch_tuple}. Raw Angles (deg): {np.round(np.degrees(q_raw), 3)}")
                    # --- END DEBUG ---
                    q_norm = np.array(normalize_angles(q_raw))
                    # Validate joint limits
                    if check_joint_limits(q_norm):
                        found_q = q_norm
                        # --- DEBUG PRINT ---
                        print(f"DEBUG [get_ik_branch]: Solution passed joint limits. Selected q_norm (deg): {np.round(np.degrees(found_q), 3)}")
                        # --- END DEBUG ---
                        break # Found the matching valid solution
                    else:
                        # --- DEBUG PRINT ---
                        print(f"DEBUG [get_ik_branch]: Solution {sol_idx} for branch {branch_tuple} FAILED joint limits.")
                        # --- END DEBUG ---
            # else: print("Warning: Unexpected IK solution format.") # Debug
        except Exception as e_parse:
            print(f"Error parsing IK solution info {sol_idx}: {e_parse}")
            continue # Skip malformed solutions

    # If a valid solution for the branch was found, perform FK check
    if found_q is not None:
        try:
            _, T_branch_check, _ = forward_kinematics(found_q)
            if T_branch_check is None:
                print(f"ERROR: FK failed for selected branch {branch_tuple} solution.")
                return None, None # FK failure is critical

            # Optional: Check if T_branch_check is close to T_target_B
            pos_diff = np.linalg.norm(T_branch_check[:3,3] - T_target_B[:3,3])
            # --- DEBUG PRINT ---
            print(f"DEBUG [get_ik_branch]: FK Check for branch {branch_tuple}. Result Pos={np.round(T_branch_check[:3,3], 5)}. Pos Diff from Target={pos_diff:.6f}")
            # --- END DEBUG ---
            if pos_diff > 1e-3: # Adjust tolerance as needed
                print(f"WARNING: FK of branch {branch_tuple} solution differs significantly from target pose by {pos_diff:.4f}m")

            return found_q, T_branch_check
        except Exception as e_fk:
            print(f"ERROR during final FK for branch {branch_tuple} solution: {e_fk}")
            return None, None
    else:
        # --- DEBUG PRINT ---
        print(f"DEBUG [get_ik_branch]: No valid solution found or selected for branch {branch_tuple}.")
        # --- END DEBUG ---
#         return None, None # No valid solution found for this branch
# # *** NEW HELPER FUNCTION for Branch Following ***
# def get_ik_solution_for_branch(T_target_B: np.ndarray,
#                                branch_tuple: Tuple[int, int, int],
#                                current_config_hint: Optional[np.ndarray] = None
#                                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#     """
#     Finds the specific IK solution corresponding to the given branch tuple.
#     Performs FK check and basic validation.

#     Args:
#         T_target_B (np.ndarray): Target 4x4 TCP pose matrix relative to the Base frame.
#         branch_tuple (Tuple[int, int, int]): The desired branch indices (t1_idx, t5_idx, t3_idx).
#         current_config_hint (Optional[np.ndarray]): Optional hint for IK solver (not used by closed-form but kept for signature).

#     Returns:
#         Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#             - q_branch: Joint configuration (radians) for the specific branch, or None if not found/valid.
#             - T_branch_check: Pose achieved by q_branch from FK check, or None.
#     """
#     if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver not available."); return None, None
#     try:
#         # Get all raw solutions and their branch indices
#         raw_solutions_info = inverse_kinematics(T_target_B) # Returns list of (angles, t1, t5, t3)
#     except Exception as e:
#         print(f"ERROR calling inverse_kinematics in get_ik_solution_for_branch: {e}")
#         return None, None

#     found_q = None
#     # Iterate through solutions to find the one matching the branch tuple
#     for sol_info in raw_solutions_info:
#         try:
#             # Check format: expects (angles, t1_idx, t5_idx, t3_idx)
#             if isinstance(sol_info, (tuple, list)) and len(sol_info) == 4:
#                 q_raw, t1_idx, t5_idx, t3_idx = sol_info
#                 current_branch_tuple = (t1_idx, t5_idx, t3_idx)

#                 # Check if this solution matches the desired branch
#                 if current_branch_tuple == branch_tuple:
#                     q_norm = np.array(normalize_angles(q_raw))
#                     # Validate joint limits
#                     if check_joint_limits(q_norm):
#                         found_q = q_norm
#                         # print(f"  Found matching IK solution for branch {branch_tuple}") # Debug
#                         break # Found the matching valid solution
#                     # else:
#                         # print(f"  Branch {branch_tuple} solution failed joint limits.") # Debug
#             # else: print("Warning: Unexpected IK solution format.") # Debug
#         except Exception as e_parse:
#             print(f"Error parsing IK solution info: {e_parse}")
#             continue # Skip malformed solutions

#     # If a valid solution for the branch was found, perform FK check
#     if found_q is not None:
#         try:
#             _, T_branch_check, _ = forward_kinematics(found_q)
#             if T_branch_check is None:
#                 print(f"ERROR: FK failed for selected branch {branch_tuple} solution.")
#                 return None, None # FK failure is critical

#             # Optional: Check if T_branch_check is close to T_target_B
#             pose_diff = np.linalg.norm(T_branch_check[:3,3] - T_target_B[:3,3])
#             if pose_diff > 1e-3: # Adjust tolerance as needed
#                 print(f"WARNING: FK of branch {branch_tuple} solution differs from target pose by {pose_diff:.4f}m")

#             return found_q, T_branch_check
#         except Exception as e_fk:
#             print(f"ERROR during final FK for branch {branch_tuple} solution: {e_fk}")
#             return None, None
#     else:
#         # print(f"  No valid IK solution found for branch {branch_tuple}.") # Debug
#         return None, None # No valid solution found for this branch


# # --- REMOVED: sample_pose_from_set ---
# # This function was specific to the traditional approach

# # --- REMOVED: generate_perturbed_config ---
# This function was likely obsolete


print(f"utils.py loaded (SCIPY_AVAILABLE={SCIPY_AVAILABLE}). Probabilistic approach only.")
