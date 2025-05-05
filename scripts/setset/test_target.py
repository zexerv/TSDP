#!/usr/bin/env python3
import numpy as np
import sys
import time
import yaml
from pathlib import Path
import random

# --- Assume these modules are in the same directory or accessible ---
try:
    import config
    import utils
    # Import the specific IK/FK functions needed
    from ik_solver import forward_kinematics, inverse_kinematics, normalize_angles
    print("Successfully imported config, utils, and ik_solver functions.")
except ImportError as e:
    print(f"FATAL: Failed to import necessary modules: {e}")
    print("Make sure this script is run from the correct directory (e.g., where main.py is)")
    print("and config.py, utils.py, ik_solver.py are accessible.")
    sys.exit(1)

# --- Configuration for the Test ---
NUM_TEST_RUNS = 10  # How many times to call IK for the same target
BOUNDARY_INDEX_TO_TEST = 0 # Which boundary GMM to use from the list (0 = first)
# Use a deterministic method for GMM extraction
GMM_EXTRACTION_METHOD = 'mean_of_best_component'
# Define a few slightly different, valid hint configurations (in radians)
# These simulate the slightly varying q_current from OMPL runs
HINT_CONFIGS = [
    np.radians([0, -90, 0, -90, 0, 0]),
    np.radians([5, -85, 5, -85, 5, 5]),
    np.radians([-5, -95,-5, -95,-5, -5]),
    np.radians([10, -80, 10, -80, 10, 10]),
]
# Ensure hints are valid according to your joint limits if necessary

print("--- Test Script Configuration ---")
print(f"Number of IK Runs per Target: {NUM_TEST_RUNS}")
print(f"Boundary Index Tested: {BOUNDARY_INDEX_TO_TEST}")
print(f"GMM Extraction Method: {GMM_EXTRACTION_METHOD}")
print(f"Number of Hint Variations: {len(HINT_CONFIGS)}")
print("-" * 30)

# --- 1. Load Common Configurations & Transforms ---
print("Loading common configurations...")
H_I_B = None
H_B_I = None
interface_transforms = None
H_B_D = None
global_interface_id = config.INTERFACE_ID

try:
    interface_transforms = utils.load_interface_transforms(config.INTERFACE_TRANSFORMS_YAML_PATH)
    if interface_transforms is None: raise ValueError("Failed to load interface transforms.")
    H_B_D = utils.load_environment_config(config.ENVIRONMENT_YAML_PATH)
    if H_B_D is None: raise ValueError("Failed to load environment config.")

    if not global_interface_id: raise ValueError("INTERFACE_ID not set in config.py")
    if global_interface_id not in interface_transforms: raise ValueError(f"Interface ID '{global_interface_id}' not found.")

    H_D_I = interface_transforms[global_interface_id]
    H_B_I = H_B_D @ H_D_I
    H_I_B = np.linalg.inv(H_B_I) # Needed for visualization comparison if desired
    print("Transforms H_B_I and H_I_B calculated.")

except Exception as e:
    print(f"FATAL: Error during configuration loading: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- 2. Load Boundary GMM Data ---
print("\nLoading probabilistic boundary stats...")
boundary_stats_list = None
try:
    boundary_stats_list = utils.load_yaml_file(config.PROBABILISTIC_STATS_YAML_PATH)
    if boundary_stats_list is None or not isinstance(boundary_stats_list, list) or not boundary_stats_list:
        raise ValueError(f"Probabilistic Stats YAML invalid, empty, or not a list: {config.PROBABILISTIC_STATS_YAML_PATH}")
    if BOUNDARY_INDEX_TO_TEST >= len(boundary_stats_list):
        raise IndexError(f"Boundary index {BOUNDARY_INDEX_TO_TEST} is out of range (0-{len(boundary_stats_list)-1}).")
    print(f"Loaded {len(boundary_stats_list)} boundaries.")
    boundary_data_to_test = boundary_stats_list[BOUNDARY_INDEX_TO_TEST]
    gmm_params = boundary_data_to_test.get('segment_gmm_params')
    if gmm_params is None:
        raise ValueError(f"Missing 'segment_gmm_params' for boundary index {BOUNDARY_INDEX_TO_TEST}.")
    print(f"Using GMM parameters from boundary index {BOUNDARY_INDEX_TO_TEST}.")

except Exception as e:
    print(f"FATAL: Error loading or selecting boundary data: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- 3. Extract the SINGLE Target Pose (Deterministic) ---
print(f"\nExtracting target pose using method: '{GMM_EXTRACTION_METHOD}'...")
T_B_target = utils.extract_target_pose_from_gmm(gmm_params, GMM_EXTRACTION_METHOD, H_B_I)

if T_B_target is None:
    print("FATAL: Failed to extract a target pose from the GMM. Cannot proceed.")
    sys.exit(1)

target_position = T_B_target[:3, 3]
print(f"Determined Target Pose T_B_target (Position: {np.round(target_position, 5)})")
# print(f"Target Pose Matrix:\n{np.round(T_B_target, 4)}") # Optional: print full matrix

# --- 4. Repeatedly Call IK with Different Hints ---
print(f"\nRunning IK {NUM_TEST_RUNS} times for the SAME target pose with varying hints...")
resulting_positions = []
resulting_configs = []
successful_ik_calls = 0
failed_ik_calls = 0

for i in range(NUM_TEST_RUNS):
    # Cycle through the hint configurations
    current_hint = HINT_CONFIGS[i % len(HINT_CONFIGS)]
    print(f" Run {i+1}/{NUM_TEST_RUNS} (Hint: {i % len(HINT_CONFIGS)} - {np.round(np.degrees(current_hint),1)} deg)")

    # Call the IK solver function from utils
    q_best, T_final_check, _ = utils.initialize_robot_taskspace(
        T_B_target, # Use the SAME target pose each time
        current_config_hint=current_hint
    )

    if q_best is not None and T_final_check is not None:
        successful_ik_calls += 1
        final_position = T_final_check[:3, 3]
        resulting_positions.append(final_position)
        resulting_configs.append(q_best)
        # Optional: Check how close this position is to the target
        pos_diff = np.linalg.norm(final_position - target_position)
        print(f"  -> Success! Found q_best. Resulting Pos: {np.round(final_position, 5)}. Diff from target: {pos_diff:.6f} m")
    else:
        failed_ik_calls += 1
        print(f"  -> IK Failed for this hint.")

print(f"\nFinished IK runs. Successes: {successful_ik_calls}, Failures: {failed_ik_calls}")

# --- 5. Analyze Position Results ---
if not resulting_positions:
    print("\nFATAL: No successful IK calls were made. Cannot analyze results.")
    sys.exit(1)

print("\n--- Position Variation Analysis ---")
positions_array = np.array(resulting_positions) # Shape (N_success, 3)

# Calculate stats for each axis (X, Y, Z)
min_pos = np.min(positions_array, axis=0)
max_pos = np.max(positions_array, axis=0)
mean_pos = np.mean(positions_array, axis=0)
std_dev_pos = np.std(positions_array, axis=0)
range_pos = max_pos - min_pos

print(f"Target Position:      {np.round(target_position, 6)}")
print(f"Mean Result Position: {np.round(mean_pos, 6)}")
print("-" * 20)
print(f"Min Result Position:  {np.round(min_pos, 6)}")
print(f"Max Result Position:  {np.round(max_pos, 6)}")
print(f"Range (Max-Min):    {np.round(range_pos, 6)}")
print(f"Std Deviation:      {np.round(std_dev_pos, 6)}")
print("-" * 20)

# Highlight the maximum range observed across all axes
max_variation = np.max(range_pos)
print(f"\n>>> Maximum Position Variation Observed (Max Range): {max_variation:.6f} meters <<<")

if max_variation > 0.01: # Threshold for significant variation (e.g., 1 cm)
    print("\nWARNING: Significant position variation detected!")
    print("This suggests the issue might be within 'initialize_robot_taskspace' logic,")
    print("the underlying IK/FK implementation, or how poses are handled,")
    print("rather than just the hint variation affecting IK *selection*.")
    # Optional: Print the configs that led to min/max positions
    idx_min_x = np.argmin(positions_array[:, 0])
    idx_max_x = np.argmax(positions_array[:, 0])
    # ... (similarly for y, z) ...
    # print(f"Config for Min X: {np.round(np.degrees(resulting_configs[idx_min_x]), 2)}")
    # print(f"Config for Max X: {np.round(np.degrees(resulting_configs[idx_max_x]), 2)}")

elif max_variation > 1e-5: # Threshold for minor variation
    print("\nMinor position variation detected.")
    print("This level might be acceptable due to numerical precision and IK tolerance,")
    print("especially if different IK solutions are selected due to the hint.")
else:
    print("\nNegligible position variation detected.")
    print("The resulting position is consistent for the target pose, as expected.")

print("\nTest finished.")
