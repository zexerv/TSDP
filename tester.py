# tester.py
import pandas as pd
import numpy as np
import re
import os
import config # Import config to get the threshold

# --- Configuration for Tester ---
# !!! --- UPDATE THESE PATHS --- !!!
# Path to the folder containing your CSV files (e.g., 1.csv, 2.csv)
DATA_FOLDER = config.PARENT_FOLDER_PATH # Use path from config

# Specific files to test (relative to DATA_FOLDER)
FILE_ID_1 = '2.csv' # A file where interface_id should be 1
FILE_ID_0 = '1.csv' # A file where interface_id should be 0
# !!! --- END UPDATE --- !!!

# Threshold (usually from config.py)
LEVER_THRESHOLD = config.LEVER_CHANGE_THRESHOLD

# --- Helper Function: Load CSV ---
def load_csv(filepath):
    """Loads a single CSV file and strips column whitespace."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found - {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip() for col in df.columns]
        print(f"Successfully loaded {filepath}, shape: {df.shape}")
        # Print first few rows of key columns
        print("First 5 rows of key columns:")
        key_cols = ['interface_id'] + [c for c in df.columns if c.startswith('TCA_lever_')]
        print(df[key_cols].head())
        print("-" * 20)
        return df
    except Exception as e:
        print(f"ERROR: Failed to load or process {filepath}: {e}")
        return None

# --- Test Function: Column Matching Logic ---
def test_column_matching(df, interface_id_to_test):
    """Tests the regex column matching logic from find_events."""
    print(f"\n--- Testing Column Matching for ID = {interface_id_to_test} ---")
    if df is None:
        print("DataFrame is None. Skipping test.")
        return None, None

    if 'interface_id' not in df.columns:
        print("ERROR: 'interface_id' column missing from DataFrame.")
        return None, None

    # Verify the actual ID in the first row matches the expected ID
    try:
        actual_id_in_df = int(df['interface_id'].iloc[0])
        if actual_id_in_df != interface_id_to_test:
             print(f"WARNING: Expected ID {interface_id_to_test} but found ID {actual_id_in_df} in DataFrame's first row.")
             # Continue test with actual_id_in_df for pattern matching robustness
             interface_id_to_use = actual_id_in_df
        else:
             interface_id_to_use = interface_id_to_test
             print(f"Confirmed interface_id {interface_id_to_use} in first row.")
    except Exception as e:
        print(f"ERROR: Could not read or convert interface_id from DataFrame: {e}")
        return None, None

    potential_matches = []
    target_col = None
    interface_type = None
    try:
        # Define the pattern using the ID we will actually use for matching
        pattern = re.compile(rf"TCA_(?P<type>button|switch|lever)_{interface_id_to_use}_(?P<suffix>state|value)")
        print(f"Using Regex Pattern: '{pattern.pattern}'")
        print(f"Checking columns:")

        all_columns = df.columns.tolist()
        print(f"  Available columns: {all_columns}") # Print all columns once

        found_target_lever_col = False # Flag specifically for lever column
        target_lever_col_name = f"TCA_lever_{interface_id_to_use}_value"

        for col_name in all_columns:
            # print(f"  - Checking: '{col_name}'") # Can be very verbose
            match = pattern.match(col_name)
            if match:
                print(f"    MATCH FOUND for '{col_name}'! Type='{match.group('type')}', Suffix='{match.group('suffix')}'")
                potential_matches.append({
                    "col_name": col_name,
                    "type": match.group('type'),
                    "suffix": match.group('suffix')
                })
            # Explicit check if the expected lever column didn't match
            if col_name == target_lever_col_name:
                 found_target_lever_col = True
                 if not match and interface_id_to_use == 1: # If it *should* have matched but didn't
                     print(f"    ERROR: Explicit check failed! Pattern did not match expected column '{col_name}'")


        print(f"Finished checking columns. Found {len(potential_matches)} potential matches.")

        if len(potential_matches) == 1:
            target_col = potential_matches[0]["col_name"]
            interface_type = potential_matches[0]["type"]
            print(f"Result: Success - Found unique target column: '{target_col}' (Type: {interface_type})")
        elif len(potential_matches) == 0:
            print(f"Result: Failure - No column matched the pattern.")
            if found_target_lever_col and interface_id_to_use == 1:
                 print("  >> The expected column 'TCA_lever_1_value' WAS present but regex failed! Check pattern/name closely.")
        else:
            print(f"Result: Failure - Found multiple matching columns (ambiguous): {[m['col_name'] for m in potential_matches]}")

    except Exception as e:
         print(f"ERROR during column matching process: {e}")

    return target_col, interface_type


# --- Test Function: Lever Start Detection Logic ---
def test_lever_start(df, target_col_name, threshold):
    """Tests the revised lever start detection logic."""
    print(f"\n--- Testing Lever Start Detection for Column = '{target_col_name}' ---")
    if df is None:
        print("DataFrame is None. Skipping test.")
        return None
    if target_col_name is None or target_col_name not in df.columns:
        print(f"Target column '{target_col_name}' not provided or not found in DataFrame. Skipping test.")
        return None

    n_points = len(df)
    if n_points <= 1:
        print("Not enough data points (<=1) to detect change. Skipping test.")
        return None

    detected_start_index = None
    try:
        col_data = df[target_col_name]
        df_indices = df.index # Get actual index labels

        # 1. Check overall range
        numeric_col_data = pd.to_numeric(col_data, errors='coerce').dropna()
        if numeric_col_data.empty:
            print("Lever column contains no valid numeric data. Cannot calculate range.")
            return None

        min_val = numeric_col_data.min()
        max_val = numeric_col_data.max()
        total_range = max_val - min_val
        print(f"Lever Range Check: Min={min_val:.4f}, Max={max_val:.4f}, Range={total_range:.4f}, Threshold={threshold}")

        if total_range >= threshold:
            print("Range Check Passed (>= Threshold). Searching for first movement...")
            # 2. Find first movement
            diffs = col_data.diff()
            epsilon = 1e-6
            first_move_found = False

            for k in range(1, n_points):
                diff_val = diffs.iloc[k] # Difference between k and k-1
                current_index = df_indices[k] # Get the actual index label

                # Optional: Print diffs around potential start
                # if k < 10 or (detected_start_index and k < detected_start_index + 5):
                # print(f"  k={k}, index={current_index}, diff={diff_val}")

                if pd.notna(diff_val) and abs(diff_val) > epsilon:
                    detected_start_index = current_index
                    print(f"Result: First Movement Detected at index = {detected_start_index} (diff = {diff_val:.6f})")
                    first_move_found = True
                    break # Stop searching

            if not first_move_found:
                print("Result: Range >= threshold, but no movement (> epsilon) detected after index 0.")
        else:
            print("Result: Total range < threshold. No state change start marked.")

    except KeyError:
         print(f"ERROR: Column '{target_col_name}' not found during processing.")
    except Exception as e:
        print(f"ERROR during lever start detection: {e}")

    return detected_start_index


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Tester Script...")

    filepath_id1 = os.path.join(DATA_FOLDER, FILE_ID_1)
    filepath_id0 = os.path.join(DATA_FOLDER, FILE_ID_0)

    # --- Test ID 1 File ---
    print("\n" + "="*30 + f" TESTING FILE ID=1 ({FILE_ID_1}) " + "="*30)
    df1 = load_csv(filepath_id1)
    if df1 is not None:
        target_col_1, type_1 = test_column_matching(df1, 1)
        # Only test lever logic if the column found was indeed the lever column
        if target_col_1 == f"TCA_lever_1_value":
             test_lever_start(df1, target_col_1, LEVER_THRESHOLD)
        elif target_col_1:
             print(f"Column matching found '{target_col_1}' (type {type_1}), skipping lever start test.")
        else:
             print("Column matching failed for ID 1, skipping lever start test.")

    # --- Test ID 0 File ---
    print("\n" + "="*30 + f" TESTING FILE ID=0 ({FILE_ID_0}) " + "="*30)
    df0 = load_csv(filepath_id0)
    if df0 is not None:
        target_col_0, type_0 = test_column_matching(df0, 0)
        # Only test lever logic if the column found was indeed the lever column
        if target_col_0 == f"TCA_lever_0_value":
            test_lever_start(df0, target_col_0, LEVER_THRESHOLD)
        elif target_col_0:
             print(f"Column matching found '{target_col_0}' (type {type_0}), skipping lever start test.")
        else:
             print("Column matching failed for ID 0, skipping lever start test.")


    print("\n" + "="*30 + " TESTING COMPLETE " + "="*30)