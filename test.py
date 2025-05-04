#!/usr/bin/env python3
import unittest
import os
import pandas as pd
import numpy as np # <--- IMPORT NUMPY HERE
import re
import sys
from pathlib import Path

# --- Configuration ---
# !!! IMPORTANT: Set the ABSOLUTE path to the PARENT data directory !!!
# This is the directory containing 'button', 'lever', 'switch' etc.
PARENT_DATA_DIR_PATH = '/home/kadi/Desktop/Thesis/demo_processor_new/data/reorganized_data'

# --- Make sure we can import sibling modules ---
# Get the directory where this script (test_data.py) is located
# Assuming it's in scripts/segdp
script_dir = Path(__file__).parent.resolve()
# Add the script's directory to the Python path to allow direct imports
sys.path.insert(0, str(script_dir))

try:
    # Now try importing the necessary functions/variables
    from data_loading import find_events # Assuming find_events is in data_loading.py
    import config # Import config to get LEVER_CHANGE_THRESHOLD etc.
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules.")
    print(f"Ensure 'data_loading.py' and 'config.py' are in the same directory as this script ({script_dir})")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during imports: {e}")
    sys.exit(1)


# --- Helper Function to find the first actual change ---
# This helps verify if the detected index matches reality
def find_first_change_index(series, change_type, threshold=1e-6):
    """
    Finds the index of the first significant change in a pandas Series.

    Args:
        series (pd.Series): The data series (e.g., button state or lever value).
        change_type (str): 'button'/'switch' or 'lever'.
        threshold (float): Minimum difference for lever change.

    Returns:
        int or None: The index label of the first change, or None if no change found.
    """
    if series.empty:
        return None
    n_points = len(series)
    if n_points < 2:
        return None

    if change_type in ['button', 'switch']:
        # Look for 0 -> 1 transition (handle various types)
        try:
            def is_active(value):
                try: return float(value) == 1.0
                except (ValueError, TypeError): return str(value).lower() in ['true', 'on', '1']
            def is_inactive(value):
                try: return float(value) == 0.0
                except (ValueError, TypeError): return str(value).lower() in ['false', 'off', '0']

            current_active = series.apply(is_active)
            prev_inactive = series.shift(1).apply(is_inactive)
            change_mask = (current_active & prev_inactive).fillna(False)
            
            # Use np.where, which requires numpy import
            first_change_iloc = np.where(change_mask)[0] 
            if len(first_change_iloc) > 0:
                return series.index[first_change_iloc[0]] # Return index label
            else:
                return None
        except Exception:
            # Fallback for non-standard button/switch data: check any difference
            diffs = series.diff().abs()
            # Use np.where, which requires numpy import
            first_change_iloc = np.where(diffs > threshold)[0] 
            if len(first_change_iloc) > 0:
                 return series.index[first_change_iloc[0]] # Return index label
            else:
                return None


    elif change_type == 'lever':
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            diffs = numeric_series.diff().abs()
            # Find first difference greater than threshold
            # Use np.where, which requires numpy import
            first_change_iloc = np.where(diffs > threshold)[0] 
            if len(first_change_iloc) > 0:
                return series.index[first_change_iloc[0]] # Return index label
            else:
                return None
        except Exception as e:
            print(f"      [WARN] Error calculating lever diff: {e}")
            return None
    else:
        return None


# --- Test Class ---
class TestEventDetection(unittest.TestCase):

    def test_state_change_detection(self):
        """
        Tests the 'state_change' detection in find_events across all subdirs.
        """
        parent_dir = Path(PARENT_DATA_DIR_PATH)
        if not parent_dir.is_dir():
            self.fail(f"Parent data directory not found: {PARENT_DATA_DIR_PATH}")

        subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name != '__pycache__'] # Exclude __pycache__
        if not subdirs:
            self.fail(f"No subdirectories (like 'button', 'lever') found in {PARENT_DATA_DIR_PATH}")

        print("\n--- Starting State Change Detection Test ---")

        found_any_csv = False
        overall_success = True

        for subdir in subdirs:
            interface_type_from_dir = subdir.name.split('_')[-1] # e.g., 'button', 'lever', 'switch'
            print(f"\n[Testing Directory: {subdir.name} (Assumed Type: {interface_type_from_dir})] ")

            csv_files = sorted(list(subdir.glob('*.csv')))
            if not csv_files:
                print("  No CSV files found in this directory.")
                continue

            found_any_csv = True

            for csv_file in csv_files:
                print(f"\n  Testing File: {csv_file.name}")
                try:
                    # Load only necessary columns - ensure 'interface_id' and all potential TCA columns are loaded
                    # It's simpler to load all columns for testing purposes here
                    df = pd.read_csv(csv_file)
                    df.columns = [col.strip() for col in df.columns] # Clean column names

                    if df.empty:
                        print("    -> DataFrame is empty. Skipping event check.")
                        continue

                    # --- Run find_events (the function under test) ---
                    # We pass a minimal event_config, focusing on state_change logic
                    # Note: find_events itself prints warnings if it can't find the TCA col
                    # Use a copy of the relevant part of the config event columns
                    event_config_for_test = {'state_change': config.EVENT_COLUMNS.get('state_change')}
                    events_found = find_events(df, event_config_for_test)
                    # -------------------------------------------------

                    # --- Verification and Reporting ---
                    self.assertIsInstance(events_found, dict, "find_events should return a dict")

                    # 1. Check interface_id reading
                    interface_id_read = None
                    if 'interface_id' in df.columns:
                        first_id_val = df['interface_id'].iloc[0]
                        if pd.notna(first_id_val):
                            try:
                                interface_id_read = int(first_id_val)
                                print(f"    Interface ID found in file: {interface_id_read}")
                            except (ValueError, TypeError):
                                print(f"    [WARN] Could not convert interface_id '{first_id_val}' to int.")
                        else:
                             print(f"    [WARN] interface_id in first row is NaN.")
                    else:
                        print(f"    [WARN] 'interface_id' column missing.")

                    # 2. Check if find_events detected a state change
                    detected_indices = events_found.get('state_change', [])
                    if detected_indices:
                        print(f"    ✅ find_events DETECTED state_change at index(es): {detected_indices}")
                    else:
                        print(f"    ℹ️ find_events did NOT detect state_change.")

                    # 3. Try to manually verify based on ID and expected column type
                    if interface_id_read is not None:
                        expected_col_found = False
                        target_col_name = None
                        actual_interface_type = None

                        # Try to find the column find_events *should* have looked for
                        pattern = re.compile(rf"TCA_(?P<type>button|switch|lever)_{interface_id_read}_(?P<suffix>state|value)")
                        potential_matches = []
                        for col_name in df.columns:
                           match = pattern.match(col_name)
                           if match:
                               potential_matches.append({
                                   "col_name": col_name, "type": match.group('type'), "suffix": match.group('suffix')
                               })

                        if len(potential_matches) == 1:
                            target_col_name = potential_matches[0]["col_name"]
                            actual_interface_type = potential_matches[0]["type"]
                            print(f"    Manually identified target column: '{target_col_name}' (Type: {actual_interface_type})")
                            expected_col_found = True

                            # 4. Check if the data in that column actually changes
                            if target_col_name in df.columns:
                                first_actual_change = find_first_change_index(
                                    df[target_col_name],
                                    actual_interface_type,
                                    config.LEVER_CHANGE_THRESHOLD # Use threshold from config
                                )
                                if first_actual_change is not None:
                                     print(f"    Manually found first change in '{target_col_name}' at index: {first_actual_change}")
                                     # Compare manual find with find_events detection
                                     if detected_indices:
                                         # Check if the first manually found index is in the detected list
                                         if first_actual_change in detected_indices:
                                             print("      -> Matches find_events detection. 👍")
                                         else:
                                             print(f"      [MISMATCH] Manual change index {first_actual_change} NOT in find_events list {detected_indices}. Potential logic issue?")
                                             overall_success = False
                                     else:
                                         print(f"      [MISMATCH] Manual change found, but find_events detected nothing. Potential logic issue?")
                                         overall_success = False

                                else:
                                    print(f"    Manually found NO significant change in '{target_col_name}'.")
                                    if detected_indices:
                                        print(f"      [MISMATCH] No manual change, but find_events detected change at {detected_indices}. Potential logic issue?")
                                        overall_success = False
                                    else:
                                         print("      -> Consistent with find_events (no detection). 👍")

                            else:
                                # This should not happen if pattern matched
                                print(f"    [ERROR] Identified target column '{target_col_name}' not actually in DataFrame columns!")
                                overall_success = False

                        elif len(potential_matches) == 0:
                             print(f"    Manually confirmed: NO column found matching pattern for ID {interface_id_read}.")
                             if detected_indices:
                                 print(f"      [MISMATCH] No target column, but find_events detected change at {detected_indices}. How?")
                                 overall_success = False
                             else:
                                 print("      -> Consistent with find_events (no detection). 👍")

                        else: # Multiple matches
                             print(f"    Manually confirmed: MULTIPLE columns matching pattern for ID {interface_id_read}: {[m['col_name'] for m in potential_matches]}. Ambiguous.")
                             if detected_indices:
                                 print(f"      [MISMATCH] Ambiguous target, but find_events detected change at {detected_indices}. How?")
                                 overall_success = False
                             else:
                                 print("      -> Consistent with find_events (no detection due to ambiguity). 👍")

                    else:
                        print("    Skipping manual verification (interface_id not determined).")
                        if detected_indices:
                             print(f"    [WARN] No interface ID, but find_events detected change at {detected_indices}. How?")
                             overall_success = False


                except FileNotFoundError:
                    print(f"    [ERROR] File not found: {csv_file}")
                    overall_success = False
                except pd.errors.EmptyDataError:
                    print(f"    [WARN] File is empty: {csv_file.name}")
                except Exception as e:
                    print(f"    [ERROR] Failed to process file {csv_file.name}: {e}")
                    # Optionally re-raise if you want the test to hard fail on any error
                    # raise e
                    overall_success = False


        print("\n--- Test Summary ---")
        if not found_any_csv:
             self.fail(f"No CSV files were found to test in any subdirectories of {PARENT_DATA_DIR_PATH}")

        self.assertTrue(overall_success, "One or more inconsistencies or errors were found during state change detection tests. See logs above.")
        print("State change detection test finished.")


if __name__ == '__main__':
    unittest.main() # Use unittest.main() to run the tests\