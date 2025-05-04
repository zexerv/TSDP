import os
import pandas as pd
import numpy as np
from collections import defaultdict
import config # Import configuration
import re # Import regex module

def load_selected_data(parent_folder, load_all=True, file_list=None, columns_to_load=None):
    """
    Loads trajectories from CSV files, ensuring specified columns are present.

    Args:
        parent_folder (str): Path to the folder containing CSV files.
        load_all (bool): Whether to load all CSVs in the folder.
        file_list (list, optional): Specific list of filenames to load if load_all is False.
        columns_to_load (list): List of column names required in the CSV files.

    Returns:
        tuple: (list of DataFrames, list of loaded filenames)
               Returns ([], []) if errors occur or no data is loaded.
    """
    all_trajectories = []
    loaded_filenames = []

    if columns_to_load is None:
        print("Warning: No columns specified to load.")
        columns_to_load = [] # Default to empty, but should be provided

    if load_all:
        try:
            all_files_in_dir = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]
            # Filter only CSV files
            filenames = [f for f in all_files_in_dir if f.endswith('.csv')]
            # Try to sort numerically based on filename before extension
            try:
                filenames.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                filenames.sort() # Fallback to alphabetical sort if names aren't purely numeric
            print(f"Found {len(filenames)} CSV files to load: {filenames}")
            if not filenames:
                print(f"Warning: No CSV files found in {parent_folder}")
                return [], []
        except FileNotFoundError:
            print(f"Error: Parent folder not found - {parent_folder}")
            return [], []
    elif file_list:
        filenames = file_list
        print(f"Loading specified files: {filenames}")
    else:
        print("Error: No files specified to load.")
        return [], []

    required_columns_set = set(columns_to_load if columns_to_load else [])

    for filename in filenames:
        file_path = os.path.join(parent_folder, filename)
        try:
            # Load the CSV. We rely on the calling script (main.py via config.py)
            # to provide a comprehensive list in columns_to_load.
            # If a column in columns_to_load is missing, pandas might raise error later,
            # or we handle it by checking available columns.
            # For simplicity here, load all columns first, then select.
            # More efficient would be pd.read_csv(..., usecols=...) but requires error handling if cols missing.
            df_full = pd.read_csv(file_path)
            df_full.columns = [col.strip() for col in df_full.columns] # Clean column names

            # Check which of the requested columns are actually present
            available_cols = [col for col in columns_to_load if col in df_full.columns]
            missing_cols = required_columns_set - set(available_cols)

            if missing_cols:
                # Don't raise error immediately, allows loading files missing optional event columns
                print(f"Warning: File '{filename}' is missing requested columns: {missing_cols}. Will proceed with available columns.")

            if not available_cols:
                 print(f"Warning: File '{filename}' contains NONE of the requested columns. Skipping.")
                 continue

            # Select only the available columns from the requested list
            df = df_full[available_cols]
            all_trajectories.append(df)
            loaded_filenames.append(filename)

        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except pd.errors.EmptyDataError:
             print(f"Warning: File '{filename}' is empty. Skipping.")
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    if not all_trajectories:
        print("Warning: No valid trajectory data loaded.")

    return all_trajectories, loaded_filenames


# --- UPDATED find_events function ---
def find_events(df, event_config):
    """
    Identifies indices of specified events within a single trajectory DataFrame.
    'state_change' logic updated for levers: Marks the *first* point of movement
    if the total range of the lever exceeds the threshold. Buttons/switches still use 0->1.

    Args:
        df (pd.DataFrame): The input trajectory data (MUST contain 'interface_id' and relevant TCA columns).
        event_config (dict): Configuration specifying columns for 'wp_saved', 'gripper'.
                           Note: 'state_change' key in event_config is NOT used by this function.

    Returns:
        dict: Event dictionary ('start', 'end', 'state_change', 'wp_saved', 'gripper_change').
              Returns an empty dict structure even if df is empty or errors occur finding events.
    """
    # Initialize with empty lists using defaultdict
    events = defaultdict(list)
    # Ensure basic keys exist even if df is empty
    for key in ['start', 'end', 'state_change', 'wp_saved', 'gripper_change']:
        events[key] = []

    if df.empty:
        print("Warning: find_events received an empty DataFrame.")
        return dict(events) # Return the initialized empty dict

    n_points = len(df)
    df_indices = df.index # Get the actual index values (can be non-sequential)

    # 1. Start and End
    if n_points > 0:
        events['start'] = [df_indices[0]]
        events['end'] = [df_indices[-1]]
    else: # Should not happen if df is not empty, but for safety
        return dict(events)


    # 2. WP Saved Event
    wp_col = event_config.get('wp_saved')
    if wp_col and wp_col in df.columns:
        try:
            # Ensure index alignment when creating boolean Series
            wp_mask = (pd.to_numeric(df[wp_col], errors='coerce') == 1.0)
            events['wp_saved'] = df_indices[wp_mask].tolist()
        except Exception as e:
            print(f"Warning: Could not process 'wp_saved' column '{wp_col}': {e}")

    # 3. Gripper State Change
    gripper_col = event_config.get('gripper')
    if gripper_col and gripper_col in df.columns:
        try:
            # Ensure diff is calculated correctly even with non-numeric indices
            gripper_data = df[gripper_col]
            # Compare current to previous using shift, handling potential NaNs from shift
            gripper_change_mask = (gripper_data != gripper_data.shift(1)) & pd.notna(gripper_data) & pd.notna(gripper_data.shift(1))
            # Alternatively, using diff if data is numeric:
            # gripper_change_mask = pd.to_numeric(df[gripper_col], errors='coerce').diff().fillna(0).ne(0)
            events['gripper_change'] = df_indices[gripper_change_mask].tolist()
        except Exception as e:
            print(f"Warning: Could not process 'gripper' column '{gripper_col}': {e}")

    # 4. Primary State Change (Detect START using ID to find column, NEW lever logic)
    state_change_indices = []
    if 'interface_id' in df.columns and n_points > 1:
        target_col = None
        interface_type = None
        potential_matches = []
        interface_id = None # Initialize
        try:
            # Read ID, allowing for potential NaNs or conversion errors
            first_id_val = df['interface_id'].iloc[0]
            if pd.notna(first_id_val):
                interface_id = int(first_id_val)
            else:
                print("Warning: interface_id in first row is NaN. Cannot determine target column.")

            if interface_id is not None: # Proceed only if ID was read successfully
                # Construct the regex pattern to find the relevant TCA column
                pattern = re.compile(rf"TCA_(?P<type>button|switch|lever)_{interface_id}_(?P<suffix>state|value)")

                # Search through the columns *available in the current DataFrame*
                for col_name in df.columns:
                    match = pattern.match(col_name)
                    if match:
                        potential_matches.append({
                            "col_name": col_name, "type": match.group('type'), "suffix": match.group('suffix')
                        })

                # --- Determine the target column based on matches ---
                if len(potential_matches) == 1:
                    target_col = potential_matches[0]["col_name"]
                    interface_type = potential_matches[0]["type"]
                    # print(f"  find_events: Found unique target column '{target_col}' (type: {interface_type}) for ID {interface_id}") # Less verbose
                elif len(potential_matches) == 0:
                    print(f"  find_events: Warning - No TCA column found matching pattern for ID {interface_id} in the *loaded* columns. Skipping state change.")
                else:
                    print(f"  find_events: Warning - Found multiple columns matching pattern for ID {interface_id} in the *loaded* columns: {[m['col_name'] for m in potential_matches]}. Skipping state change (ambiguous).")

        except Exception as e:
             print(f"Warning: Error during column search/ID processing for state change: {e}")
             interface_id = None # Ensure we don't proceed if error occurred

        # --- Proceed ONLY if a unique target column and type were found ---
        if target_col and interface_type:
            try:
                col_data = df[target_col]

                if interface_type == 'button' or interface_type == 'switch':
                    # --- Button/Switch Logic (0 -> 1 transition) ---
                    try:
                        # Helper functions to handle various representations of active/inactive
                        def is_active(value):
                            try: return float(value) == 1.0
                            except (ValueError, TypeError): return str(value).lower() in ['true', 'on', '1']
                        def is_inactive(value):
                            try: return float(value) == 0.0
                            except (ValueError, TypeError): return str(value).lower() in ['false', 'off', '0']

                        # Use vectorization for speed
                        current_active = col_data.apply(is_active)
                        prev_inactive = col_data.shift(1).apply(is_inactive)
                        # Find where current is active AND previous was inactive
                        # Need to handle the NaN from shift(1) at the first element
                        change_mask = (current_active & prev_inactive).fillna(False)
                        state_change_indices = df_indices[change_mask].tolist()

                        if state_change_indices:
                             print(f"  *** State Change Detected (Button/Switch ON) at index(es) {state_change_indices} for {target_col} ***")

                    except Exception as e_inner:
                         print(f"Warn: Error comparing button/switch states for {target_col}: {e_inner}")

                elif interface_type == 'lever':
                    # --- REVISED Lever Logic: Mark first movement if total range > threshold ---
                    try:
                        numeric_col_data = pd.to_numeric(col_data, errors='coerce').dropna()
                        if not numeric_col_data.empty:
                            min_val = numeric_col_data.min()
                            max_val = numeric_col_data.max()
                            total_range = max_val - min_val
                            threshold = config.LEVER_CHANGE_THRESHOLD # Get threshold from config

                            if total_range >= threshold:
                                # Find the first index where *any* movement occurs (diff > epsilon)
                                diffs = col_data.diff()
                                epsilon = 1e-6 # Small value to detect any change
                                first_move_found = False

                                # Iterate using positional index but get actual label
                                for k_pos in range(1, n_points):
                                    diff_val = diffs.iloc[k_pos] # Get diff by position
                                    if pd.notna(diff_val) and abs(diff_val) > epsilon:
                                        first_move_index_label = df_indices[k_pos] # Get corresponding label
                                        print(f"  *** State Change Detected (Lever - First Move, Range >= Thr) at index {first_move_index_label} for {target_col} ***")
                                        state_change_indices.append(first_move_index_label)
                                        first_move_found = True
                                        break # Stop after finding the first movement

                                # if not first_move_found: # Less verbose message needed?
                                #     print(f"    Lever {interface_id}: Range >= threshold, but no movement > epsilon found.")
                            # else: # Less verbose message needed?
                                # print(f"    Lever {interface_id}: Range < threshold or column not numeric.")

                    except Exception as e_inner:
                        print(f"Warn: Error processing lever range/first move for {target_col}: {e_inner}")

            except Exception as e:
                print(f"Warning: Error processing state change detection logic for column {target_col}: {e}")

    # Final assignment to the events dict
    try:
        # Ensure indices are unique and sorted
        events['state_change'] = sorted(list(set(state_change_indices)))
    except Exception as e_final:
         print(f"Error during final processing of state_change_indices: {e_final}")
         events['state_change'] = [] # Ensure it's at least an empty list

    # Final status message if no state change detected for a valid ID
    if interface_id is not None and not events['state_change'] and target_col is None:
         # This message now correctly reflects that the column wasn't found or was ambiguous
         print(f"  find_events: Final Result - No state change start detected for ID {interface_id} (column not found/ambiguous).")
    elif interface_id is not None and not events['state_change'] and target_col is not None:
         # This message indicates the column was found, but no change met the criteria
         print(f"  find_events: Final Result - No state change start detected for ID {interface_id} (no qualifying change found in {target_col}).")


    # Ensure the return value is always a standard dictionary
    return dict(events)
