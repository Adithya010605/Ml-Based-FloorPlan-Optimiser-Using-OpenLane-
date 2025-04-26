import re
import pandas as pd
import os

# Function to extract values from config.tcl
def extract_config_params(config_path):
    params = {
        "FP_CORE_UTIL": None,
        "DIE_AREA": None,
        "CORE_AREA": None,
        "PL_TARGET_DENSITY": None
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            for line in file:
                for key in params.keys():
                    match = re.search(rf"set ::env\({key}\) (.+)", line)
                    if match:
                        try:
                            params[key] = float(match.group(1).strip().replace('"', ''))  # Convert to float, remove quotes
                        except ValueError:
                            params[key] = match.group(1).strip().replace('"', '')  # Remove quotes if not a number

    return params

# List to store dataframes
dataframes = []

# Iterate over matrix_config folders
for i in range(101):  # Assuming runs 0 to 82
    base_path = f"/home/adi/Desktop/tools/OpenLane/designs/carryskip/runs/matrix_config_{i}"
    report_path = f"{base_path}/report.csv"
    config_path = f"{base_path}/config_in.tcl"

    try:
        # Read report.csv
        df = pd.read_csv(report_path)

        # Extract parameters from config.tcl
        config_params = extract_config_params(config_path)

        # Add extracted parameters to the DataFrame
        for key, value in config_params.items():
            df[key] = value

        # Append to list
        dataframes.append(df)

    except FileNotFoundError:
        print(f"Warning: {report_path} not found. Skipping...")

# Merge all DataFrames
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv("merged_output6.csv", index=False)
    print(f"Successfully merged {len(dataframes)} CSV files into 'merged_output.csv'.")
else:
    print("No CSV files found. Nothing to merge.")
