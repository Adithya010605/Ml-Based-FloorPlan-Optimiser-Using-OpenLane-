import re
import pandas as pd

# Path to the existing merged CSV file
csv_path = "/home/adi/Documents/Scripts/Python/merged_output6.csv"

# Read the existing CSV file
df = pd.read_csv(csv_path)

# List to store congestion values
congestion_values = []

# Iterate through the log files to extract congestion data
for i in range(101):
    log_path = f"/home/adi/Desktop/tools/OpenLane/designs/carryskip/runs/matrix_config_{i}/logs/routing/22-global.log"
    try:
        with open(log_path, "r") as f:
            log_content = f.read()
        
        # Adjusted regex pattern with flexible spaces
        pattern = r"Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)"
        match = re.search(pattern, log_content)
        
        if match:
            total_congestion = float(match.group(3))  # Extract total congestion percentage
        else:
            print(f"No match found in: {log_path}")
            total_congestion = None  # Assign None if no match found
    except FileNotFoundError:
        print(f"File not found: {log_path}")
        total_congestion = None
    
    congestion_values.append(total_congestion)

# Ensure the length matches the existing dataframe
while len(congestion_values) < len(df):
    congestion_values.append(None)

# Add congestion data to the dataframe
df["Congestion"] = congestion_values[:len(df)]

# Save the updated dataframe
output_path = "/home/adi/Documents/Scripts/Python/merged_output_updated6.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV saved to {output_path}")
