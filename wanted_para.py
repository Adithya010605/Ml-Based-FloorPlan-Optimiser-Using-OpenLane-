import pandas as pd
import os
import glob

# 📍 Update this path to your folder containing CSV files
folder_path = "/home/adi/Documents/Scripts/csv"  # Change this to your actual path

# 🔍 Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 🛑 Check if CSV files exist
if not csv_files:
    print("❌ No CSV files found! Check your folder path.")
    exit()

# 📚 Define the required columns
selected_columns = [
    "DIEAREA_mm^2", "CORE_AREA","DIE_AREA", "FP_CORE_UTIL", "PL_TARGET_DENSITY",
    "power_typical_switching_uW", "Congestion", "wire_length"
]

# 🏗 Read and merge only the selected columns
df_list = []
for file in csv_files:
    try:
        df = pd.read_csv(file, usecols=selected_columns)
        df_list.append(df)
        print(f"✅ Successfully loaded: {file}")
    except Exception as e:
        print(f"⚠️ Error loading {file}: {e}")

# 🔄 Combine all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# 💾 Save the merged CSV file
output_file = os.path.join(folder_path, "merged.csv")
combined_df.to_csv(output_file, index=False)

print(f"\n🎉 Merged CSV saved as: {output_file}")
