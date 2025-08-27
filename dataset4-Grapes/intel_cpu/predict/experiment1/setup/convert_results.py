import csv
import re

# Read the file
with open("results.txt", "r") as file:
    lines = file.readlines()

# Initialize values
speed_data = {}
performance_data = {}

# Patterns
speed_pattern = re.compile(
    r"Speed: ([\d.]+)ms preprocess, ([\d.]+)ms inference, ([\d.]+)ms loss, ([\d.]+)ms postprocess"
)
perf_pattern = re.compile(
    r"\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

# Extract info
for line in lines:
    # Match speed info
    speed_match = speed_pattern.search(line)
    if speed_match:
        speed_data = {
            "Preprocess (ms)": speed_match.group(1),
            "Inference (ms)": speed_match.group(2),
            "Loss (ms)": speed_match.group(3),
            "Postprocess (ms)": speed_match.group(4),
        }

    # Match performance summary
    perf_match = perf_pattern.match(line)
    if perf_match:
        performance_data = {
            "Class": "all",
            "Images": perf_match.group(1),
            "Instances": perf_match.group(2),
            "Box(P)": perf_match.group(3),
            "Recall": perf_match.group(4),
            "mAP50": perf_match.group(5),
            "mAP50-95": perf_match.group(6),
        }

# Combine and write to CSV
if performance_data and speed_data:
    # Merge dictionaries
    combined_data = {**performance_data, **speed_data}

    # Write to CSV
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=combined_data.keys())
        writer.writeheader()
        writer.writerow(combined_data)

    print("CSV saved as results.csv")
else:
    print("Required data not found in results.txt")
