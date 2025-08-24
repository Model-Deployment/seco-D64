import re
import csv

# Input and output files
input_file = 'results.txt'
results_output_file = 'results.csv'
results_summary_output_file = 'results_summary_info.csv'

# Regular expression to match image lines
image_pattern = re.compile(
    r"image\s+(\d+)/(\d+)\s+(\S+):\s+(\d+)x(\d+)\s+(\d+)\s+0[s,]?\s*,?\s*([\d.]+)ms"
)

# Patterns for summary data
speed_pattern = re.compile(
    r"Speed:\s+([\d.]+)ms preprocess,\s+([\d.]+)ms inference,\s+([\d.]+)ms postprocess"
)
results_dir_pattern = re.compile(r"Results saved to (\S+)")
inference_csv_pattern = re.compile(r"Inference completed\. Results saved to (\S+)")
tegrastats_pattern = re.compile(r"Tegrastats log saved to (\S+)")

# Storage for summary info
summary_info = {}

# Parse the file
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Write image data to CSV
with open(results_output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image_Index', 'Total_Images', 'Image_Path', 'Width', 'Height', 'Object_Count', 'Time_ms'])

    for line in lines:
        match = image_pattern.search(line)
        if match:
            writer.writerow([
                int(match.group(1)),  # Image index
                int(match.group(2)),  # Total images
                match.group(3),       # Image path
                int(match.group(4)),  # Width
                int(match.group(5)),  # Height
                int(match.group(6)),  # Object count
                float(match.group(7)) # Inference time
            ])

# Extract summary data
for line in lines:
    if speed_match := speed_pattern.search(line):
        summary_info['Preprocess_ms'] = float(speed_match.group(1))
        summary_info['Inference_ms'] = float(speed_match.group(2))
        summary_info['Postprocess_ms'] = float(speed_match.group(3))
    elif results_match := results_dir_pattern.search(line):
        summary_info['Results_Directory'] = results_match.group(1)
    elif inference_csv_match := inference_csv_pattern.search(line):
        summary_info['Inference_CSV'] = inference_csv_match.group(1)
    elif tegrastats_match := tegrastats_pattern.search(line):
        summary_info['Tegrastats_Log'] = tegrastats_match.group(1)

# Write summary data to another CSV
with open(results_summary_output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    for key, value in summary_info.items():
        writer.writerow([key, value])

print(f'Image results saved to: {results_output_file}')
print(f'Summary info saved to: {results_summary_output_file}')
