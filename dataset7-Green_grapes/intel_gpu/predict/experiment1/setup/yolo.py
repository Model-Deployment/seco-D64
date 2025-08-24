# Libraries
import torch
import intel_extension_for_pytorch as ipex
from ultralytics import YOLO
import argparse
import pyRAPL
import time
import psutil
import os
import numpy as np
import cv2
# Initialize pyRAPL
pyRAPL.setup()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='Evaluate the model')
parser.add_argument('--no-eval', dest='eval', action='store_false', help='Run prediction and benchmark')
parser.set_defaults(eval=True)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--device', type=str, default="intel:gpu", help='Device to use')
parser.add_argument('--dataset', type=str, default='Green_grapes', help='Dataset to use')
args = parser.parse_args()

# Load pretrained OpenVINO YOLO model
model = YOLO('global_model_openvino_model/', task='detect')

if args.eval:
    meter = pyRAPL.Measurement('evaluation')
    meter.begin()

    results = model.val(
        data='data_yolo.yaml',
        split='test',
        batch=args.batch_size,
      #  save=True,
      #  save_txt=True,
        half=True,  
        device=args.device
    )
    meter.end()

    map_50 = results.box.map50
    map_50_95 = results.box.map

    with open("evaluation_results_Yolo12s.csv", "w") as f:
        f.write("Metric,Value\n")
        f.write(f"mAP@50,{map_50:.4f}\n")
        f.write(f"mAP@50-95,{map_50_95:.4f}\n")
        f.write(f"Energy Package 0 (uJ),{meter.result.pkg[0]}\n")

    print("Evaluation complete. Results saved to evaluation_results_Yolo12s.csv")

else:
    meter = pyRAPL.Measurement('inference')
    start_time = time.time()
    meter.begin()
# Measure the cpu usage
    cpu_before = psutil.cpu_percent(interval=None)
# Measure the memory usage

    mem_before = psutil.virtual_memory().used / 1024**2

    results = model.predict(
        source=f"Datasets/{args.dataset}/test/images",
        save=True,
       # save_txt=True,
        device=args.device
    )

    meter.end()
    end_time = time.time()
    total_time = end_time - start_time
    num_images = len(results)
    avg_inference_time = total_time / num_images if num_images else 0

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used / 1024**2
# Save the data to a .csv file 
    with open("inference_benchmark_results.csv", "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Total Time (End-to-End) (s),{total_time:.2f}\n")
        f.write(f"Average Inference Time (End-to-End) (s/image),{avg_inference_time:.4f}\n")
        f.write(f"CPU Usage Before (%),{cpu_before}\n")
        f.write(f"CPU Usage After (%),{cpu_after}\n")
        f.write(f"Memory Usage Before (MB),{mem_before:.2f}\n")
        f.write(f"Memory Usage After (MB),{mem_after:.2f}\n")
        f.write(f"Energy Package 0 (uJ),{meter.result.pkg[0]}\n")

    print("Inference complete. Results saved to inference_benchmark_results.csv")
