#!/usr/bin/env python3
"""
CLI Entrypoint for High-HOTA Multi-Camera Person Tracking & Re-Identification.
Usage:
    python main.py --videos videos/init/cam1.mp4 videos/init/cam2.mp4 --model yolov8m.pt --tracker bytetrack
"""

import os
import argparse
import yaml
from src.pipeline.multicam_pipeline import MultiCamTrackingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-HOTA Multi-Camera Person Tracking & Re-Identification System"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="List of video file paths to process (e.g. video1.mp4 video2.mp4)"
    )
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help="YOLO model version (yolov8n.pt, yolov8m.pt, yolov8x.pt, yolo11m.pt, yolo11x.pt)"
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="MOT tracking algorithm"
    )
    parser.add_argument(
        "--reid-thresh",
        type=float,
        default=0.25,
        help="Cosine distance threshold for cross-camera ReID identity fusion"
    )
    parser.add_argument(
        "--output-dir",
        default="videos/output",
        help="Directory to save output annotated videos"
    )
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to YAML configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config file if present, else override with CLI flags
    config_dict = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            
    # Override with CLI arguments
    if 'detector' not in config_dict:
        config_dict['detector'] = {}
    config_dict['detector']['model_name'] = args.model
    
    if 'tracker' not in config_dict:
        config_dict['tracker'] = {}
    config_dict['tracker']['tracker_type'] = args.tracker
    
    if 'reid' not in config_dict:
        config_dict['reid'] = {}
    config_dict['reid']['reid_threshold'] = args.reid_thresh

    print("=" * 70)
    print("  HIGH-HOTA MULTI-CAMERA PERSON TRACKING & RE-IDENTIFICATION  ")
    print("=" * 70)
    print(f"Input Videos   : {args.videos}")
    print(f"YOLO Model     : {args.model}")
    print(f"Tracker Engine : {args.tracker.upper()}")
    print(f"ReID Threshold : {args.reid_thresh}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 70)

    # Initialize and execute pipeline
    pipeline = MultiCamTrackingPipeline()
    pipeline.config = config_dict
    output_files = pipeline.process_videos(args.videos, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETED SUCCESSFULLY!  ")
    print("=" * 70)
    for f in output_files:
        print(f" -> Result File: {os.path.abspath(f)}")


if __name__ == "__main__":
    main()
