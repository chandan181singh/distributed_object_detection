#!/usr/bin/env python3
"""
Script to download pre-trained models for distributed object detection.
"""
import os
import sys
import argparse
import logging
import requests
import torch
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Download pre-trained models')
    parser.add_argument('--model', type=str, default='yolov5s',
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
                                'yolov3', 'yolov3-tiny', 'ssd_mobilenet'],
                        help='Model to download')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for downloaded models')
    
    return parser.parse_args()

def download_file(url, output_path):
    """
    Download a file from a URL with progress reporting.
    
    Args:
        url (str): URL to download
        output_path (str): Output path for downloaded file
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {output_path}")
        
        # Get file size
        response = requests.head(url, allow_redirects=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Download the file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Report progress
                        if total_size_in_bytes > 0:
                            percent = int(100 * downloaded / total_size_in_bytes)
                            sys.stdout.write(f"\rDownloading: {percent}% [{downloaded} / {total_size_in_bytes}]")
                            sys.stdout.flush()
        
        sys.stdout.write("\n")
        logger.info(f"Downloaded {output_path} successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_yolov5(model_name, output_dir):
    """
    Download YOLOv5 model.
    
    Args:
        model_name (str): YOLOv5 model name (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
        output_dir (str): Output directory
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}.pt")
        
        # Check if already exists
        if os.path.isfile(output_path):
            logger.info(f"Model {model_name} already exists at {output_path}")
            return True
        
        # Use torch hub to download
        logger.info(f"Downloading {model_name} from torch hub")
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        
        # Save model
        model.save(output_path)
        logger.info(f"Saved {model_name} to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error downloading YOLOv5 {model_name}: {e}")
        return False

def download_yolov3(model_name, output_dir):
    """
    Download YOLOv3 model.
    
    Args:
        model_name (str): YOLOv3 model name (yolov3, yolov3-tiny)
        output_dir (str): Output directory
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Set URLs for weights, config, and names
        if model_name == 'yolov3':
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
        else:  # yolov3-tiny
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        
        # Set output paths
        weights_path = os.path.join(output_dir, f"{model_name}.weights")
        config_path = os.path.join(output_dir, f"{model_name}.cfg")
        names_path = os.path.join(output_dir, "coco.names")
        
        # Download files
        success = True
        if not os.path.isfile(weights_path):
            success &= download_file(weights_url, weights_path)
        else:
            logger.info(f"Weights file already exists at {weights_path}")
        
        if not os.path.isfile(config_path):
            success &= download_file(config_url, config_path)
        else:
            logger.info(f"Config file already exists at {config_path}")
        
        if not os.path.isfile(names_path):
            success &= download_file(names_url, names_path)
        else:
            logger.info(f"Names file already exists at {names_path}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error downloading YOLOv3 {model_name}: {e}")
        return False

def download_ssd_mobilenet(output_dir):
    """
    Download SSD MobileNet model.
    
    Args:
        output_dir (str): Output directory
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Set URLs for weights, config, and names
        model_name = "ssd_mobilenet"
        weights_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        
        # Set output paths
        tar_path = os.path.join(output_dir, f"{model_name}.tar.gz")
        extract_dir = os.path.join(output_dir, "ssd_mobilenet_v2_coco_2018_03_29")
        weights_path = os.path.join(output_dir, f"{model_name}.pb")
        config_path = os.path.join(output_dir, f"{model_name}.pbtxt")
        names_path = os.path.join(output_dir, "coco.names")
        
        # Download and extract tar file
        success = True
        if not os.path.isfile(weights_path):
            if not os.path.isfile(tar_path):
                success &= download_file(weights_url, tar_path)
            
            # Extract tar file
            import tarfile
            if success and os.path.isfile(tar_path):
                logger.info(f"Extracting {tar_path}")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=output_dir)
                
                # Move files to desired locations
                frozen_graph = os.path.join(extract_dir, "frozen_inference_graph.pb")
                if os.path.isfile(frozen_graph):
                    shutil.copy(frozen_graph, weights_path)
                    logger.info(f"Copied weights to {weights_path}")
                
                # Create pbtxt file
                with open(config_path, 'w') as f:
                    f.write(SSD_MOBILENET_PBTXT)
                logger.info(f"Created config at {config_path}")
                
                # Clean up
                if os.path.isfile(tar_path):
                    os.remove(tar_path)
                if os.path.isdir(extract_dir):
                    shutil.rmtree(extract_dir)
        else:
            logger.info(f"Weights file already exists at {weights_path}")
        
        # Download names file
        if not os.path.isfile(names_path):
            success &= download_file(names_url, names_path)
        else:
            logger.info(f"Names file already exists at {names_path}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error downloading SSD MobileNet: {e}")
        return False

# SSD MobileNet config in pbtxt format
SSD_MOBILENET_PBTXT = """
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "person"
  label: 1
  display_name: "person"
}
item {
  name: "bicycle"
  label: 2
  display_name: "bicycle"
}
item {
  name: "car"
  label: 3
  display_name: "car"
}
# ... more classes (truncated for brevity)
"""

def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download requested model
    success = False
    
    if args.model.startswith('yolov5'):
        success = download_yolov5(args.model, args.output_dir)
    
    elif args.model.startswith('yolov3'):
        success = download_yolov3(args.model, args.output_dir)
    
    elif args.model == 'ssd_mobilenet':
        success = download_ssd_mobilenet(args.output_dir)
    
    else:
        logger.error(f"Unsupported model: {args.model}")
        return 1
    
    if success:
        logger.info(f"Model {args.model} downloaded successfully")
        return 0
    else:
        logger.error(f"Failed to download model {args.model}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 