# Distributed Real-Time Object Detection

A distributed system for real-time object detection using MPI, OpenCV, and CUDA across multiple laptops.

## Overview

This project parallelizes the computationally intensive task of real-time object detection by distributing video frames across multiple computers. It uses:

- MPI (Message Passing Interface) for communication between nodes
- OpenCV with CUDA acceleration for object detection
- Pre-trained YOLO models for detection

## System Architecture

- **Master Node**: Captures video, distributes frames, collects results, reconstructs output
- **Worker Nodes**: Process received frames using object detection models and return results

## Requirements

- Python 3.7+
- CUDA-capable NVIDIA GPUs on worker nodes (recommended)
- Network connection between all participating machines
- MPI implementation (e.g., MPICH, OpenMPI)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/distributed_object_detection.git
cd distributed_object_detection
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install MPI on all participating machines (if not already installed)

## Usage

1. Edit `config.yaml` to specify your setup configuration
2. Start the distributed system with MPI:
```
mpiexec -n <num_processes> -hosts <host1,host2,...> python src/main.py --config config.yaml
```

## Configuration

Edit `config.yaml` to configure:
- Video source (camera device ID or video file path)
- Detection model (YOLO version, weights)
- Frame distribution strategy
- Display options

## License

MIT 