#!/usr/bin/env python3
"""
Example script to run the distributed detection system locally with multiple processes.
"""
import os
import sys
import subprocess
import argparse

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run distributed detection locally')
    parser.add_argument('--processes', type=int, default=2,
                        help='Number of processes to start (must be >= 2)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    # Parse arguments
    args = parse_args()
    
    # Ensure at least 2 processes
    if args.processes < 2:
        print("Error: Number of processes must be at least 2 (1 master + 1+ workers)")
        return 1
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check if config file exists
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Run the UI first to create a configuration file or use --config to specify one")
        return 1
    
    # Build command
    cmd = [
        'mpiexec',
        '-n', str(args.processes),
        'python', os.path.join(project_root, 'src', 'main.py'),
        '--config', config_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Execute command
        result = subprocess.run(cmd)
        return result.returncode
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 0
    
    except Exception as e:
        print(f"Error running distributed detection: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 