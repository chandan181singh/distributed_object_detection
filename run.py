#!/usr/bin/env python3
"""
Run script for distributed object detection with a simple UI.
"""
import os
import sys
import time
import argparse
import logging
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    parser = argparse.ArgumentParser(description='Run distributed object detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--hosts', type=str, default='localhost',
                        help='Comma-separated list of hostnames')
    parser.add_argument('--processes', type=int, default=2,
                        help='Number of processes to start')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without GUI')
    
    return parser.parse_args()

def run_distributed(config_path, hosts, num_processes):
    """
    Run the distributed object detection system using MPI.
    
    Args:
        config_path (str): Path to configuration file
        hosts (str): Comma-separated list of hostnames
        num_processes (int): Number of processes to start
        
    Returns:
        int: Exit code
    """
    try:
        # Build command
        if hosts == "localhost":
            cmd = [
                'mpiexec',
                '-n', str(num_processes),
                'python', 'src/main.py',
                '--config', config_path
            ]
        else:
            # Use -hosts for distributed execution
            cmd = [
                'mpiexec',
                '-hosts', hosts,
                'python', 'src/main.py',
                '--config', config_path
            ]

        
        # Log command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run command
        result = subprocess.run(cmd)
        
        return result.returncode
    
    except Exception as e:
        logger.error(f"Error running distributed detection: {e}")
        return 1

class DistributedDetectionGUI:
    """
    GUI for running distributed object detection.
    """
    def __init__(self, root, args):
        """
        Initialize GUI.
        
        Args:
            root (tk.Tk): Root window
            args (Namespace): Command line arguments
        """
        self.root = root
        self.args = args
        
        # Set window title and size
        root.title("Distributed Object Detection")
        root.geometry("600x500")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        # Config file path
        ttk.Label(config_frame, text="Config file:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.config_path = tk.StringVar(value=args.config)
        ttk.Entry(config_frame, textvariable=self.config_path, width=40).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_config).grid(row=0, column=2, padx=5)
        
        # Hosts
        ttk.Label(config_frame, text="Hosts:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hosts = tk.StringVar(value=args.hosts)
        ttk.Entry(config_frame, textvariable=self.hosts, width=40).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5)
        
        # Number of processes
        ttk.Label(config_frame, text="Processes:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.num_processes = tk.IntVar(value=args.processes)
        ttk.Spinbox(config_frame, from_=1, to=16, textvariable=self.num_processes, width=5).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Create model section
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Model type
        ttk.Label(model_frame, text="Model type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_type = tk.StringVar(value="yolov5")
        ttk.Combobox(model_frame, textvariable=self.model_type, values=["yolov5", "yolov3", "ssd"], width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Model variant
        ttk.Label(model_frame, text="Variant:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=10)
        self.model_variant = tk.StringVar(value="s")
        self.variant_combo = ttk.Combobox(model_frame, textvariable=self.model_variant, values=["n", "s", "m", "l", "x"], width=5)
        self.variant_combo.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Download model button
        ttk.Button(model_frame, text="Download Model", command=self.download_model).grid(row=0, column=4, padx=10)
        
        # Create video source section
        video_frame = ttk.LabelFrame(main_frame, text="Video Source", padding=10)
        video_frame.pack(fill=tk.X, pady=5)
        
        # Video source type
        ttk.Label(video_frame, text="Source type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_type = tk.StringVar(value="webcam")
        ttk.Radiobutton(video_frame, text="Webcam", variable=self.source_type, value="webcam", command=self.update_source_ui).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(video_frame, text="Video file", variable=self.source_type, value="file", command=self.update_source_ui).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Webcam ID
        self.webcam_frame = ttk.Frame(video_frame)
        self.webcam_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)
        ttk.Label(self.webcam_frame, text="Camera ID:").pack(side=tk.LEFT, padx=5)
        self.camera_id = tk.IntVar(value=0)
        ttk.Spinbox(self.webcam_frame, from_=0, to=10, textvariable=self.camera_id, width=3).pack(side=tk.LEFT, padx=5)
        
        # Video file path
        self.file_frame = ttk.Frame(video_frame)
        self.file_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        ttk.Label(self.file_frame, text="Video file:").pack(side=tk.LEFT, padx=5)
        self.video_path = tk.StringVar()
        ttk.Entry(self.file_frame, textvariable=self.video_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(self.file_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        
        # Initially hide file frame
        self.file_frame.grid_remove()
        
        # Create distribution strategy section
        strategy_frame = ttk.LabelFrame(main_frame, text="Distribution Strategy", padding=10)
        strategy_frame.pack(fill=tk.X, pady=5)
        
        # Strategy type
        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.strategy = tk.StringVar(value="frame")
        ttk.Radiobutton(strategy_frame, text="Frame-by-frame", variable=self.strategy, value="frame").grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(strategy_frame, text="Spatial splitting", variable=self.strategy, value="spatial").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Load balancing
        self.load_balancing = tk.BooleanVar(value=True)
        ttk.Checkbutton(strategy_frame, text="Enable load balancing", variable=self.load_balancing).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Create action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Generate Config", command=self.generate_config, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run", command=self.run_detection, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=root.destroy, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Create output log
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar to log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Initialize UI
        self.update_source_ui()
        
        # Redirect stdout and stderr
        self.redirect_output()
    
    def redirect_output(self):
        """
        Redirect stdout and stderr to the log text widget.
        """
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
            
            def flush(self):
                pass
        
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)
    
    def update_source_ui(self):
        """
        Update UI based on source type selection.
        """
        if self.source_type.get() == "webcam":
            self.webcam_frame.grid()
            self.file_frame.grid_remove()
        else:
            self.webcam_frame.grid_remove()
            self.file_frame.grid()
    
    def browse_config(self):
        """
        Browse for configuration file.
        """
        filepath = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filepath:
            self.config_path.set(filepath)
    
    def browse_video(self):
        """
        Browse for video file.
        """
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filepath:
            self.video_path.set(filepath)
    
    def generate_config(self):
        """
        Generate configuration file based on UI settings.
        """
        try:
            import yaml
            
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path.get())
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Build model configuration
            model_type = self.model_type.get()
            if model_type == "yolov5":
                model_weights = f"models/yolov5{self.model_variant.get()}.pt"
            elif model_type == "yolov3":
                if self.model_variant.get() in ["n", "s"]:
                    model_weights = "models/yolov3-tiny.weights"
                else:
                    model_weights = "models/yolov3.weights"
            else:  # ssd
                model_weights = "models/ssd_mobilenet.pb"
            
            # Build video source configuration
            if self.source_type.get() == "webcam":
                video_source = self.camera_id.get()
            else:
                video_source = self.video_path.get()
            
            # Create configuration
            config = {
                "video": {
                    "source": video_source,
                    "width": 640,
                    "height": 480,
                    "fps": 30
                },
                "model": {
                    "type": model_type,
                    "weights": model_weights,
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.45,
                    "use_cuda": True,
                    "input_size": [640, 640]
                },
                "distribution": {
                    "strategy": self.strategy.get(),
                    "load_balancing": self.load_balancing.get(),
                    "buffer_size": 5
                },
                "display": {
                    "show_fps": True,
                    "show_detections": True,
                    "window_name": "Distributed Object Detection",
                    "font_scale": 0.5,
                    "line_thickness": 2
                },
                "mpi": {
                    "master_rank": 0,
                    "timeout_ms": 1000
                },
                "logging": {
                    "level": "INFO",
                    "save_path": "logs/"
                }
            }
            
            # Save configuration
            with open(self.config_path.get(), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {self.config_path.get()}")
            messagebox.showinfo("Success", f"Configuration saved to {self.config_path.get()}")
        
        except Exception as e:
            logger.error(f"Error generating configuration: {e}")
            messagebox.showerror("Error", f"Error generating configuration: {e}")
    
    def download_model(self):
        """
        Download model using the download_models.py script.
        """
        try:
            model_type = self.model_type.get()
            model_variant = self.model_variant.get()
            
            if model_type == "yolov5":
                model_name = f"yolov5{model_variant}"
            elif model_type == "yolov3":
                if model_variant in ["n", "s"]:
                    model_name = "yolov3-tiny"
                else:
                    model_name = "yolov3"
            else:  # ssd
                model_name = "ssd_mobilenet"
            
            # Run download script
            logger.info(f"Downloading model: {model_name}")
            
            cmd = [
                sys.executable,
                "download_models.py",
                "--model", model_name
            ]
            
            # Start download in a separate thread
            import threading
            
            def download_thread():
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    logger.info(f"Model {model_name} downloaded successfully")
                else:
                    logger.error(f"Failed to download model {model_name}")
            
            thread = threading.Thread(target=download_thread)
            thread.daemon = True
            thread.start()
        
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            messagebox.showerror("Error", f"Error downloading model: {e}")
    
    def run_detection(self):
        """
        Run distributed object detection.
        """
        try:
            # Check if configuration file exists
            if not os.path.isfile(self.config_path.get()):
                logger.warning(f"Configuration file {self.config_path.get()} does not exist")
                generate = messagebox.askyesno("Warning", f"Configuration file {self.config_path.get()} does not exist. Generate it now?")
                if generate:
                    self.generate_config()
                else:
                    return
            
            # Start detection in a separate thread
            import threading
            
            def detection_thread():
                logger.info("Starting distributed object detection...")
                run_distributed(
                    self.config_path.get(),
                    self.hosts.get(),
                    self.num_processes.get()
                )
                logger.info("Distributed object detection completed")
            
            thread = threading.Thread(target=detection_thread)
            thread.daemon = True
            thread.start()
        
        except Exception as e:
            logger.error(f"Error running detection: {e}")
            messagebox.showerror("Error", f"Error running detection: {e}")

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Run with or without GUI
    if args.no_gui:
        return run_distributed(args.config, args.hosts, args.processes)
    else:
        try:
            # Create GUI
            root = tk.Tk()
            app = DistributedDetectionGUI(root, args)
            root.mainloop()
            return 0
        
        except Exception as e:
            logger.error(f"Error in GUI: {e}")
            return 1

if __name__ == '__main__':
    sys.exit(main()) 