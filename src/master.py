"""
Master node module for distributed object detection.

This module is responsible for:
1. Capturing video frames
2. Distributing frames to worker nodes
3. Collecting and displaying results
"""
import os
import cv2
import time
import logging
import numpy as np
from collections import deque
from mpi4py import MPI

from distributed_utils import (
    load_config, 
    VideoCapture, 
    FrameDisplay, 
    MPIHelper, 
    LoadBalancer,
    READY, 
    WORK, 
    EXIT, 
    RESULT
)

logger = logging.getLogger(__name__)

class Master:
    """
    Master node class for distributed object detection.
    """
    def __init__(self, config_path):
        """
        Initialize master node.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize MPI
        self.mpi = MPIHelper()
        if not self.mpi.is_master_node():
            raise ValueError("This module should only be run on the master node")
        
        # Initialize video capture
        video_config = self.config.get('video', {})
        self.video_source = video_config.get('source', 0)
        self.video_width = video_config.get('width', 640)
        self.video_height = video_config.get('height', 480)
        self.video_fps = video_config.get('fps', 30)
        
        self.video_capture = None  # Will be initialized in run()
        
        # Initialize display
        display_config = self.config.get('display', {})
        self.show_fps = display_config.get('show_fps', True)
        self.show_detections = display_config.get('show_detections', True)
        self.window_name = display_config.get('window_name', 'Distributed Object Detection')
        self.font_scale = display_config.get('font_scale', 0.5)
        self.line_thickness = display_config.get('line_thickness', 2)
        
        self.display = None  # Will be initialized in run()
        
        # Initialize load balancer
        dist_config = self.config.get('distribution', {})
        self.strategy = dist_config.get('strategy', 'frame')
        self.load_balancing = dist_config.get('load_balancing', True)
        self.buffer_size = dist_config.get('buffer_size', 5)
        
        self.load_balancer = LoadBalancer(
            num_workers=self.mpi.size - 1,
            strategy=self.strategy,
            buffer_size=self.buffer_size
        )
        
        # Initialize result buffer for ordered display
        self.result_buffer = {}
        self.next_frame_to_show = 1
        self.max_out_of_order_frames = 10
        
        # Performance tracking
        self.start_time = time.time()
        self.frames_processed = 0
        self.running = False
        
        logger.info(f"Master node initialized with {self.mpi.size - 1} workers")
    
    def initialize_video(self):
        """
        Initialize video capture.
        """
        try:
            self.video_capture = VideoCapture(
                source=self.video_source,
                width=self.video_width,
                height=self.video_height,
                fps=self.video_fps
            )
            logger.info(f"Video capture initialized: {self.video_width}x{self.video_height} @ {self.video_fps}fps")
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            raise
    
    def initialize_display(self):
        """
        Initialize frame display.
        """
        try:
            self.display = FrameDisplay(
                window_name=self.window_name,
                show_fps=self.show_fps,
                font_scale=self.font_scale,
                line_thickness=self.line_thickness
            )
            logger.info(f"Display initialized: {self.window_name}")
        except Exception as e:
            logger.error(f"Error initializing display: {e}")
            raise
    
    def distribute_frame(self, frame, frame_id, worker_rank):
        """
        Distribute a frame to a worker.
        
        Args:
            frame (ndarray): Frame to distribute
            frame_id (int): Frame ID
            worker_rank (int): Worker rank
            
        Returns:
            bool: True if distribution succeeded, False otherwise
        """
        try:
            # Create a message with frame metadata
            message = {
                'frame': frame,
                'frame_id': frame_id,
                'timestamp': time.time()
            }
            
            # Send the frame to the worker
            self.mpi.send_array(message, dest=worker_rank, tag=WORK)
            
            # Update load balancer
            self.load_balancer.assign_frame(frame_id, worker_rank)
            
            return True
        except Exception as e:
            logger.error(f"Error distributing frame {frame_id} to worker {worker_rank}: {e}")
            return False
    
    def process_result(self, result):
        """
        Process a result from a worker.
        
        Args:
            result (dict): Result data
        """
        if not result:
            return
        
        frame_id = result.get('frame_id')
        worker_rank = result.get('worker_rank')
        processing_time = result.get('processing_time', 0)
        frame = result.get('frame')
        detections = result.get('detections', [])
        
        if frame_id is None or worker_rank is None or frame is None:
            logger.warning(f"Received invalid result from worker {worker_rank}")
            return
        
        # Add the processed frame to the result buffer
        self.result_buffer[frame_id] = (frame, detections)
        
        # Update load balancer with completed work
        self.load_balancer.complete_frame(frame_id, worker_rank, processing_time)
        
        # Display results if possible
        self.display_results()
    
    def display_results(self):
        """
        Display results in order.
        """
        # Display frames in order
        while self.next_frame_to_show in self.result_buffer:
            frame, detections = self.result_buffer[self.next_frame_to_show]
            
            if self.display:
                is_open = self.display.show(
                    frame=frame,
                    fps=self.video_capture.get_fps() if self.video_capture else 0,
                    detections=detections if self.show_detections else None
                )
                
                if not is_open:
                    self.running = False
                    break
            
            # Remove displayed frame from buffer
            del self.result_buffer[self.next_frame_to_show]
            
            # Update statistics
            self.frames_processed += 1
            
            # Move to next frame
            self.next_frame_to_show += 1
        
        # Check if buffer has too many frames waiting (skip frames if necessary)
        if len(self.result_buffer) > self.max_out_of_order_frames:
            # Find the lowest frame ID in the buffer
            lowest_frame_id = min(self.result_buffer.keys())
            
            if lowest_frame_id > self.next_frame_to_show:
                logger.warning(f"Skipping frames {self.next_frame_to_show}-{lowest_frame_id-1}")
                self.next_frame_to_show = lowest_frame_id
    
    def run(self):
        """
        Run the master node.
        """
        try:
            # Initialize video and display
            self.initialize_video()
            self.initialize_display()
            
            # Broadcast configuration to all workers
            self.mpi.broadcast(self.config)
            
            # Initialize performance metrics
            self.start_time = time.time()
            self.frames_processed = 0
            self.running = True
            
            # Process frames until interrupted
            while self.running:
                # Check for worker status messages
                worker_data, worker_rank, tag = self.mpi.recv_array(tag=MPI.ANY_TAG)
                
                if tag == READY:
                    # Worker is ready for a new frame
                    if self.load_balancer.can_assign_more(worker_rank):
                        # Read a new frame
                        ret, frame, frame_id = self.video_capture.read()
                        
                        if ret:
                            # Distribute the frame to the worker
                            self.distribute_frame(frame, frame_id, worker_rank)
                        else:
                            # End of video or error
                            logger.info("End of video stream or error reading frame")
                            self.running = False
                            break
                    else:
                        # Worker has enough work, send ready again later
                        time.sleep(0.01)
                        self.mpi.send_array(None, dest=worker_rank, tag=READY)
                
                elif tag == RESULT:
                    # Process the result from a worker
                    self.process_result(worker_data)
                    
                    # Signal worker ready for more work
                    self.mpi.send_array(None, dest=worker_rank, tag=READY)
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.running = False
            
            # Log final statistics
            self.mpi.log_stats()
            stats = self.load_balancer.get_worker_stats()
            logger.info(f"Worker loads: {stats['loads']}")
            logger.info(f"Worker processing times: {stats['processing_times']}")
            
            # Terminate all workers
            self.mpi.terminate_workers()
        
        except KeyboardInterrupt:
            logger.info("Master node interrupted")
        
        finally:
            # Clean up resources
            if self.video_capture:
                self.video_capture.release()
            
            if self.display:
                self.display.close()
            
            cv2.destroyAllWindows()
            logger.info("Master node terminated") 