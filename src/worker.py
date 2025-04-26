"""
Worker node module for distributed object detection.

This module is responsible for:
1. Receiving frames from the master
2. Performing object detection
3. Sending results back to the master
"""
import os
import cv2
import time
import logging
import numpy as np

from distributed_utils import (
    load_config, 
    ObjectDetector, 
    MPIHelper, 
    READY, 
    WORK, 
    EXIT, 
    RESULT
)

logger = logging.getLogger(__name__)

class Worker:
    """
    Worker node class for distributed object detection.
    """
    def __init__(self):
        """
        Initialize worker node.
        """
        # Initialize MPI
        self.mpi = MPIHelper()
        if self.mpi.is_master_node():
            raise ValueError("This module should not be run on the master node")
        
        # Initialize worker state
        self.worker_rank = self.mpi.rank
        self.running = False
        self.detector = None
        
        logger.info(f"Worker node initialized with rank {self.worker_rank}")
    
    def initialize_detector(self, config):
        """
        Initialize the object detector with the given configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        try:
            model_config = config.get('model', {})
            self.detector = ObjectDetector(model_config)
            logger.info(f"Object detector initialized: {model_config.get('type', 'yolov5')}")
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
            raise
    
    def process_frame(self, frame_data):
        """
        Process a received frame.
        
        Args:
            frame_data (dict): Frame data containing frame and metadata
            
        Returns:
            dict: Detection results
        """
        if not frame_data:
            return None
        
        frame = frame_data.get('frame')
        frame_id = frame_data.get('frame_id')
        timestamp = frame_data.get('timestamp', time.time())
        
        if frame is None or frame_id is None:
            logger.warning(f"Received invalid frame data")
            return None
        
        start_time = time.time()
        
        try:
            # Perform object detection
            detections, inference_time = self.detector.detect(frame)
            
            # Draw results on frame (optional)
            result_frame = frame.copy()
            
            # Prepare result
            processing_time = time.time() - start_time
            
            result = {
                'frame_id': frame_id,
                'worker_rank': self.worker_rank,
                'timestamp': timestamp,
                'processing_time': processing_time,
                'inference_time': inference_time,
                'frame': result_frame,
                'detections': detections
            }
            
            # Update MPI stats
            self.mpi.update_stats(1, processing_time)
            
            logger.debug(f"Processed frame {frame_id} in {processing_time:.3f}s, found {len(detections)} objects")
            
            return result
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return None
    
    def run(self):
        """
        Run the worker node.
        """
        try:
            # Receive configuration from master
            config = self.mpi.broadcast(None)
            
            # Initialize detector with received configuration
            self.initialize_detector(config)
            
            # Signal ready for work
            self.mpi.send_array(None, dest=0, tag=READY)
            
            # Initialize running state
            self.running = True
            
            # Process frames until terminated
            while self.running:
                # Receive data from master
                data, source, tag = self.mpi.recv_array(source=0)
                
                if tag == WORK:
                    # Process the frame
                    result = self.process_frame(data)
                    
                    # Send result back to master
                    self.mpi.send_array(result, dest=0, tag=RESULT)
                
                elif tag == EXIT:
                    # Terminate worker
                    logger.info(f"Worker {self.worker_rank} received exit signal")
                    self.running = False
                
                elif tag == READY:
                    # Master signaling worker to be ready again
                    self.mpi.send_array(None, dest=0, tag=READY)
            
            # Log final statistics
            self.mpi.log_stats()
        
        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_rank} interrupted")
        
        finally:
            logger.info(f"Worker {self.worker_rank} terminated") 