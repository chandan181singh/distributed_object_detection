"""
MPI utility module for distributed object detection.
"""
import time
import numpy as np
import logging
from mpi4py import MPI

logger = logging.getLogger(__name__)

# MPI message tags
READY = 0
WORK = 1
EXIT = 2
RESULT = 3

class MPIHelper:
    """
    Helper class for MPI-based distributed processing.
    """
    def __init__(self):
        """
        Initialize MPI helper.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_master = self.rank == 0
        
        # Initialize performance metrics
        self.total_frames_processed = 0
        self.total_processing_time = 0
        self.start_time = time.time()
        
        if self.is_master:
            logger.info(f"MPI initialized with {self.size} processes")
        
        # Synchronize processes
        self.comm.Barrier()
    
    def is_master_node(self):
        """
        Check if current process is the master node.
        
        Returns:
            bool: True if master, False otherwise
        """
        return self.is_master
    
    def send_array(self, array, dest, tag=WORK):
        """
        Send numpy array to destination.
        
        Args:
            array (ndarray): Numpy array to send
            dest (int): Destination rank
            tag (int): Message tag
        """
        if array is None:
            # Send empty message if array is None
            self.comm.send(None, dest=dest, tag=tag)
            return
            
        try:
            self.comm.send(array, dest=dest, tag=tag)
        except Exception as e:
            logger.error(f"Error sending array to rank {dest}: {e}")
    
    def recv_array(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        """
        Receive numpy array from source.
        
        Args:
            source (int): Source rank
            tag (int): Message tag
            
        Returns:
            tuple: (array, source, tag)
        """
        try:
            status = MPI.Status()
            data = self.comm.recv(source=source, tag=tag, status=status)
            return data, status.source, status.tag
        except Exception as e:
            logger.error(f"Error receiving array from rank {source}: {e}")
            return None, source, tag
    
    def broadcast(self, data, root=0):
        """
        Broadcast data to all processes.
        
        Args:
            data: Data to broadcast
            root (int): Root process rank
            
        Returns:
            Data broadcasted from root
        """
        try:
            return self.comm.bcast(data, root=root)
        except Exception as e:
            logger.error(f"Error broadcasting data: {e}")
            return None
    
    def gather(self, data, root=0):
        """
        Gather data from all processes.
        
        Args:
            data: Local data to gather
            root (int): Root process rank
            
        Returns:
            list: Gathered data on root, None on other processes
        """
        try:
            return self.comm.gather(data, root=root)
        except Exception as e:
            logger.error(f"Error gathering data: {e}")
            return None
    
    def barrier(self):
        """
        Synchronize all processes.
        """
        self.comm.Barrier()
    
    def update_stats(self, num_frames, processing_time):
        """
        Update performance statistics.
        
        Args:
            num_frames (int): Number of frames processed
            processing_time (float): Processing time in seconds
        """
        self.total_frames_processed += num_frames
        self.total_processing_time += processing_time
    
    def get_stats(self):
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        elapsed_time = time.time() - self.start_time
        
        avg_time_per_frame = 0
        if self.total_frames_processed > 0:
            avg_time_per_frame = self.total_processing_time / self.total_frames_processed
        
        avg_fps = 0
        if elapsed_time > 0:
            avg_fps = self.total_frames_processed / elapsed_time
        
        return {
            'total_frames': self.total_frames_processed,
            'elapsed_time': elapsed_time,
            'avg_time_per_frame': avg_time_per_frame,
            'avg_fps': avg_fps
        }
    
    def log_stats(self):
        """
        Log performance statistics.
        """
        stats = self.get_stats()
        
        logger.info(f"Rank {self.rank} - Performance Statistics:")
        logger.info(f"  Total frames processed: {stats['total_frames']}")
        logger.info(f"  Elapsed time: {stats['elapsed_time']:.2f}s")
        logger.info(f"  Average time per frame: {stats['avg_time_per_frame'] * 1000:.2f}ms")
        logger.info(f"  Average FPS: {stats['avg_fps']:.2f}")
    
    def terminate_workers(self):
        """
        Send termination signal to all worker processes.
        """
        if not self.is_master:
            logger.warning("Only the master can terminate workers")
            return
        
        for i in range(1, self.size):
            self.send_array(None, dest=i, tag=EXIT)
        
        logger.info("Termination signal sent to all workers")

class LoadBalancer:
    """
    Load balancer for distributing work across worker nodes.
    """
    def __init__(self, num_workers, strategy="frame", buffer_size=5):
        """
        Initialize load balancer.
        
        Args:
            num_workers (int): Number of worker processes
            strategy (str): Load balancing strategy ("frame", "spatial", "adaptive")
            buffer_size (int): Size of work buffer per worker
        """
        self.num_workers = num_workers
        self.strategy = strategy
        self.buffer_size = buffer_size
        self.worker_loads = [0] * (num_workers + 1)  # +1 for master
        self.worker_times = [0.0] * (num_workers + 1)
        self.worker_processing_times = [0.0] * (num_workers + 1)
        self.frame_assignments = {}  # frame_id -> worker_rank
        
        logger.info(f"Load balancer initialized with strategy: {strategy}")
    
    def get_next_worker(self):
        """
        Get the next worker for assignment based on current loads.
        
        Returns:
            int: Worker rank
        """
        if self.strategy == "frame" or self.strategy == "adaptive":
            # Find worker with lowest current load
            min_load = float('inf')
            min_worker = 1  # Start from 1 (not master)
            
            for i in range(1, self.num_workers + 1):
                if self.worker_loads[i] < min_load:
                    min_load = self.worker_loads[i]
                    min_worker = i
            
            return min_worker
        else:
            # For spatial strategy, just use round-robin
            return (self.frame_assignments.get(max(self.frame_assignments.keys(), default=0), 0) % self.num_workers) + 1
    
    def assign_frame(self, frame_id, worker_rank):
        """
        Assign a frame to a worker.
        
        Args:
            frame_id (int): Frame ID
            worker_rank (int): Worker rank
        """
        self.worker_loads[worker_rank] += 1
        self.frame_assignments[frame_id] = worker_rank
    
    def complete_frame(self, frame_id, worker_rank, processing_time):
        """
        Mark a frame as completed by a worker.
        
        Args:
            frame_id (int): Frame ID
            worker_rank (int): Worker rank
            processing_time (float): Time taken to process the frame
        """
        if worker_rank in range(1, self.num_workers + 1):
            self.worker_loads[worker_rank] = max(0, self.worker_loads[worker_rank] - 1)
            self.worker_times[worker_rank] = time.time()
            self.worker_processing_times[worker_rank] = processing_time
            
            if frame_id in self.frame_assignments:
                del self.frame_assignments[frame_id]
    
    def can_assign_more(self, worker_rank):
        """
        Check if more work can be assigned to a worker.
        
        Args:
            worker_rank (int): Worker rank
            
        Returns:
            bool: True if more work can be assigned, False otherwise
        """
        return self.worker_loads[worker_rank] < self.buffer_size
    
    def get_worker_stats(self):
        """
        Get current worker statistics.
        
        Returns:
            dict: Worker statistics
        """
        return {
            'loads': self.worker_loads[1:],
            'processing_times': self.worker_processing_times[1:]
        } 