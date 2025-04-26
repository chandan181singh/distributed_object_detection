"""
Spatial partitioning module for distributed object detection.
"""
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SpatialPartitioner:
    """
    Class for spatially partitioning frames among workers.
    """
    def __init__(self, num_partitions, overlap=0.1):
        """
        Initialize spatial partitioner.
        
        Args:
            num_partitions (int): Number of partitions to create
            overlap (float): Overlap ratio between adjacent partitions (0-0.5)
        """
        self.num_partitions = num_partitions
        self.overlap = min(max(0.0, overlap), 0.5)  # Clip to [0, 0.5]
        
        # Compute grid dimensions (1D or 2D)
        if num_partitions <= 2:
            self.grid_x = num_partitions
            self.grid_y = 1
        else:
            # Try to make grid as square as possible
            import math
            self.grid_x = int(math.ceil(math.sqrt(num_partitions)))
            self.grid_y = int(math.ceil(num_partitions / self.grid_x))
        
        logger.info(f"Spatial partitioner initialized with {num_partitions} partitions "
                   f"in a {self.grid_x}x{self.grid_y} grid with {self.overlap:.1%} overlap")
    
    def split_frame(self, frame):
        """
        Split a frame into multiple partitions.
        
        Args:
            frame (ndarray): Input frame
            
        Returns:
            list: List of (partition, region) tuples where:
                partition: Image partition
                region: (x1, y1, x2, y2) coordinates of the partition in the original frame
        """
        if frame is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Calculate base partition size
        part_width = width / self.grid_x
        part_height = height / self.grid_y
        
        # Calculate overlap in pixels
        overlap_x = int(part_width * self.overlap)
        overlap_y = int(part_height * self.overlap)
        
        partitions = []
        count = 0
        
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                if count >= self.num_partitions:
                    break
                
                # Calculate partition coordinates with overlap
                x1 = max(0, int(x * part_width - overlap_x))
                y1 = max(0, int(y * part_height - overlap_y))
                x2 = min(width, int((x + 1) * part_width + overlap_x))
                y2 = min(height, int((y + 1) * part_height + overlap_y))
                
                # Extract partition
                partition = frame[y1:y2, x1:x2].copy()
                
                # Store partition and its region coordinates
                partitions.append((partition, (x1, y1, x2, y2)))
                count += 1
        
        return partitions
    
    def merge_detections(self, frame, partition_detections, nms_threshold=0.4):
        """
        Merge detections from multiple partitions with non-maximum suppression.
        
        Args:
            frame (ndarray): Original frame
            partition_detections (list): List of (detections, region) tuples where:
                detections: List of detection results
                region: (x1, y1, x2, y2) coordinates of the partition
            nms_threshold (float): Non-maximum suppression threshold
            
        Returns:
            list: Merged detection results
        """
        if not partition_detections:
            return []
        
        all_detections = []
        
        # Convert detections to absolute coordinates
        for detections, region in partition_detections:
            reg_x1, reg_y1, reg_x2, reg_y2 = region
            
            for detection in detections:
                if len(detection) >= 6:
                    rel_x1, rel_y1, rel_x2, rel_y2, conf, class_id = detection[:6]
                    
                    # Convert relative coordinates to absolute
                    abs_x1 = reg_x1 + rel_x1
                    abs_y1 = reg_y1 + rel_y1
                    abs_x2 = reg_x1 + rel_x2
                    abs_y2 = reg_y1 + rel_y2
                    
                    all_detections.append([abs_x1, abs_y1, abs_x2, abs_y2, conf, class_id])
        
        # Apply non-maximum suppression
        return self._apply_nms(all_detections, nms_threshold)
    
    def _apply_nms(self, detections, nms_threshold):
        """
        Apply non-maximum suppression to detections.
        
        Args:
            detections (list): List of detection results [x1, y1, x2, y2, conf, class_id]
            nms_threshold (float): NMS threshold
            
        Returns:
            list: Filtered detection results
        """
        if not detections:
            return []
        
        # Group detections by class
        class_detections = {}
        for detection in detections:
            class_id = int(detection[5])
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(detection)
        
        # Apply NMS for each class
        filtered_detections = []
        for class_id, dets in class_detections.items():
            # Extract boxes and scores
            boxes = []
            scores = []
            for det in dets:
                boxes.append([det[0], det[1], det[2], det[3]])
                scores.append(det[4])
            
            # Convert to numpy arrays
            boxes = np.array(boxes)
            scores = np.array(scores)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, nms_threshold)
            
            # Add filtered detections
            for i in indices.flatten():
                filtered_detections.append(dets[i])
        
        return filtered_detections 