"""
Video utility module for distributed object detection.
"""
import cv2
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VideoCapture:
    """
    Video capture class with performance tracking.
    """
    def __init__(self, source, width=None, height=None, fps=None):
        """
        Initialize video capture.
        
        Args:
            source (int or str): Camera ID or path to video file
            width (int, optional): Desired frame width
            height (int, optional): Desired frame height
            fps (int, optional): Desired capture frame rate
        """
        self.source = source
        self.cap = None
        self.width = width
        self.height = height
        self.fps = fps
        self.last_frame_time = 0
        self.frame_count = 0
        self.current_fps = 0
        
        self.initialize()
    
    def initialize(self):
        """
        Initialize video capture device.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {self.source}")
            
            # Set properties if specified
            if self.width and self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if self.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video capture initialized: {self.width}x{self.height} @ {self.fps}fps")
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            raise
    
    def read(self):
        """
        Read a frame from the video source and update FPS calculation.
        
        Returns:
            tuple: (success, frame, frame_id)
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video capture not initialized or closed")
            return False, None, None
        
        # Read frame
        ret, frame = self.cap.read()
        
        if ret:
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time:
                time_diff = current_time - self.last_frame_time
                self.current_fps = 1.0 / time_diff if time_diff > 0 else 0
            
            self.last_frame_time = current_time
            self.frame_count += 1
            
            return ret, frame, self.frame_count
        else:
            return False, None, None
    
    def get_fps(self):
        """
        Get the current processing FPS.
        
        Returns:
            float: Current FPS
        """
        return self.current_fps
    
    def release(self):
        """
        Release video capture resources.
        """
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
            self.cap = None

class FrameDisplay:
    """
    Display handler for processed frames.
    """
    def __init__(self, window_name="Output", show_fps=True, font_scale=0.5, 
                 line_thickness=2):
        """
        Initialize frame display.
        
        Args:
            window_name (str): Name of the display window
            show_fps (bool): Whether to show FPS on frame
            font_scale (float): Font scale for text
            line_thickness (int): Line thickness for drawings
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.last_update_time = 0
        self.frame_count = 0
        self.display_fps = 0
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def show(self, frame, fps=None, detections=None):
        """
        Show frame with optional FPS and detections.
        
        Args:
            frame (ndarray): Image to display
            fps (float, optional): FPS value to display
            detections (list, optional): List of detection results
        
        Returns:
            bool: True if window is still open, False if closed
        """
        display_frame = frame.copy()
        
        # Draw detections if provided
        if detections:
            self._draw_detections(display_frame, detections)
        
        # Draw FPS if enabled
        if self.show_fps and fps is not None:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0),
                        self.line_thickness)
        
        # Update FPS calculation for display
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_update_time >= 1.0:
            self.display_fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time
        
        # Show frame
        cv2.imshow(self.window_name, display_frame)
        
        # Check for key press (ESC to exit)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            return False
        
        return True
    
    def _draw_detections(self, frame, detections):
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame (ndarray): Image to draw on
            detections (list): List of detection results [x1, y1, x2, y2, confidence, class_id]
        """
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, class_id = detection[:6]
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Generate random color based on class_id
                color = (
                    int(hash(str(int(class_id))) % 256),
                    int(hash(str(int(class_id) + 1)) % 256),
                    int(hash(str(int(class_id) + 2)) % 256)
                )
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                
                # Draw label (class_id and confidence)
                label = f"Class: {int(class_id)} ({conf:.2f})"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                              self.font_scale, self.line_thickness)
                
                # Draw filled rectangle for text background
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), 
                             (x1 + text_size[0], y1), color, -1)
                
                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           self.font_scale, (255, 255, 255), self.line_thickness)
    
    def close(self):
        """
        Close display window.
        """
        cv2.destroyWindow(self.window_name)
        logger.info(f"Display window '{self.window_name}' closed") 