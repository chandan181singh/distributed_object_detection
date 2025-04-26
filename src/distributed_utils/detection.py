"""
Object detection utility module for distributed detection.
"""
import os
import cv2
import time
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Base class for object detectors.
    """
    def __init__(self, config):
        """
        Initialize the detector with configuration.
        
        Args:
            config (dict): Model configuration
        """
        self.config = config
        self.model_type = config.get('type', 'yolov5')
        self.weights = config.get('weights', '')
        self.conf_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.use_cuda = config.get('use_cuda', True) and torch.cuda.is_available()
        self.input_size = config.get('input_size', [640, 640])
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.model = None
        self.class_names = []
        
        # Initialize detector
        self._initialize()
    
    def _initialize(self):
        """
        Initialize the detection model.
        """
        try:
            if self.model_type == 'yolov5':
                self._init_yolov5()
            elif self.model_type == 'yolov3':
                self._init_yolov3()
            elif self.model_type == 'ssd':
                self._init_ssd()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info(f"Object detector initialized: {self.model_type} on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
            raise
    
    def _init_yolov5(self):
        """
        Initialize YOLOv5 model.
        """
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                        path=self.weights, device=self.device)
            self.model.conf = self.conf_threshold
            self.model.iou = self.nms_threshold
            self.class_names = self.model.names
        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {e}")
            logger.info("Attempting to load YOLOv5 model directly...")
            
            # Try to load a pre-trained model if weights not found
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                            pretrained=True, device=self.device)
                self.model.conf = self.conf_threshold
                self.model.iou = self.nms_threshold
                self.class_names = self.model.names
                logger.info("Loaded pre-trained YOLOv5s model")
            except Exception as e2:
                logger.error(f"Error loading pre-trained YOLOv5 model: {e2}")
                raise
    
    def _init_yolov3(self):
        """
        Initialize YOLOv3 model (OpenCV DNN).
        """
        try:
            # Check if weights and config files exist
            weights_path = self.weights
            config_path = os.path.splitext(weights_path)[0] + '.cfg'
            
            if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
                raise FileNotFoundError(f"YOLOv3 weights or config not found: {weights_path}, {config_path}")
            
            # Load YOLO network
            self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            # Set backend and target
            if self.use_cuda:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            self.output_layers = self.model.getUnconnectedOutLayersNames()
            
            # Load class names
            classes_path = os.path.join(os.path.dirname(weights_path), 'coco.names')
            if os.path.isfile(classes_path):
                with open(classes_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                logger.warning(f"Class names file not found: {classes_path}")
                self.class_names = [f"Class {i}" for i in range(80)]  # Default COCO classes
        except Exception as e:
            logger.error(f"Error initializing YOLOv3: {e}")
            raise
    
    def _init_ssd(self):
        """
        Initialize SSD model (OpenCV DNN).
        """
        try:
            # Check if weights and config files exist
            weights_path = self.weights
            config_path = os.path.splitext(weights_path)[0] + '.pbtxt'
            
            if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
                raise FileNotFoundError(f"SSD weights or config not found: {weights_path}, {config_path}")
            
            # Load network
            self.model = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
            
            # Set backend and target
            if self.use_cuda:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load class names
            classes_path = os.path.join(os.path.dirname(weights_path), 'coco.names')
            if os.path.isfile(classes_path):
                with open(classes_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                logger.warning(f"Class names file not found: {classes_path}")
                self.class_names = [f"Class {i}" for i in range(91)]  # Default COCO classes
        except Exception as e:
            logger.error(f"Error initializing SSD: {e}")
            raise
    
    def detect(self, frame):
        """
        Perform object detection on the input frame.
        
        Args:
            frame (ndarray): Input image for detection
            
        Returns:
            tuple: (detections, inference_time)
                detections: list of detection results
                inference_time: detection time in seconds
        """
        start_time = time.time()
        
        if self.model_type == 'yolov5':
            results = self._detect_yolov5(frame)
        elif self.model_type == 'yolov3':
            results = self._detect_yolov3(frame)
        elif self.model_type == 'ssd':
            results = self._detect_ssd(frame)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inference_time = time.time() - start_time
        
        return results, inference_time
    
    def _detect_yolov5(self, frame):
        """
        Perform detection using YOLOv5.
        
        Args:
            frame (ndarray): Input image
            
        Returns:
            list: Detection results
        """
        # Convert frame to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(frame_rgb, size=self.input_size[0])
        
        # Convert results to standard format
        detections = []
        
        # Extract detection results
        for pred in results.pred:
            if len(pred) > 0:
                for *xyxy, conf, cls in pred:
                    x1, y1, x2, y2 = [x.item() for x in xyxy]
                    detections.append([x1, y1, x2, y2, conf.item(), cls.item()])
        
        return detections
    
    def _detect_yolov3(self, frame):
        """
        Perform detection using YOLOv3.
        
        Args:
            frame (ndarray): Input image
            
        Returns:
            list: Detection results
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, tuple(self.input_size), 
                                     swapRB=True, crop=False)
        
        # Set input and perform inference
        self.model.setInput(blob)
        outputs = self.model.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    # Convert YOLO coordinates to pixel coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, x + w, y + h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 
                                  self.nms_threshold)
        
        # Create detection results list
        detections = []
        for i in indices.flatten():
            detections.append([
                boxes[i][0], boxes[i][1], 
                boxes[i][2], boxes[i][3], 
                confidences[i], class_ids[i]
            ])
        
        return detections
    
    def _detect_ssd(self, frame):
        """
        Perform detection using SSD.
        
        Args:
            frame (ndarray): Input image
            
        Returns:
            list: Detection results
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, tuple(self.input_size), 
                                     [127.5, 127.5, 127.5], swapRB=True, crop=False)
        
        # Set input and perform inference
        self.model.setInput(blob)
        outputs = self.model.forward()
        
        # Create detection results list
        detections = []
        
        # Process outputs
        for i in range(outputs.shape[2]):
            confidence = outputs[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                class_id = int(outputs[0, 0, i, 1])
                
                # Get bounding box coordinates
                x1 = int(outputs[0, 0, i, 3] * width)
                y1 = int(outputs[0, 0, i, 4] * height)
                x2 = int(outputs[0, 0, i, 5] * width)
                y2 = int(outputs[0, 0, i, 6] * height)
                
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections 