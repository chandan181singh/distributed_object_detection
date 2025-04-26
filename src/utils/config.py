"""
Configuration utility module for distributed object detection.
"""
import os
import yaml
import logging

def setup_logging(level_name, log_path=None):
    """
    Setup logging configuration.
    
    Args:
        level_name (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_path (str, optional): Path to save log file
        
    Returns:
        logger: Configured logger object
    """
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to the logger
    logger.addHandler(console_handler)
    
    # Add file handler if log_path is provided
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup logging
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_path = config.get('logging', {}).get('save_path', None)
        if log_path:
            log_path = os.path.join(log_path, 'detection.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        logger = setup_logging(log_level, log_path)
        logger.info(f"Configuration loaded successfully from {config_path}")
        
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise 