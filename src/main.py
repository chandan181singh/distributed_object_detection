"""
Main entry point for distributed object detection.
"""
import os
import sys
import argparse
import logging
from mpi4py import MPI

from distributed_utils import setup_logging
from master import Master
from worker import Worker

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Distributed object detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Get MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        # Handle relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, config_path)
    
    # Create and run appropriate node
    try:
        if rank == 0:
            # Master node
            logger.info(f"Starting master node with configuration from {config_path}")
            master = Master(config_path)
            master.run()
        else:
            # Worker node
            logger.info(f"Starting worker node {rank}")
            worker = Worker()
            worker.run()
    
    except KeyboardInterrupt:
        logger.info("Process interrupted")
    
    except Exception as e:
        logger.error(f"Error in process {rank}: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 