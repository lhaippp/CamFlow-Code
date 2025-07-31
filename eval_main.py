"""
Evaluation entry point for CamFlow model.
Following KISS, YAGNI and SOLID principles.
"""

import os
import torch
import logging
import argparse
from datetime import datetime

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from common.utils import Params, set_logger
from common.manager import Manager
from dataset.data_loader import fetch_dataloader
from model.net import fetch_net
from evaluators import GHOFEvaluator, IQAEvaluator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CamFlow model')
    parser.add_argument('--model_dir', 
                       default='experiments/CAHomo/',
                       help="Directory containing params.json")
    parser.add_argument('--restore_file',
                       default='experiments/CAHomo/HEM.pth',
                       help="Path to model weights")
    parser.add_argument('--only_weights',
                       action='store_true',
                       default=True,
                       help='Only use weights to load or load all train status')
    parser.add_argument('--seed',
                       type=int,
                       default=230,
                       help='Random seed')
    parser.add_argument('--enable_iqa',
                       action='store_true',
                       default=False, 
                       help='Enable IQA metrics evaluation')
    return parser.parse_args()

def setup_environment(args):
    """Setup evaluation environment."""
    # Load params
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = Params(json_path)
    params.update(vars(args))

    # Set CUDA
    params.cuda = torch.cuda.is_available()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if params.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup logging
    logger = set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    
    return params, logger

def build_model_and_manager(params, logger):
    """Build model and evaluation manager."""
    # Create dataloaders
    dataloaders = fetch_dataloader(params)
    
    # Create model
    model = fetch_net(params)
    
    # Setup DDP
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[kwargs])
    
    # Prepare model and data
    model = accelerator.prepare(model)
    for split in ['train', 'val', 'test', 'ghof']:
        if split in dataloaders:
            dataloaders[split] = accelerator.prepare(dataloaders[split])
    
    # Create manager
    manager = Manager(
        model=model,
        optimizer=None,
        scheduler=None,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
        accelerator=accelerator
    )
    
    # Load checkpoints
    manager.load_checkpoints()
    
    return manager

def main():
    """Main evaluation function."""
    # Parse arguments and setup
    args = parse_args()
    params, logger = setup_environment(args)
    
    # Build model and manager
    manager = build_model_and_manager(params, logger)
    
    # Create evaluators
    evaluators = []
    evaluators.append(GHOFEvaluator(manager))
    if args.enable_iqa:
        evaluators.append(IQAEvaluator(manager))
    
    # Store evaluators in manager for sharing results
    manager.evaluators = evaluators
    
    # Run evaluation
    if manager.accelerator.is_main_process:
        logger.info("Starting evaluation")
        
    for evaluator in evaluators:
        evaluator.evaluate()
    
    if manager.accelerator.is_main_process:
        logger.info("Evaluation complete!")

if __name__ == '__main__':
    main() 