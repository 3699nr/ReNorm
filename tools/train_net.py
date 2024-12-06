#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("./configs/ReNorm.yml") 
    
    # args.config_file
    # cfg.OUTPUT_DIR = args.output_dir
    # cfg.DATASETS.NAMES = args.train_datasets
    # cfg.DATASETS.TESTS = args.test_datasets

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.HEADS.NUM_CLASSES = [751, 11934, 1041] 

        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        res = DefaultTrainer.test(cfg, model)

        return res

    trainer = DefaultTrainer(cfg)

    if args.resume:
        Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
