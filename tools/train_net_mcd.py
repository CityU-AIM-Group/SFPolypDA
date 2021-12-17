# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from cyclegan_generator import GeneratorResNet

def train(cfg, local_rank, distributed):
    # model = build_detection_model(cfg)
    stage = cfg.SOLVER.SFDA_STAGE
    model = {}
    device = torch.device(cfg.MODEL.DEVICE)
    generator_t = build_backbone(cfg).to(device)
    generator_s = build_backbone(cfg).to(device)
    predictor_t = build_rpn(cfg, generator_t.out_channels).to(device)
    predictor_s = build_rpn(cfg, generator_s.out_channels).to(device)

    assert cfg.MODEL.USE_SYNCBN is False
    # if cfg.MODEL.USE_SYNCBN:
    #     assert is_pytorch_1_1_0_or_later(), \
    #         "SyncBatchNorm is only available in pytorch >= 1.1.0"
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = {}
    # We shall not optimize the teacher model via optimizer
    # optimizer["generator_t"] = make_optimizer(cfg, generator_t)
    optimizer["generator_s"] = make_optimizer(cfg, generator_s)
    # optimizer["predictor_t"] = make_optimizer(cfg, predictor_t)
    optimizer["predictor_s"] = make_optimizer(cfg, predictor_s)
    
    scheduler = {}
    # scheduler["generator_t"] = make_lr_scheduler(cfg, optimizer["generator_t"])
    scheduler["generator_s"] = make_lr_scheduler(cfg, optimizer["generator_s"])
    # scheduler["predictor_t"] = make_lr_scheduler(cfg, optimizer["predictor_t"])
    scheduler["predictor_s"] = make_lr_scheduler(cfg, optimizer["predictor_s"])

    if stage == 2:
        cyclegan = GeneratorResNet(num_colors=3).to(device)
        optimizer["cyclegan"] = torch.optim.Adam(cyclegan.parameters(), lr=3e-4)
        scheduler["cyclegan"] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer["cyclegan"], 50, 1)
        
    if distributed:
        if stage == 2:
            cyclegan = torch.nn.parallel.DistributedDataParallel(cyclegan, device_ids=[local_rank], \
                output_device=local_rank, broadcast_buffers=False)
        generator_t = torch.nn.parallel.DistributedDataParallel(generator_t, device_ids=[local_rank], \
            output_device=local_rank, broadcast_buffers=False)
        generator_s = torch.nn.parallel.DistributedDataParallel(generator_s, device_ids=[local_rank], \
            output_device=local_rank, broadcast_buffers=False)
        predictor_t = torch.nn.parallel.DistributedDataParallel(predictor_t, device_ids=[local_rank], \
            output_device=local_rank, broadcast_buffers=False)
        predictor_s = torch.nn.parallel.DistributedDataParallel(predictor_s, device_ids=[local_rank], \
            output_device=local_rank, broadcast_buffers=False)
    
    if stage == 2:
        model["cyclegan"] = cyclegan
    model["generator_t"] = generator_t
    model["generator_s"] = generator_s
    model["predictor_t"] = predictor_t
    model["predictor_s"] = predictor_s

    # for arch in model.keys():
    #     print(arch, model[arch].state_dict().keys())

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        data_loaders_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if not cfg.SOLVER.TEST_ONLY:
        if cfg.SOLVER.SFDA_ON:
            dis_func = cfg.SOLVER.SFDA_DIS_FUNC
            do_train(
                cfg,
                model,
                data_loader,
                data_loaders_val,
                optimizer,
                scheduler,
                checkpointer,
                device,
                checkpoint_period,
                test_period,
                arguments,
                stage,
                dis_func,
            )
        else:
            do_train(
                cfg,
                model,
                data_loader,
                data_loaders_val,
                optimizer,
                scheduler,
                checkpointer,
                device,
                checkpoint_period,
                test_period,
                arguments,
            )

    return model

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
