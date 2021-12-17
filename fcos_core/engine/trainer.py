# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as TT
from torchvision.utils import save_image
from collections import OrderedDict

from fcos_core.structures.image_list import to_image_list
from fcos_core.structures.bounding_box import BoxList
from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later, synchronize
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.engine.inference import inference
from fcos_core.layers import StyleLoss, ChannelSimLoss, BatchSimLoss, PixelSimLoss, TotalVariationLoss

from fcos_core.data import transforms as T

from .fda import do_fda
from .custom_utils import read_json, Hook, get_size, plot

@torch.no_grad()
def _update_teacher_model(generator_t, generator_s, predictor_t, predictor_s, keep_rate=0.996):
    '''
    PTR for teacher update
    '''
    if get_world_size() > 1:
        student_generator_dict = {
            key[7:]: value for key, value in generator_s.state_dict().items()
        }
        student_predictor_dict = {
            key[7:]: value for key, value in predictor_s.state_dict().items()
        }
    else:
        student_generator_dict = generator_s.state_dict()
        student_predictor_dict = predictor_s.state_dict()

    new_teacher_generator_dict = OrderedDict()
    new_teacher_predictor_dict = OrderedDict()
    for key, value in generator_t.state_dict().items():
        if key in student_generator_dict.keys():
            new_teacher_generator_dict[key] = (
                student_generator_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    for key, value in predictor_t.state_dict().items():
        if key in student_predictor_dict.keys():
            new_teacher_predictor_dict[key] = (
                student_predictor_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    # generator_t.load_state_dict(new_teacher_generator_dict)
    predictor_t.load_state_dict(new_teacher_predictor_dict)
    return generator_t, predictor_t

@torch.no_grad() # Requires no grad
def threshold_bbox(list_of_boxes, th=0.8, ugob=True):
    new_list_of_boxes = []
    for box_list in list_of_boxes:
        distance = 0. # init confidence
        # get field
        boxes_pre = box_list.bbox
        scores_pre = box_list.get_field("scores")
        labels_pre = box_list.get_field("labels")
        # thresholding
        valid_map = scores_pre > th
        ############### At least one possible output ###############
        if sum(valid_map) == 0:
            valid_map = scores_pre >= max(scores_pre)
            distance = th - max(scores_pre)
        ############################################################
        # indexing
        boxes = boxes_pre[valid_map, :].detach()
        scores = scores_pre[valid_map].detach()
        labels = labels_pre[valid_map].detach()
        # make new box list
        bbox = BoxList(boxes, box_list.size, box_list.mode)
        bbox.add_field("scores", scores)
        bbox.add_field("labels", labels)
        if ugob:
            uncertainty = 1 / math.exp(distance)
            bbox.add_field("uncertainty", uncertainty)
        new_list_of_boxes.append(bbox)
    return new_list_of_boxes

@torch.no_grad()
def _relaxation_student_model(predictor_t, predictor_s, channels=None):
    '''
    MTR for student update:
    '''
    if channels:
        GN_keys = {'head.cls_tower.1.weight': [],
        'head.cls_tower.1.bias': [],
        'head.cls_tower.4.weight': [],
        'head.cls_tower.4.bias': [],
        'head.cls_tower.7.weight': [],
        'head.cls_tower.7.bias': [],
        'head.cls_tower.10.weight': [],
        'head.cls_tower.10.bias': []}
        for chn in channels:
            if chn < 256:
                GN_keys['head.cls_tower.1.weight'].append(chn)
                GN_keys['head.cls_tower.1.bias'].append(chn)
            if 256 <= chn < 512:
                GN_keys['head.cls_tower.4.weight'].append(chn - 256)
                GN_keys['head.cls_tower.4.bias'].append(chn - 256)
            if 512 <= chn < 768:
                GN_keys['head.cls_tower.7.weight'].append(chn - 512)
                GN_keys['head.cls_tower.7.bias'].append(chn - 512)
            if 768 <= chn < 1024:
                GN_keys['head.cls_tower.10.weight'].append(chn - 768)
                GN_keys['head.cls_tower.10.bias'].append(chn - 768)
        if get_world_size() > 1:
            teacher_predictor_dict = {
                key[7:]: value for key, value in predictor_t.state_dict().items()
            }
        else:
            teacher_predictor_dict = predictor_t.state_dict()

        new_predictor_s_dict = OrderedDict()

        for key, value in predictor_s.state_dict().items():
            if key in GN_keys:
                new_predictor_s_dict[key] = predictor_s.state_dict()[key]
                for chn in GN_keys[key]:
                    rc = random.choice([-2,-1,0,1,2]) / 100
                    new_predictor_s_dict[key][chn] = value[chn] + (value[chn] - teacher_predictor_dict[key][chn]) * rc
            else:
                new_predictor_s_dict[key] = predictor_s.state_dict()[key]

        predictor_s.load_state_dict(new_predictor_s_dict)
    return predictor_s


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def do_train(
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
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    for k in model:
        model[k].train()
    generator_t = model["generator_t"]
    generator_s = model["generator_s"]
    predictor_t = model["predictor_t"]
    predictor_s = model["predictor_s"]
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST
    update_part = cfg.SOLVER.SFDA_UPDATE_PART

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    # For saving purpose
    ap = 0 
    ap50 = 0 
    # Flow Saver
    CSB = torch.zeros(cfg.SOLVER.FLOW_SAMPLES, 30, 4 * 256)
    ####################
    if stage == 1: # Train the network with source data
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            images = images.to(device)
            images = to_image_list(images)
            targets = [target.to(device) for target in targets]

            features = generator_s(images.tensors)
            _, loss_dict_1 = predictor_s(images, features, targets=targets)
            loss_dict_1 = {k + "_p1": loss_dict_1[k] for k in loss_dict_1}

            losses = sum(loss for loss in loss_dict_1.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced_1 = reduce_loss_dict(loss_dict_1)
            losses_reduced_1 = sum(loss for loss in loss_dict_reduced_1.values())
            meters.update(loss=losses_reduced_1, **loss_dict_reduced_1)
            
            optimizer["generator_s"].zero_grad()
            optimizer["predictor_s"].zero_grad()
            losses.backward()
            for k in optimizer:
                optimizer["generator_s"].step()
                optimizer["predictor_s"].step()

            if pytorch_1_1_0_or_later:
                scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr_G: {lr_G:.6f}",
                            "lr_P1: {lr_P1:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_G=optimizer["generator_s"].param_groups[0]["lr"],
                        lr_P1=optimizer["predictor_s"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loaders_val is not None and test_period > 0 and iteration % test_period == 0:
                synchronize()
                for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                    results, _ = inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                    )
                ap_tmp, ap50_tmp = results.results['bbox']['AP'], results.results['bbox']['AP50']
                if ap50_tmp > ap50:
                    ap50 = ap50_tmp
                    checkpointer.save("model_best_ap50", **arguments)
                    logger.info("Updating Best mAP50 Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                if ap_tmp > ap:
                    ap = ap_tmp
                    checkpointer.save("model_best_ap", **arguments)
                    logger.info("Updating Best mAP Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                synchronize()
                for k in model:
                    model[k].train()
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )

    elif stage == 2: # GAN
        
        save_period = 50 # each image every _ iters save one
        ann_file = "./annotations/KID_train.json"
        ann_file_js = read_json(ann_file)
        img_count = len(ann_file_js["images"])
        save_dir = './flows/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        cyclegan = model["cyclegan"]
        # -----------------
        #  Train Generator
        # -----------------
        cyclegan.train()
        generator_t.eval()
        predictor_t.eval()
        generator_s.eval()
        predictor_s.eval()

        id_loss = nn.L1Loss()
        tv_loss = TotalVariationLoss()
        batch_loss = BatchSimLoss()
        pixel_loss = PixelSimLoss()
        style_loss = StyleLoss()
        channel_loss = ChannelSimLoss()

        normalize_transform = T.Normalize_weak(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        norm_transform = T.Compose_weak([normalize_transform,])
        for iteration, (real_tgt_img, _, idx) in enumerate(data_loader, start_iter):
            img_name = ann_file_js["images"][idx[0]].get("file_name")
            print(img_name)
            saved_img_list = os.listdir(save_dir) # Refresh the img list every update
            if img_name in saved_img_list:
                print("Image count : {}, no need to do {} again.".format(len(saved_img_list), img_name))
                pass
            else:
                real_tgt_img = real_tgt_img[0]
                real_tgt_img_for_fda = TT.ToPILImage()(real_tgt_img).convert('RGB')
                for each_image_iter in range(cfg.SOLVER.GAN_EACH_IMAGE_ITER):
                    data_time = time.time() - end
                    each_image_iter = each_image_iter + 1
                    arguments["iteration"] = each_image_iter

                    # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
                    if not pytorch_1_1_0_or_later:
                        scheduler["cyclegan"].step()
                    # Only use 1 img because of size mismatch ...
                    real_tgt_img3d = real_tgt_img # real_tgt_img also 3d
                    # real target
                    real_tgt_img_norm = norm_transform(real_tgt_img3d)
                    real_tgt_img_norm = real_tgt_img_norm.to(device)
                    with torch.no_grad():
                        real_tgt_img_norm = to_image_list(real_tgt_img_norm)
                        featmaps_tgt_T = generator_s(real_tgt_img_norm.tensors)
                    # fake source
                    gen_src_img4d = cyclegan(real_tgt_img3d.unsqueeze(0).to(device))
                    gen_src_img3d = gen_src_img4d.squeeze(0)
                    gen_src_img_norm = norm_transform(gen_src_img3d)
                    gen_src_img_norm = to_image_list(gen_src_img_norm)
                    featmaps_src_S = generator_t(gen_src_img_norm.tensors)
                    # loss
                    # style
                    loss_style = torch.zeros([]).to(device)
                    loss_channel = torch.zeros([]).to(device)
                    loss_batch = torch.zeros([]).to(device)
                    loss_pixel = torch.zeros([]).to(device)
                    for layer_id in range(len(featmaps_tgt_T)):
                        loss_style += style_loss(featmaps_src_S[layer_id], featmaps_tgt_T[layer_id])
                        loss_channel += channel_loss(featmaps_src_S[layer_id], featmaps_tgt_T[layer_id])
                        loss_batch += batch_loss(featmaps_src_S[layer_id], featmaps_tgt_T[layer_id])
                        loss_pixel += pixel_loss(featmaps_src_S[layer_id], featmaps_tgt_T[layer_id])
                    # others
                    if gen_src_img3d.shape != real_tgt_img.shape:
                        hg, wg = gen_src_img3d.shape[-2], gen_src_img3d.shape[-1]
                        hr, wr = real_tgt_img.shape[-2], real_tgt_img.shape[-1]
                        if hg < hr:
                            real_tgt_img = real_tgt_img[:, :hg, :]
                        elif hg > hr:
                            gen_src_img3d = gen_src_img3d[:, :hr, :]
                        if wg < wr:
                            real_tgt_img = real_tgt_img[:, :, :wg]
                        elif wg > wr:
                            gen_src_img3d = gen_src_img3d[:, :, :wr]

                    loss_id = 1. * id_loss(gen_src_img3d, real_tgt_img.to(device))
                    loss_tv = tv_loss(gen_src_img3d.unsqueeze(0))[1]
                    loss_dict = {"loss_style": loss_style,
                                "loss_channel": loss_channel,
                                "loss_batch": loss_batch,
                                "loss_pixel": loss_pixel,
                                "loss_id": loss_id,
                                "loss_tv": loss_tv
                                }

                    losses = sum(loss for loss in loss_dict.values())

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters.update(loss=losses_reduced, **loss_dict_reduced)

                    optimizer["cyclegan"].zero_grad()
                    losses.backward()
                    optimizer["cyclegan"].step()

                    if pytorch_1_1_0_or_later:
                        scheduler["cyclegan"].step()


                    batch_time = time.time() - end
                    end = time.time()
                    meters.update(time=batch_time, data=data_time)

                    eta_seconds = meters.time.global_avg * (max_iter - each_image_iter)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                    if each_image_iter % 10 == 0 or each_image_iter == max_iter:
                        logger.info(
                            meters.delimiter.join(
                                [
                                    "eta: {eta}",
                                    "iter: {iter}",
                                    "{meters}",
                                    "lr_gan: {lr_gan:.6f}",
                                    "max mem: {memory:.0f}",
                                ]
                            ).format(
                                eta=eta_string,
                                iter=each_image_iter,
                                meters=str(meters),
                                lr_gan=optimizer["cyclegan"].param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )
                    if each_image_iter % save_period == 0: # Save generated images
                        styimgpath = save_dir + str(each_image_iter) + '_' + img_name
                        save_image(gen_src_img3d, styimgpath)
                        do_fda(real_tgt_img_for_fda, styimgpath)

            if len(os.listdir(save_dir)) >= img_count * cfg.SOLVER.GAN_EACH_IMAGE_ITER / save_period: # If all images have been translated
                break

    elif stage == 3: # Flow-based Sampler
        img_dir = './flows_etis/'
        img_name_list = os.listdir(img_dir)
        img_name_list = list(set([i.split('_')[1] for i in img_name_list]))
        img_name_list_ = random.sample(img_name_list, cfg.SOLVER.FLOW_SAMPLES)
        # img_name_list = ['2.png', '3.png', '4.png', ... ]
        print('Samples:', img_name_list_)
        for iteration in range(len(img_name_list_)):
            data_time = time.time() - end
            arguments["iteration"] = iteration
            img_name_base = img_name_list_[iteration]
            for i in range(30):
                img_name = str((i + 1) * 2) + '_' + str(img_name_base)
                image = Image.open(img_dir + img_name).convert('RGB')  
                print('Operating', img_name)
                size = get_size(image.size)
                image = TT.functional.resize(image, size)
                normalize_transform = T.Normalize_weak(mean=cfg.INPUT.PIXEL_MEAN, 
                                                    std=cfg.INPUT.PIXEL_STD, 
                                                    to_bgr255=cfg.INPUT.TO_BGR255)
                norm_transform = T.Compose_weak([T.ToTensor_weak(), normalize_transform,])

                hook_dic = {}
                for name, m in predictor_t.named_modules():
                    if isinstance(m, nn.GroupNorm) and 'cls_tower' in name:
                        hook_dic[name] = Hook(m, backward=True)

                image = to_image_list(norm_transform(image).unsqueeze(0).to(device))
                features_t = generator_t(image.tensors)

                boxes, _ = predictor_t(image, features_t, targets=None)
                boxes = threshold_bbox(boxes, th=cfg.SOLVER.SFDA_PSEUDO_TH, ugob=cfg.SOLVER.UGOB)
                _, loss_dict = predictor_t(image, features_t, targets=boxes)

                loss_dict = {k + "_st": loss_dict[k] for k in loss_dict}

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)
                
                # optimizer["predictor_s"].zero_grad()
                losses.backward()

                hd = dict((k, v.grad_weight) for (k, v) in hook_dic.items())
                for head_layer, (k, v) in enumerate(hd.items()):
                    stt = head_layer * 256
                    end = (head_layer + 1) * 256
                    CSB[iteration, i, stt:end] = v

                for k, hook in hook_dic.items():
                    hook.close()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)

        CSB = torch.mean(torch.var(CSB, dim = 1), dim = 0)
        tops = torch.topk(CSB, 50)
        print(tops)
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )

    else: # Strong-weak
        for iteration, (images_weak, images_strong, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                if update_part == "both":
                    scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            images_weak, images_strong = images_weak.to(device), images_strong.to(device)
            images_weak, images_strong = to_image_list(images_weak), to_image_list(images_strong)

            features_t = generator_t(images_weak.tensors)
            features_s = generator_s(images_strong.tensors)
            boxes, _ = predictor_t(images_weak, features_t, targets=None)

            # CORE FOR PSEUDO LABELING
            boxes = threshold_bbox(boxes, th=cfg.SOLVER.SFDA_PSEUDO_TH)
            
            _, loss_dict = predictor_s(images_strong, features_s, targets=boxes)
            if not cfg.SOLVER.SFDA_REG_ON:
                if "loss_reg" in loss_dict:
                    loss_dict.pop("loss_reg")
            if not cfg.SOLVER.SFDA_CTR_ON:
                if "loss_centerness" in loss_dict:
                    loss_dict.pop("loss_centerness")

            loss_dict = {k + "_st": loss_dict[k] for k in loss_dict}

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            
            if update_part == "both":
                optimizer["generator_s"].zero_grad()
            optimizer["predictor_s"].zero_grad()
            losses.backward()

            if update_part == "both":
                optimizer["generator_s"].step()
            optimizer["predictor_s"].step()

            if pytorch_1_1_0_or_later:
                if update_part == "both":
                    scheduler["generator_s"].step()
                scheduler["predictor_s"].step()

            if iteration % cfg.SOLVER.SFDA_TEACHER_UPDATE_ITER == 0:
                generator_t, predictor_t = _update_teacher_model(generator_t, 
                                                            generator_s, 
                                                            predictor_t, 
                                                            predictor_s, 
                                                            keep_rate=cfg.SOLVER.SFDA_EMA_KEEP_RATE)
            
            if cfg.SOLVER.MTR_ON:
                if iteration % cfg.SOLVER.SFDA_STUDENT_UPDATE_ITER == 0:
                    channels = [] # Change the style-sensitive channels
                    predictor_s = _relaxation_student_model(predictor_t, predictor_s, channels=channels)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr_G_s: {lr_G_s:.6f}",
                            "lr_P_s: {lr_P_s:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr_G_s=optimizer["generator_s"].param_groups[0]["lr"],
                        lr_P_s=optimizer["predictor_s"].param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if data_loaders_val is not None and test_period > 0 and iteration % test_period == 0:
                synchronize()
                for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
                    results, _ = inference(
                        model,
                        data_loader_val,
                        dataset_name=dataset_name,
                        iou_types=iou_types,
                        box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                        device=cfg.MODEL.DEVICE,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                    )
                ap_tmp, ap50_tmp = results.results['bbox']['AP'], results.results['bbox']['AP50']
                if ap50_tmp > ap50:
                    ap50 = ap50_tmp
                    checkpointer.save("model_best_ap50", **arguments)
                    logger.info("Updating Best mAP50 Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                if ap_tmp > ap:
                    ap = ap_tmp
                    checkpointer.save("model_best_ap", **arguments)
                    logger.info("Updating Best mAP Model with AP: {:.4f}, AP50: {:.4f}".format(float(ap_tmp), float(ap50_tmp)))
                synchronize()
                for k in model:
                    model[k].train()
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )