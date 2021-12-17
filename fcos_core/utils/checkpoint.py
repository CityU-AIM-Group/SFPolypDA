# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.utils.c2_model_loading import load_c2_format
from fcos_core.utils.imports import import_file
from fcos_core.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "fcos_core.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded and "generator" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["generator_t"] = self.model["generator_t"].state_dict()
        data["generator_s"] = self.model["generator_s"].state_dict()
        data["predictor_t"] = self.model["predictor_t"].state_dict()
        data["predictor_s"] = self.model["predictor_s"].state_dict()

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, load_opt_sch=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

        if load_opt_sch:
            if "generator_t" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer["generator_t"].load_state_dict(checkpoint.pop("generator_t"))
                self.optimizer["generator_s"].load_state_dict(checkpoint.pop("generator_s"))
                self.optimizer["predictor_t"].load_state_dict(checkpoint.pop("predictor_t"))
                self.optimizer["predictor_s"].load_state_dict(checkpoint.pop("predictor_s"))
            else:
                self.logger.info(
                    "No optimizer found in the checkpoint. Initializing model from scratch"
                )

            if "generator_t" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler["generator_t"].load_state_dict(checkpoint.pop("generator_t"))
                self.scheduler["generator_s"].load_state_dict(checkpoint.pop("generator_s"))
                self.scheduler["predictor_t"].load_state_dict(checkpoint.pop("predictor_t"))
                self.scheduler["predictor_s"].load_state_dict(checkpoint.pop("predictor_s"))
            else:
                self.logger.info(
                    "No scheduler found in the checkpoint. Initializing model from scratch"
                )

        return checkpoint

    def _load_model(self, checkpoint):
        if checkpoint.get("generator"):
            self.logger.info("Loading the source only model ==> ")
            # load checkpoint of our model
            load_state_dict(self.model["generator_t"], checkpoint["generator"])
            load_state_dict(self.model["generator_s"], checkpoint["generator"])
            load_state_dict(self.model["predictor_t"], checkpoint["predictor1"])
            load_state_dict(self.model["predictor_s"], checkpoint["predictor1"])
        elif checkpoint["model"].get("generator_t"):
            load_state_dict(self.model["generator_t"], checkpoint["model"].pop("generator_t"))
            load_state_dict(self.model["generator_s"], checkpoint["model"].pop("generator_s"))
            load_state_dict(self.model["predictor_t"], checkpoint["model"].pop("predictor_t"))
            load_state_dict(self.model["predictor_s"], checkpoint["model"].pop("predictor_s"))
        else:
            self.logger.info("Loading ImageNet pretrained model")
            # load others, e.g., Imagenet pretrained pkl
            load_state_dict(self.model["generator_s"], checkpoint.pop("model"))
