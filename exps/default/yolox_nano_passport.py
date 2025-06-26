#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        self.exp_name = "yolox_nano_passport_500_epoch"
        # Optional config
        self.occupy = False
        
        # Model scale
        self.depth = 0.33
        self.width = 0.25

        # Image size
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (10, 20)

        # Augmentation
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        
        # Dataset config
        self.class_names = ("full_passport", "mrz", "passport_number")
        self.data_dir = "datasets/your_dataset"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.num_classes = 3

        # Pre-train config
        self.ckpt = "weights/yolox_nano.pth"
        
        # Pre-train config
        self.mosaic_prob = 0.3
        self.mixup_prob = 0.0
        self.degrees = 5.0
        self.translate = 0.05
        self.scale = (0.9, 1.1)
        self.shear = 0.5
        self.enable_mixup = True
        
        # Training config
        self.max_epoch = 1000
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.print_interval = 20
        self.data_num_workers = 4

        # Output dir
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
