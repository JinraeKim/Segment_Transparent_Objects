#!/usr/bin/env python3

from __future__ import print_function

import time
import os
# import sys

# # for the path; you can remove the following and set `PYTHONPATH`,
# e.g., `PYTHONPATH="." python ./tools/test_ros.py`.
# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
# from segmentron.utils.options import parse_args
from segmentron.utils.options import parse_args_ros
from segmentron.utils.default_setup import default_setup
from segmentron.utils.filesystem import makedirs
import cv2


class Evaluator(object):
    def __init__(self):
        # default setup
        args = parse_args_ros()
        cfg.update_from_file(args.config_file)
        cfg.update_from_list(args.opts)
        cfg.PHASE = 'test'
        # cfg.ROOT_PATH = root_path
        cfg.DATASET.NAME = 'trans10k_extra_ros'
        cfg.check_and_freeze()
        default_setup(args)
        #
        self.args = args
        self.device = torch.device(self.args.device)
        # image transform
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
                logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
                self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if self.args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        self.model.to(self.device)
        self.count_easy = 0
        self.count_hard = 0

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def prepare_dataset(self):
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            cfg.DATASET.NAME,
            root=cfg.DEMO_DIR,
            split='val',
            mode='val',
            transform=self.input_transform,
            base_size=cfg.TRAIN.BASE_SIZE,
        )
        import pdb; pdb.set_trace()

        val_sampler = make_data_sampler(
            val_dataset,
            shuffle=False,
            distributed=self.args.distributed,
        )
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_sampler=val_batch_sampler,
                                     num_workers=cfg.DATASET.WORKERS,
                                     pin_memory=True)
        # self.classes = val_dataset.classes  # TODO: idk what it is so just commented it
        return val_loader

    def eval(self):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        # dataloader
        val_loader = self.prepare_dataset()
        for i, (image, _, filename) in enumerate(val_loader):
            image = image.to(self.device)
            filename = filename[0]
            save_name = os.path.basename(filename).replace('.jpg', '').replace('.png', '')

            with torch.no_grad():
                t0 = time.time()
                output, output_boundary = model.evaluate(image)
                ori_img = cv2.imread(filename)
                h, w, _ = ori_img.shape

                glass_res = output.argmax(1)[0].data.cpu().numpy().astype('uint8') * 127
                boundary_res = output_boundary[0, 0].data.cpu().numpy().astype('uint8') * 255
                glass_res = cv2.resize(glass_res, (w, h), interpolation=cv2.INTER_NEAREST)
                boundary_res = cv2.resize(boundary_res, (w, h), interpolation=cv2.INTER_NEAREST)
                t1 = time.time()
                print(f"Elapsed time: {t1-t0} s")

                save_path = os.path.join('/'.join(cfg.DEMO_DIR.split('/')[:-2]), 'result')
                makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, '{}_glass.png'.format(save_name)), glass_res)
                cv2.imwrite(os.path.join(save_path, '{}_boundary.png'.format(save_name)), boundary_res)
                print('save {}'.format(save_name))


if __name__ == '__main__':
    # Usage
    # At the root path, `PYTHONPATH="." python ./tools/test_ros.py`
    evaluator = Evaluator()
    evaluator.eval()
