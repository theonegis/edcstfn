import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
from data import PatchSet, get_pair_path, SCALE_FACTOR
import utils

from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import sys


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.resolution_scale = SCALE_FACTOR
        self.image_size = option.image_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.checkpoint = self.train_dir / 'last.pth'
        self.best = self.train_dir / 'best.pth'

        self.logger = utils.get_logger()
        self.logger.info('Model initialization')

        self.model = FusionNet().to(self.device)
        self.pretrained = Pretrained().to(self.device)
        utils.load_pretrained(self.pretrained, option.pretrained)
        if option.cuda and option.ngpu > 1:
            device_ids = [i for i in range(option.ngpu)]
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.pretrained = nn.DataParallel(self.pretrained, device_ids=device_ids)

        self.criterion = CompoundLoss(self.pretrained)
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.lr, weight_decay=1e-6)

        self.logger.info(str(self.model))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters.')

    def train_on_epoch(self, n_epoch, data_loader):
        self.model.train()
        epoch_loss = utils.AverageMeter()
        epoch_error = utils.AverageMeter()
        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]

            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = (0.5 * (self.criterion(predictions[0], target) +
                           self.criterion(predictions[1], target))
                    if len(predictions) == 2 else self.criterion(predictions, target))
            epoch_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                score = (0.5 * (F.mse_loss(predictions[0], target) +
                                F.mse_loss(predictions[1], target))
                         if len(predictions) == 2 else F.mse_loss(predictions, target))
            epoch_error.update(score.item())
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'Loss: {loss.item():.10f} - '
                             f'MSE: {score.item():.5f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        return epoch_loss.avg, epoch_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.model.eval()
        epoch_loss = utils.AverageMeter()
        epoch_error = utils.AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.model(inputs)
            loss = self.criterion(prediction, target)
            epoch_loss.update(loss.item())
            score = F.mse_loss(prediction, target)
            epoch_error.update(score.item())
        utils.save_checkpoint(self.model, self.optimizer, self.checkpoint)
        return epoch_loss.avg, epoch_error.avg

    def train(self, train_dir, val_dir, patch_size, patch_stride, batch_size,
              train_refs, num_workers=0, epochs=30, resume=True):
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, patch_size, patch_stride,
                             n_refs=train_refs)
        val_set = PatchSet(val_dir, self.image_size, patch_size, n_refs=train_refs)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        least_error = sys.maxsize
        start_epoch = 0
        if resume and self.checkpoint.exists():
            utils.load_checkpoint(self.checkpoint, self.model, self.optimizer)
            if self.history.exists():
                df = pd.read_csv(self.history)
                least_error = df['val_error'].min()
                start_epoch = int(df.iloc[-1]['epoch']) + 1

        self.logger.info('Training...')
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        for epoch in range(start_epoch, epochs + start_epoch):
            for param_group in self.optimizer.param_groups:
                self.logger.info(f"Current learning rate: {param_group['lr']}")

            train_loss, train_error = self.train_on_epoch(epoch, train_loader)
            val_loss, val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_loss', 'val_error']
            csv_values = [epoch, train_loss, train_error, val_loss, val_error]
            utils.log_csv(self.history, csv_values, header=csv_header)
            scheduler.step(val_loss)
            if val_error < least_error:
                shutil.copy(str(self.checkpoint), str(self.best))
                least_error = val_error

    @torch.no_grad()
    def test(self, test_dir, patch_size, test_refs, num_workers=0):
        self.model.eval()
        patch_size = utils.make_tuple(patch_size)
        utils.load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        # 记录测试文件夹中的文件路径，用于最后投影信息的匹配
        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_paths = [get_pair_path(d, test_refs) for d in image_dirs]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        test_set = PatchSet(test_dir, self.image_size, patch_size, n_refs=test_refs)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scaled_patch_size = tuple(i * self.resolution_scale for i in patch_size)
        scaled_image_size = tuple(i * self.resolution_scale for i in self.image_size)
        pixel_value_scale = 10000
        im_count = 0
        patches = []
        t_start = datetime.now()
        for inputs in test_loader:
            # 如果包含了target数据，则去掉最后的target
            if len(inputs) % 2 == 0:
                del inputs[-1]
            name = image_paths[im_count][-1].name
            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {name}')

            # 分块进行预测（每次进入深度网络的都是影像中的一块）
            inputs = [im.to(self.device) for im in inputs]
            prediction = self.model(inputs)
            prediction = prediction.cpu().numpy()
            patches.append(prediction * pixel_value_scale)

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *scaled_image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * scaled_patch_size[1]
                    for j in range(cols):
                        col_start = j * scaled_patch_size[0]
                        result[:,
                        col_start: col_start + scaled_patch_size[0],
                        row_start: row_start + scaled_patch_size[1]
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果
                result = result.astype(np.int16)
                prototype = str(image_paths[im_count][1])
                utils.save_array_as_tif(result, self.test_dir / name, prototype=prototype)
                im_count += 1
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')
