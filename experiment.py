import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import AutoEncoder, NUM_BANDS, VisionLoss
from data import PatchSet
import utils

from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import math


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.image_size = utils.make_tuple(option.image_size)

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

        self.model = AutoEncoder().to(self.device)
        if option.cuda and option.ngpu > 1:
            self.model = nn.DataParallel(self.model,
                                         device_ids=[i for i in range(option.ngpu)])

        self.criterion = VisionLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.lr, weight_decay=1e-6)

        self.logger.info(str(self.model))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters.')

    def train_on_epoch(self, n_epoch, data_loader):
        self.model.train()
        epoch_loss = utils.AverageMeter()
        epoch_score = utils.AverageMeter()
        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = data.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.criterion(predictions, data)
            epoch_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                score = F.mse_loss(predictions, data)
            epoch_score.update(score.item())
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'Loss: {loss.item():.10f} - '
                             f'MSE: {score.item():.5f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        return epoch_loss.avg, epoch_score.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.model.eval()
        epoch_loss = utils.AverageMeter()
        epoch_error = utils.AverageMeter()
        for data in data_loader:
            data = data.to(self.device)
            prediction = self.model(data)
            loss = self.criterion(prediction, data)
            epoch_loss.update(loss.item())
            score = F.mse_loss(prediction, data)
            epoch_error.update(score.item())
        utils.save_checkpoint(self.model, self.optimizer, self.checkpoint)
        return epoch_loss.avg, epoch_error.avg

    def train(self, train_dir, patch_size, patch_stride, batch_size,
              num_workers=0, epochs=30, resume=True):
        self.logger.info('Loading data...')
        dataset = PatchSet(train_dir, self.image_size, patch_size, patch_stride)
        lengths = (math.floor(len(dataset) * 0.8), math.ceil(len(dataset) * 0.2))
        train_set, val_set = torch.utils.data.random_split(dataset, lengths)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
        least_error = 0
        start_epoch = 0
        if resume and self.checkpoint.exists():
            utils.load_checkpoint(self.checkpoint, model=self.model, optimizer=self.optimizer)
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
            val_loss, val_error = self.test_on_epoch(epoch, val_loader, least_error)
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_error', 'val_error']
            csv_values = [epoch, train_loss, train_error, val_loss, val_error]
            utils.log_csv(self.history, csv_values, header=csv_header)
            scheduler.step(val_loss)
            if val_error <= least_error:
                shutil.copy(self.checkpoint, self.best)
                least_error = val_error

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        self.model.eval()
        patch_size = utils.make_tuple(patch_size)
        utils.load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        # 记录测试文件夹中的文件路径，用于最后投影信息的匹配
        images = [im for im in test_dir.glob('*.tif')]

        # 在预测阶段，对图像进行切块的时候必须刚好裁切完全，这样才能在预测结束后进行完整的拼接
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        test_set = PatchSet(test_dir, self.image_size, patch_size)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        scale_factor = 10000
        im_count = 0
        patches = []
        t_start = datetime.now()
        for data in test_loader:
            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {images[im_count].name}')

            # 分块进行预测（每次进入深度网络的都是影像中的一块）
            data = data.to(self.device)
            prediction = self.model(data)
            prediction = prediction.cpu().numpy()
            patches.append(prediction * scale_factor)

            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *self.image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                        col_start: col_start + patch_size[0],
                        row_start: row_start + patch_size[1]
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果
                result = result.astype(np.int16)
                utils.save_array_as_tif(result, self.test_dir / images[im_count].name,
                                        prototype=str(images[im_count]))
                im_count += 1
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')
