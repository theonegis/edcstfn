import sys
import logging
import shutil
import csv
import warnings
import rasterio

import torch


def make_tuple(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list):
        if len(x) == 1:
            return x[0], x[0]
    else:
        return x


def save_array_as_tif(matrix, path, profile=None, prototype=None):
    assert matrix.ndim == 2 or matrix.ndim == 3
    if prototype:
        with rasterio.open(str(prototype)) as src:
            profile = src.profile
    if not profile:
        warnings.warn('the geographic profile is not provided')
    with rasterio.open(path, mode='w', **profile) as dst:
        if matrix.ndim == 3:
            for i in range(matrix.shape[0]):
                dst.write(matrix[i], i + 1)
        else:
            dst.write(matrix, 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cov(x, y):
    return torch.mean((x - torch.mean(x)) * (y - torch.mean(y)))


def ssim(prediction, target, data_range=10000):
    K1 = 0.01
    K2 = 0.03
    L = data_range

    mu_x = prediction.mean()
    mu_y = target.mean()

    sig_x = prediction.std()
    sig_y = target.std()
    sig_xy = cov(target, prediction)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) /
            ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2)))


def score(prediction, target, metric):
    assert prediction.shape == target.shape
    prediction = prediction.detach() * 10000
    target = target.detach() * 10000

    if prediction.dim() == 2:
        return metric(prediction.view(-1), target.view(-1)).item()
    if prediction.dim() == 4:
        prediction = prediction.view(-1, prediction.shape[2], prediction.shape[3])
        target = prediction.view(-1, target.shape[2], target.shape[3])
    if prediction.dim() == 3:
        n_samples = prediction.shape[0]
        value = 0.0
        for i in range(n_samples):
            value += metric(prediction[i].view(-1), target[i].view(-1)).item()
        value = value / n_samples
        return value
    else:
        raise ValueError('The dimension of the inputs is not right.')


def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 文件日志
        if logpath:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        # 控制台日志
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(state, is_best, checkpoint='last.pth', best='best.pth'):
    torch.save(state, checkpoint)

    if is_best:
        shutil.copy(str(checkpoint), str(best))


def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):
    if not checkpoint.exists():
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    state = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(state['optim_dict'])

    return state


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def load_pretrained(model, pretrained, requires_grad=False):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained)['state_dict']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
