import argparse
from pathlib import Path
import numpy as np
from osgeo import gdal_array
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.measure import compare_psnr, compare_ssim, shannon_entropy


def sam(y_true, y_pred):
    """Spectral Angle Mapper"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    y_true_prod = np.sum(np.sqrt(y_true ** 2), axis=0)
    y_pred_prod = np.sum(np.sqrt(y_pred ** 2), axis=0)
    true_pred_prod = np.sum(y_true * y_pred, axis=0)
    ratio = true_pred_prod / (y_true_prod * y_pred_prod)
    angle = np.mean(np.arccos(ratio))
    return angle


def ergas(y_true, y_pred, scale_factor=16):
    errors = []
    for i in range(y_true.shape[0]):
        errors.append(rmse(y_true[i], y_pred[i]))
        errors[i] /= np.mean(y_pred[i])
    return 100.0 / scale_factor * sqrt(np.mean(errors))


def evaluate(y_true, y_pred, func):
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = []
    for i in range(y_true.shape[0]):
        metrics.append(func(y_true[i], y_pred[i]))
    return metrics


def mae(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: mean_absolute_error(x.ravel(), y.ravel()))


def rmse(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: sqrt(mean_squared_error(x.ravel(), y.ravel())))


def r2(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: r2_score(x.ravel(), y.ravel()))


def kge(y_true, y_pred):
    def compute(x, y):
        im_true = x.ravel()
        im_pred = y.ravel()
        r = np.corrcoef(im_true, im_pred)[1, 0]
        m_true = np.mean(im_true)
        m_pred = np.mean(im_pred)
        std_true = np.std(im_true)
        std_pred = np.std(im_pred)
        return 1 - np.sqrt((r - 1) ** 2
                           + (std_pred / std_true - 1) ** 2
                           + (m_pred / m_true - 1) ** 2)

    return evaluate(y_true, y_pred, compute)


def psnr(y_true, y_pred, data_range=10000):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_psnr(x, y, data_range=data_range))


def ssim(y_true, y_pred, data_range=10000):
    return evaluate(y_true, y_pred,
                    lambda x, y: compare_ssim(x, y, data_range=data_range))


def entropy(image):
    if image.ndim == 2:
        return shannon_entropy(image)
    if image.ndim == 3:
        entropies = []
        for i in range(image.shape[0]):
            entropies.append(shannon_entropy(image[i]))
        return entropies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 输入数据为真实数据和预测数据
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    inputs = args.inputs
    assert len(inputs) == 2

    ix = gdal_array.LoadFile(str(inputs[0].expanduser().resolve()))
    iy = gdal_array.LoadFile(str(inputs[1].expanduser().resolve()))

    scale_factor = 0.0001
    xx = ix * scale_factor
    yy = iy * scale_factor
    print('RMSE: ', *rmse(xx, yy))
    print('ERGAS: ', ergas(xx, yy))
    print('SAM: ', sam(xx, yy))
    print('SSIM: ', *ssim(ix, iy))