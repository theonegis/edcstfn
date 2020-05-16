from pathlib import Path
import numpy as np
import rasterio
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils import make_tuple


root_dir = Path(__file__).parents[1]
data_dir = root_dir / 'data'

REF_PREFIX_1 = '00'
PRE_PREFIX = '01'
REF_PREFIX_2 = '02'
COARSE_PREFIX = 'MOD09A1'
FINE_PREFIX = 'LC08'
SCALE_FACTOR = 16


def get_pair_path(im_dir, n_refs):
    # 将一组数据集按照规定的顺序组织好
    paths = []
    order = OrderedDict()
    order[0] = REF_PREFIX_1 + '_' + COARSE_PREFIX
    order[1] = REF_PREFIX_1 + '_' + FINE_PREFIX
    order[2] = PRE_PREFIX + '_' + COARSE_PREFIX
    order[3] = PRE_PREFIX + '_' + FINE_PREFIX

    if n_refs == 2:
        order[2] = REF_PREFIX_2 + '_' + COARSE_PREFIX
        order[3] = REF_PREFIX_2 + '_' + FINE_PREFIX
        order[4] = PRE_PREFIX + '_' + COARSE_PREFIX
        order[5] = PRE_PREFIX + '_' + FINE_PREFIX

    for prefix in order.values():
        for path in Path(im_dir).glob('*.tif'):
            if path.name.startswith(prefix):
                paths.append(path.expanduser().resolve())
                break

    if n_refs == 2:
        assert len(paths) == 6 or len(paths) == 5
    else:
        assert len(paths) == 3 or len(paths) == 4
    return paths


def load_image_pair(im_dir, n_refs):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(im_dir, n_refs)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read().astype(np.float32)  # C*H*W (numpy.ndarray)
            images.append(im)

    # 对数据的尺寸进行验证
    assert images[0].shape[1] * SCALE_FACTOR == images[1].shape[1]
    assert images[0].shape[2] * SCALE_FACTOR == images[1].shape[2]
    return images


def im2tensor(im):
    im = torch.from_numpy(im)
    out = im.mul_(0.0001)
    return out


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """
    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, n_refs=1):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        if not patch_stride:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.refs = n_refs

        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

        self.transform = im2tensor

    def map_index(self, index):
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n], self.refs)
        patches = [None] * len(images)

        scales = [1, SCALE_FACTOR]
        for i in range(len(patches)):
            scale = scales[i % 2]
            im = images[i][:,
                 id_x * scale:(id_x + self.patch_size[0]) * scale,
                 id_y * scale:(id_y + self.patch_size[1]) * scale]
            patches[i] = self.transform(im)

        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patches
