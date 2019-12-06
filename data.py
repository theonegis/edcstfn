import numpy as np
import rasterio
import math

import torch
from torch.utils.data import Dataset

from utils import make_tuple


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_stride = patch_size if not patch_stride else make_tuple(patch_stride)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.images = [im for im in self.root_dir.glob('*.tif')]
        self.num_of_im = len(self.images)

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_of_im * self.num_patches_x * self.num_patches_y

        self.transform = lambda im: torch.from_numpy(im).mul_(0.0001)

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        with rasterio.open(str(self.images[id_n])) as ds:
            image = ds.read()

        patch = image[:,
                id_x:(id_x + self.patch_size[0]),
                id_y:(id_y + self.patch_size[1])].astype(np.float32)
        patch = self.transform(patch)

        return patch

    def __len__(self):
        return self.num_patches
