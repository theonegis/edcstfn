import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)


def deconv3x3(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 3,
                              stride=2, padding=1, output_padding=1)


def gen_mask(prediction, reference1, reference2):
    diff = (torch.abs(reference1 - prediction) -
            torch.abs(reference2 - prediction))
    mask = torch.where(diff > 0,
                       diff.new_tensor(0.0),
                       diff.new_tensor(1.0))
    invert = torch.ones_like(mask) - mask
    return mask, invert


def upsample(x, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)


class CharbonnierLoss(nn.Module):
    """L1 Charbonnier Loss"""
    def __init__(self, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error) if self.reduction == 'mean' else torch.sum(error)
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.module = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.ReLU(inplace=True),
            deconv3x3(out_channels, out_channels)
        )
        self.updim = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = self.module(x)
        src = upsample(self.updim(x))
        out = src + out
        return out


class DenseBlock(nn.Module):
    def __init__(self, module, upsample=False):
        super(DenseBlock, self).__init__()
        self.module = module
        self.upsample = upsample

    def forward(self, x):
        out = self.module(x)
        if self.upsample:
            out = torch.cat((upsample(x), out), dim=1)
        else:
            out = torch.cat((x, out), dim=1)
        return out


class LNet(nn.Module):
    def __init__(self):
        super(LNet, self).__init__()
        channels = 32
        self.module = nn.Sequential(
            conv3x3(1, channels),
            nn.ReLU(inplace=True),
            DenseBlock(conv3x3(channels, channels)),
            nn.ReLU(inplace=True),
            DenseBlock(conv3x3(channels * 2, channels)),
            nn.ReLU(inplace=True),
            DenseBlock(conv3x3(channels * 3, channels)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.module(x)
        return out


class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        channels = 32
        self.model = nn.Sequential(
            ResidualBlock(1, channels),
            nn.ReLU(inplace=True),
            DenseBlock(ResidualBlock(channels, channels), upsample=True),
            nn.ReLU(inplace=True),
            DenseBlock(ResidualBlock(channels * 2, channels), upsample=True),
            nn.ReLU(inplace=True),
            DenseBlock(ResidualBlock(channels * 3, channels), upsample=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channels = [128, 64, 32, 1]
        self.model = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 1),
            nn.ReLU(inplace=True),
            conv3x3(channels[0], channels[1]),
            nn.ReLU(inplace=True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class FusionNet(nn.Module):
    def __init__(self, n_refs):
        super(FusionNet, self).__init__()
        self.refs = n_refs
        self.higher = LNet()
        self.lower = MNet()
        self.decoder = Decoder()

    def forward(self, inputs):
        if self.refs == 2:
            assert len(inputs) == 5
        elif self.refs == 1:
            assert len(inputs) == 3

        pre_coarse = self.lower(inputs[-1])

        ref_coarse_1 = self.lower(inputs[0])
        ref_fine_1 = self.higher(inputs[1])
        res_fusion_1 = ref_fine_1 - ref_coarse_1 + pre_coarse
        result = res_fusion_1

        if self.refs == 2:
            ref_coarse_2 = self.lower(inputs[2])
            ref_fine_2 = self.higher(inputs[3])
            res_fusion_2 = ref_fine_2 - ref_coarse_2 + pre_coarse
            mask, invert = gen_mask(pre_coarse, ref_coarse_1, ref_coarse_2)
            result = res_fusion_1 * mask + res_fusion_2 * invert

        out = self.decoder(result)
        return out
