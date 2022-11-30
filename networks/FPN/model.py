from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from networks.FPN.pvtv2 import pvt_v2_b2, pvt_v2_b0
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import segmentation_models_pytorch as smp
import timm

# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def BuildFPN(num_classes, encoder='pvtb2', decoder='fpn'):
    if encoder == 'pvtb0':
        backbone = pvt_v2_b0()
        path = 'weights/pvt_v2_b0.pth'
        chs = [32, 64, 160, 256]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == 'pvtb2':
        backbone = pvt_v2_b2()
        path = 'weights/pvt_v2_b2.pth'
        chs = [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = backbone.state_dict()
        state_dict = {
            k: v
            for k, v in save_model.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
    elif encoder == 'resnet50':
        # backbone = timm.models.resnetv2_50(pretrained=True, features_only=True)
        backbone = smp.encoders.get_encoder('resnet50')
        chs = [256, 512, 1024, 2048]
    elif encoder == 'resnet18':
        backbone = smp.encoders.get_encoder('resnet18')
        chs = [64, 128, 256, 512]
    else:
        raise NotImplementedError

    if 'resnet' in encoder:
        trans = False
    else:
        trans = True
    head = _head(num_classes, in_chs=128)
    decoder = FPNDecoder(chs)
    model = _SimpleSegmentationModel(trans, backbone, decoder, head)
    return model


class _head(nn.Module):
    def __init__(self, num_classes, in_chs):
        super(_head, self).__init__()
        self.p_head = nn.Conv2d(in_chs, num_classes, 1)

    def forward(self, feature):
        o = self.p_head(feature)
        o = F.sigmoid(o)
        return o


class _SimpleSegmentationModel(nn.Module):
    # general segmentation model
    def __init__(self, trans, backbone, decoder, head):
        super(_SimpleSegmentationModel, self).__init__()
        self.trans = trans
        self.backbone = backbone
        self.head = head
        self.decoder = decoder

    def forward(self, x, return_features=False, return_att_maps=False):
        input_shape = x.shape[-2:]
        if self.trans:
            features, maps = self.backbone(x, rt_info=return_att_maps)
        else:
            features = self.backbone(x)
        # print([f.shape for f in features])
        x = self.decoder(features[-4], features[-3], features[-2],
                         features[-1])
        x = self.head(x)
        x = F.interpolate(x,
                          size=input_shape,
                          mode='bilinear',
                          align_corners=False)
        if return_features:
            return x, features
        elif return_att_maps:
            return x, maps
        else:
            return x


if __name__ == '__main__':
    import os
    # from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = BuildFPN(1, 'resnet18', 'fpn').cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1 = model(input_tensor)

    # summary(model, (3, 352, 352))
