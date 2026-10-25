import os
import torch
from torchvision import transforms
import math
import numpy as np


class ScalerNetwork(torch.nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(ScalerNetwork, self).__init__()

        layers = [torch.nn.Conv2d(1, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [torch.nn.LeakyReLU(0.2,True),]
        layers += [torch.nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [torch.nn.Sigmoid(),]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, val):
        return self.model.forward(val)


class MaskFinder(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(MaskFinder, self).__init__()

        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputChannels):
        out_net = self.netBasic(inputChannels)

        out = self.sigmoid(out_net)

        return out


class MILO(torch.nn.Module):
    def __init__(self):

        super(MILO, self).__init__()

        # Init All Components
        self.mask_finder_1 = MaskFinder(9)

        self.mask_finder_1.requires_grad = False
        self.number_of_scales = 3

        self.scaler_network = ScalerNetwork()

        model_path = os.path.abspath(os.path.join('weights', 'MILO_latent.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)


    def mask_generator(self, x, y):
        B, C, H, W = y.shape[0:4]

        refScale = [x]
        distScale = [y]

        for intLevel in range(self.number_of_scales):
            refScale.insert(0, torch.nn.functional.avg_pool2d(input=refScale[0], kernel_size=2, stride=2,
                                                              count_include_pad=False))
            distScale.insert(0, torch.nn.functional.avg_pool2d(input=distScale[0], kernel_size=2, stride=2,
                                                               count_include_pad=False))
            # end
        # end

        mask = refScale[0].new_zeros([refScale[0].shape[0], 1, int(math.floor(refScale[0].shape[2] / 2.0)),
                                      int(math.floor(refScale[0].shape[3] / 2.0))])

        for intLevel in range(len(refScale)):
            maskUpsampled = torch.nn.functional.interpolate(input=mask, scale_factor=2, mode='bilinear',
                                                            align_corners=True)

            if maskUpsampled.shape[2] != refScale[intLevel].shape[2]: maskUpsampled = torch.nn.functional.pad(
                input=maskUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if maskUpsampled.shape[3] != refScale[intLevel].shape[3]: maskUpsampled = torch.nn.functional.pad(
                input=maskUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            mask = self.mask_finder_1(
                torch.cat([refScale[intLevel], distScale[intLevel], maskUpsampled], 1)) + maskUpsampled

        return mask


    def forward(self, y, x):

        mask = self.mask_generator(x, y)
        
        score = self.scaler_network(((mask) * torch.abs(x - y)).mean([1, 2, 3], keepdim=True))  

        return score

def prepare_latent(image):
    return torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)

if __name__ == '__main__':

    import argparse
    from data import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='latents/ref.npy')
    parser.add_argument('--dist', type=str, default='latents/dist.npy')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_latent(np.load(args.ref)).to(device)
    dist = prepare_latent(np.load(args.dist)).to(device)
    
    model_milo = MILO().to(device)
    score = model_milo(dist, ref)
    score_cpu = score.detach().cpu().item()
    print('Quality Score: ' + str(score_cpu))