import os
import torch
from torchvision import transforms
import math


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
        self.cuda()

        self.mask_finder_1 = MaskFinder(7).cuda()

        self.mask_finder_1.requires_grad = False
        self.number_of_scales = 3

        self.scaler_network = ScalerNetwork()

        model_path = os.path.abspath(os.path.join('weights', 'MILO.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)


    def mask_generator(self, y, x):
        B, C, H, W = y.shape[0:4]

        refScale = [x]
        distScale = [y]

        for intLevel in range(self.number_of_scales):
            # if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
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

        score = ((mask * torch.abs(x - y))).mean()

        return score 

    def MOS_score(self, y, x):
        
        mask = self.mask_generator(x, y)

        score = ((mask * torch.abs(x - y))).mean()

        return 5 * (1 - self.scaler_network(score.reshape(1,1,1,1)))


    def MILO_map(self, y, x):

        C, H, W = x.shape[0:3]

        masks = self.mask_generator(x, y)

        return self.scaler_network((masks*torch.abs(x - y)).mean([1], keepdim=True)) - self.scaler_network(torch.tensor(0.0).cuda().reshape(1,1,1,1)), masks[0]

def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

def map_visualization(input):
    
    input= input.detach().squeeze().cpu().numpy()
    output = CHWtoHWC(index2color(np.round(input * 255.0), get_magma_map()))

    return output

    
if __name__ == '__main__':

    import argparse
    from data import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/ref.png')
    parser.add_argument('--dist', type=str, default='images/dist.png')
    parser.add_argument('--save_dir', type=str, default='save_map')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image(Image.open(args.dist).convert("RGB")).to(device)

    model_milo = MILO().to(device)
    raw_error = model_milo(dist, ref)
    raw_error_cpu = raw_error.detach().cpu().item()
    print('Raw Error: ' + str(raw_error_cpu))
    
    MOS_score = model_milo.MOS_score(dist, ref)
    MOS_score_cpu = MOS_score.detach().cpu().item()
    print('MOS Score: ' + str(MOS_score_cpu))
    
    MILO_err, MILO_mask = model_milo.MILO_map(dist, ref)

    MILO_err = map_visualization(MILO_err)

    MILO_mask = MILO_mask.detach().squeeze().cpu().numpy()
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_image(os.path.join(args.save_dir, 'MILO_map.png'), MILO_err)
    save_image(os.path.join(args.save_dir, 'Mask.png'), MILO_mask)