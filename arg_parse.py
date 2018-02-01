import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn

# imagenet

# python3 main.py --dataset=imagenet --dataroot=/media/annusha/BigPapa/Study/DL/ImageNet_images
#  --workers=10 --batchSize=256 --imageSize=32 --netG=/media/annusha/BigPapa/Study/DL/out_imagenet/netG_epoch_3.pth
#  --netD=/media/annusha/BigPapa/Study/DL/out_imagenet/netD_epoch_3.pth --outf=/media/annusha/BigPapa/Study/DL/out_imagenet

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataset', default='lsun', help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='/media/annusha/BigPapa/Study/DL/lsun/data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='/media/annusha/BigPapa/Study/DL/out', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--train_svm', action='store_true', help='enable train svm using saved features')


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
# if opt.cuda:
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

# define inner parameters of network which depend on imagesize
kernels = []
strides = []
pads = []
if opt.imageSize == 64:
    kernels = [4, 4, 4, 4, 4]
    strides = [1, 2, 2, 2, 2]
    pads =    [0, 1, 1, 1, 1]

if opt.imageSize == 32:
    # first structure
    # kernels = [4, 4, 2, 4, 4]
    # strides = [1, 2, 2, 1, 2]
    # pads =    [0, 1, 2, 0, 1]

    # second structure
    kernels = [2, 4, 4, 4, 4]
    strides = [1, 2, 2, 2, 2]
    pads =    [0, 1, 1, 1, 1]