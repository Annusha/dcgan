import os
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import gan_body
from arg_parse import opt
import re

# /media/annusha/BigPapa/Study/DL/out

# how to launch this file for rooms
# python3 get_samples.py --niter=1000 --outf=./conference_room/
# --netG=/media/annusha/BigPapa/Study/DL/out/netG_epoch_24.pth --nz=100 --ngf=64

# for imagenet
# --niter=1 --outf=./conference_room/ --dataset=imagenet --imageSize=32
# --dataroot=/media/annusha/BigPapa/Study/DL/out_imagenet


def _create_and_save(netG):
    number = len(os.listdir(opt.outf))

    for i in range(number, number + opt.niter):
        noise = torch.FloatTensor(16, opt.nz, 1, 1).normal_(0, 1)
        noise = Variable(noise)
        noise = noise.cuda()

        fake = netG(noise)
        vutils.save_image(fake.data, opt.outf + '%d.png' % i, normalize=True, nrow=4)

if __name__ == '__main__':
    netG = gan_body._netG()


    if opt.dataset == 'imagenet' :
        path_root = opt.dataroot
        path_Gs = [os.path.join(path_root, i) for i in os.listdir(path_root) if 'netG' in i]

        for path_G in path_Gs:
            digit = int(re.findall(r'\d+', path_G)[0])
            if digit < 30:
                print('save from epoch %d'%digit)
                netG.load_state_dict(torch.load(path_G))
                netG.cuda()
                netG.eval()

                _create_and_save(netG)

    else:
        if opt.netG == '':
            print('load weights for generator')
            exit(-1)
        netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        netG.cuda()
        netG.eval()

        _create_and_save(netG)
