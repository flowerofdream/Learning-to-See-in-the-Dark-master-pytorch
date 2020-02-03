from __future__ import division
import os, time
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import rawpy
import glob
from PIL import Image
import pickle

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
train_ids = train_ids[0:1]
ps = 512  # patch size for training
# save_freq = 500
save_freq = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 1
    train_ids = train_ids[0:2]


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class Net_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Net_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Net_upblock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Net_upblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Net(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = Net_block(ch_in, 32)
        self.conv2 = Net_block(32, 64)
        self.conv3 = Net_block(64, 128)
        self.conv4 = Net_block(128, 256)
        self.conv5 = Net_upblock(256, 512)

        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=(2, 2))
        self.conv6 = Net_upblock(512, 256)
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=(2, 2))
        self.conv7 = Net_upblock(256, 128)
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=(2, 2))
        self.conv8 = Net_upblock(128, 64)
        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=(2, 2))
        self.conv9 = Net_upblock(64, 32)
        self.conv10 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

        self.Up_conv1 = nn.PixelShuffle(2)  # out channels = inchannels // 2 **2  #batch 3 1024 1024

    def forward(self, x):
        out1 = self.conv1(x)    # 32 512 512
        pool1 = self.pool(out1)    # 32 256 256

        out2 = self.conv2(pool1)     # 64 256 256
        pool2 = self.pool(out2)      # 64 128 128

        out3 = self.conv3(pool2)     # 128 128 128
        pool3 = self.pool(out3)     # 128 64 64

        out4 = self.conv4(pool3)     # 256 64 64
        pool4 = self.pool(out4)     # 256 32 32

        out5 = self.conv5(pool4)     # 512 32 32

        up6 = self.upconv6(out5)    # 256 64 64
        de_output6 = torch.cat((up6, out4), dim=1)  # 512 64 64
        out6 = self.conv6(de_output6)   # 256 64 64

        up7 = self.upconv7(out6)    # 128 128 128
        de_output7 = torch.cat((up7, out3), dim=1)  # 256 128 128
        out7 = self.conv7(de_output7)   # 128 128 128

        up8 = self.upconv8(out7)    # 64 256 256
        de_output8 = torch.cat((up8, out2), dim=1)  # 256 256 256
        out8 = self.conv8(de_output8)   # 64 256 256

        up9 = self.upconv9(out8)    # 32 512 512
        de_output9 = torch.cat((up9, out1), dim=1)  # 64 512 512
        out9 = self.conv9(de_output9)   # 32 512 512

        out10 = self.conv10(out9)   # 12 512 512
        out = self.Up_conv1(out10)  # 3 1024 1024
        return out

def loss_function(in_img, gt_img):
    return torch.mean(torch.abs(gt_img - in_img))

def main(lastepoch, epoch, PreTrain):
    gt_images = [None] * 6000
    input_images = {}
    input_images['300'] = [None] * len(train_ids)
    input_images['250'] = [None] * len(train_ids)
    input_images['100'] = [None] * len(train_ids)
    learning_rate = 1e-4
    UModel = Net(4, 3).cuda()
    Net_optimizer = torch.optim.Adam(UModel.parameters(), lr=learning_rate)
    if PreTrain:
        UModel.load_state_dict(torch.load("./CNNModel.pth"))
    for epoch in range(lastepoch, epoch):
        if os.path.isdir("result/%04d" % epoch):
            continue
        cnt = 0
        if epoch > 2000:
            learning_rate = 1e-5
            Net_optimizer = torch.optim.Adam(UModel.parameters(), lr=learning_rate)
        epoch_time = time.time()

        g_oneloss = np.zeros(len(train_ids))  
        for ind in np.random.permutation(len(train_ids)):  
            st = time.time()  
            train_id = train_ids[ind]
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id) 
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)] 
            in_fn = os.path.basename(in_path)


            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)

            in_exposure = float(in_fn[9:-5]) 
            gt_exposure = float(gt_fn[9:-5]) 
            ratio = min(gt_exposure / in_exposure, 300)

            cnt += 1

            #
            if input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path)
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
                print("读取图片Time=%.3f"%(time.time() - st))

            # crop

            H = input_images[str(ratio)[0:3]][ind].shape[1]
            W = input_images[str(ratio)[0:3]][ind].shape[2]

            xx = 0
            yy = 0

            xx = np.random.randint(0, W - ps) 
            yy = np.random.randint(0, H - ps)

            input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2 , xx * 2: xx * 2 + ps * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            input_patch = np.minimum(input_patch, 1.0)

            input_patch = np.transpose(input_patch, (0, 3, 1, 2))
            gt_patch = np.transpose(gt_patch, (0, 3, 1, 2))
            gt_patch = gt_patch.copy()
            gt_patch = torch.tensor(gt_patch).cuda()
            input_patch = torch.tensor(input_patch).cuda()

            # train
            output = UModel(input_patch)
            Loss = loss_function(output, gt_patch)
            Net_optimizer.zero_grad()
            Loss.backward()
            Net_optimizer.step()
            print("第%d次迭代 第%d张图片 Loss = %.3f Time=%.3f" % (epoch + 1, cnt, Loss.item(), time.time() - st))

            output = np.array(output.cpu().data)
            output = np.minimum(np.maximum(output, 0), 1)

            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d' % epoch):
                    os.makedirs(result_dir + '%04d' % epoch)
                gt_patch = np.array(gt_patch.cpu().data)
                temp = np.transpose(np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)*255, (1, 2, 0))
                Image.fromarray(np.uint8(temp)).save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))


if __name__=="__main__":
    lastepoch = 0
    epoch = 100
    PreTrain = False
    main(lastepoch, epoch, PreTrain)