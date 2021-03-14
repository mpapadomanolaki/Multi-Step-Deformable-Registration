import torch
import torch.nn as nn
from smoothTransformer import smoothTransformer2D


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv_block,self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,padding=1),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU()
        )

    def forward(self,x):
        x = self.up_conv(x)
        return x


class DisplNet(nn.Module):
    def __init__(self,ch_in=2,step=3, c=4):
        super(DisplNet,self).__init__()

        self.step = step
        self.c = c
        self.ch_in = ch_in

        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = conv_block(ch_in, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
#        self.conv4 = conv_block(64, 128)

#        self.up_conv4 = up_conv_block(128, 128)
        self.up_conv3 = up_conv_block(64, 64)
        self.up_conv2 = up_conv_block(96, 32)
        self.up_conv1 = up_conv_block(48, 16)

        self.deformable_layer = nn.Conv2d(16, 2, 3, padding=1)
        

    def encoder(self, x):
        e1 = self.conv1(x)
        e1 = self.maxpool(e1)

        e2 = self.conv2(e1)
        e2 = self.maxpool(e2)

        e3 = self.conv3(e2)

        return e1,e2,e3


    def decoder(self,e1,e2,e3):
        d3 = self.up_conv3(e3)
        d3 = torch.cat((d3,e2),dim=1)

        d2 = self.upsample(d3)
        d2 = self.up_conv2(d2)
        d2 = torch.cat((d2,e1),dim=1)

        d1 = self.upsample(d2)
        d1 = self.up_conv1(d1)

        return d1


    def deform(self, out_d):
        deformable_1 = self.deformable_layer(out_d)
        return deformable_1


    def forward(self, moving, reference):

        deformed_images = []

        input = torch.cat( (moving, reference), 1)

        for s in range(0,self.step):
            if s==0:
                e1,e2,e3 = self.encoder(input)
                d = self.decoder(e1,e2,e3)
                deformable = self.deform(d)
                out_info = smoothTransformer2D([moving, deformable], self.c)
                deformed, sgrid = out_info
                deformed_images.append(deformed)
            else:
                new_input = torch.cat((previous_deformed,reference), 1)
                e1,e2,e3 = self.encoder(new_input)
                d = self.decoder(e1,e2,e3)
                deformable = deformable + self.deform(d)
                out_info = smoothTransformer2D([moving, deformable, previous_sgrid], self.c)
                deformed, sgrid = out_info
                deformed_images.append(deformed)

            previous_deformed = deformed
            previous_sgrid = sgrid

        return deformable, sgrid, deformed_images
