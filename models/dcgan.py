import torch
import torch.nn as nn 
import torch.nn.functional as F

def dcgan(inp=2,
          ndf=32,
          num_ups=4, need_sigmoid=True, need_bias=True, pad='zero', upsample_mode='nearest', need_convT = True):
    
    layers= [nn.ConvTranspose2d(inp, ndf, kernel_size=3, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(ndf),
             nn.LeakyReLU(True)]
    
    for i in range(num_ups-3):
        if need_convT:
            layers += [ nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(True)]
        else:
            layers += [ nn.Upsample(scale_factor=2, mode=upsample_mode),
                        nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(True)]
            
    if need_convT:
        layers += [nn.ConvTranspose2d(ndf, 3, 4, 2, 1, bias=False),]
    else:
        layers += [nn.Upsample(scale_factor=2, mode='bilinear'),
                   nn.Conv2d(ndf, 3, kernel_size=3, stride=1, padding=1, bias=False)]
    
    
    if need_sigmoid:
        layers += [nn.Sigmoid()]

    model =nn.Sequential(*layers)
    return model


class DCGAN_XRAY(nn.Module):
    def __init__(self, nz, ngf=64, output_size=256, nc=3, num_measurements=1000):
        super(DCGAN_XRAY, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.conv3 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv6 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        self.conv7 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)  # output is image

    def forward(self, z):
        input_size = z.size()
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = torch.sigmoid(self.conv7(x, output_size=(-1, self.nc, self.output_size, self.output_size)))

        return x


class DCGAN_MNIST(nn.Module):
    def __init__(self, nz, ngf=64, output_size=28, nc=1, num_measurements=10):
        super(DCGAN_MNIST, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False)

    def forward(self, x):
        input_size = x.size()

        # DCGAN_MNIST with old PyTorch version
        # x = F.upsample(F.relu(self.bn1(self.conv1(x))),scale_factor=2)
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.upsample(F.relu(self.bn3(self.conv3(x))),scale_factor=2)
        # x = F.upsample(F.relu(self.bn4(self.conv4(x))),scale_factor=2)
        # x = torch.tanh(self.conv5(x,output_size=(-1,self.nc,self.output_size,self.output_size)))

        x = F.interpolate(F.relu(self.bn1(self.conv1(x))), scale_factor=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(F.relu(self.bn3(self.conv3(x))), scale_factor=2)
        x = F.interpolate(F.relu(self.bn4(self.conv4(x))), scale_factor=2)
        x = torch.sigmoid(self.conv5(x, output_size=(-1, self.nc, self.output_size, self.output_size)))

        return x


class DCGAN_RETINO(nn.Module):
    def __init__(self, nz, ngf=64, output_size=256, nc=3, num_measurements=1000):
        super(DCGAN_RETINO, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.conv3 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv6 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        # self.fc = nn.Linear((output_size)*(output_size)*nc,num_measurements, bias=False) #fc layer - old version

    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.sigmoid(self.conv6(x, output_size=(-1, self.nc, self.output_size, self.output_size)))

        return x