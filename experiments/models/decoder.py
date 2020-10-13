import torch
import torch.nn as nn
import numpy as np

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)        

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*4,
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        bn = True,
        upsample_first = True,
        bias=False
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        
        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad, bias=bias))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad,bias=bias))        
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            if act_fun is not None:    
                model.add(act_fun)
            if( (not bn_before_act) and bn):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad,bias=bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model



# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, 1, padding=0, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out

def resdecoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    model = nn.Sequential()

    for i in range(len(num_channels_up)-2):
        
        model.add( ResidualBlock( num_channels_up[i], num_channels_up[i+1]) )
        
        if upsample_mode!='none':
            model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))	
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        
        if i != len(num_channels_up)-1:	
            model.add(act_fun)
            #model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
                
    # new
    model.add(ResidualBlock( num_channels_up[-1], num_channels_up[-1]))
    #model.add(nn.BatchNorm2d( num_channels_up[-1] ,affine=bn_affine))
    model.add(act_fun)
    # end new
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model

##########################


def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def set_to(tensor,mtx):
    if not len(tensor.shape)==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            if i == j:
                tensor[i,j] = np_to_tensor(mtx)
            else:
                tensor[i,j] = np_to_tensor(np.zeros(mtx.shape))
    return tensor

def conv2(in_f, out_f, kernel_size, stride=1, pad='zero',bias=False):
    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if kernel_size != 4:
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    else:
        padder = nn.ReflectionPad2d( (1,0,1,0) )
        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=1, bias=bias)
    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def fixed_decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        need_sigmoid=True, 
        pad ='reflection', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_affine = True,
        bn = True,
        mtx = np.array( [[1,3,3,1] , [3,9,9,3], [3,9,9,3], [1,3,3,1] ] )*1/16.,
        output_padding = 0,padding=1,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    model = nn.Sequential()
   
    for i in range(len(num_channels_up)-2):
        
        # those will be fixed
        model.add(conv2( num_channels_up[i], num_channels_up[i],  4, 1, pad=pad))  
        # those will be learned
        model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad))  
        
        if i != len(num_channels_up)-1:	
            if act_fun is not None:    
                model.add(act_fun)
            model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
      
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    ###
    # this is a Gaussian kernel
    
    # set filters to fixed and then set the gradients to zero
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if(m.kernel_size == mtx.shape):
                m.weight.data = set_to(m.weight.data,mtx)
                for param in m.parameters():
                    param.requires_grad = False
    ###
    
    return model


####

def deconv_decoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size=1,
        pad ='reflection', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_affine = True,
        stride=2,
        padding=0,
        output_padding=0,
        final_conv=False,
        ):
    
    n_scales = len(num_channels_up) 
    
    model = nn.Sequential()
    
    for i in range(len(num_channels_up)-1):
        
        model.add( 
            nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1], filter_size, stride=stride, padding=padding, output_padding=output_padding, groups=1, bias=False, dilation=1)
        )
        #model.add(deconv(num_channels_up[i], num_channels_up[i+1], filter_size, stride,pad))
        
        if i != len(num_channels_up)-1:	
            model.add(act_fun)
            model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    if final_conv:
        model.add(conv( num_channels_up[-1], num_channels_up[-1],  1, 1, pad=pad))
        model.add(act_fun)
        model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    model.add(nn.Sigmoid())
    
    return model


#####


def fixed_deconv_decoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size=1,
        pad ='reflection', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_affine = True,
        mtx = np.array( [[1,4,7,4,1] , [4,16,26,16,4], [7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]] ),
        output_padding=1,
        padding=2,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    model = nn.Sequential()
    
    for i in range(len(num_channels_up)-1):
        
        # those will be learned - conv
        model.add(conv( num_channels_up[i], num_channels_up[i+1],  1, 1, pad=pad)) 
        
        # those will be fixed - upsample
        model.add( nn.ConvTranspose2d(
            num_channels_up[i], 
            num_channels_up[i+1], 
            kernel_size=4, 
            stride=2, 
            padding=padding,
            output_padding=output_padding, groups=1, bias=False, dilation=1) )      
        
        if i != len(num_channels_up)-1:	
            model.add(act_fun)
            model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
       
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    model.add(nn.Sigmoid())
    
    ###
    # this is a Gaussian kernel
    # set filters to fixed and then set the gradients to zero
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            if(m.kernel_size == mtx.shape):
                m.weight.data = set_to(m.weight.data,mtx)
                for param in m.parameters():
                    param.requires_grad = False
    ###
        
    return model


