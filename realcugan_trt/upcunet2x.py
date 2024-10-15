'''
cache_mode:
0:使用cache缓存必要参数
1:使用cache缓存必要参数，对cache进行8bit量化节省显存，带来小许延时增长
2:不使用cache，耗时约为mode0的2倍，但是显存不受输入图像分辨率限制，tile_mode填得够大，1.5G显存可超任意比例
'''
import torch,pdb
from torch import nn as nn
from torch.nn import functional as F
import os,sys
import numpy as np
from typing import Tuple
import tensorrt
import tensorrt as trt
from torch2trt import torch2trt
from typing import Union, Optional, Sequence, Dict, Any
import threading

torch.backends.nnpack.enabled = False

root_path=os.path.abspath('.')
sys.path.append(root_path)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x
    
    def forward_mean(self, x, x0) -> torch.Tensor:
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z
class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1,x2) -> torch.Tensor:
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1, x2) -> torch.Tensor:
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x,alpha=1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4*=alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x2) -> Tuple[torch.Tensor, torch.Tensor]:
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x2,x3

    def forward_c(self, x2, x3) -> torch.Tensor:
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4) -> torch.Tensor:
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class UpCunet2x(nn.Module):
    def __init__(self, tile_mode, alpha, h, w, is_half, is_pro, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.tile_mode = tile_mode
        self.alpha = alpha
        self.is_half = is_half
        self.h = h
        self.w = w
        self.is_pro = is_pro

        if(self.tile_mode == 0):
            pass
        elif self.tile_mode: # 对长边减半
            if(self.w >= self.h):
                crop_size_w = ((self.w-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_h = (self.h-1)//2*2+2#能被2整除
            else:
                crop_size_h = ((self.h-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_w = (self.w-1)//2*2+2#能被2整除
            self.crop_size = (crop_size_h,crop_size_w)
        elif(self.tile_mode >= 2):
            tm = min(min(self.h,self.w)//128,int(tile_mode))#最小短边为128*128
            t2 = tm*2
            self.crop_size = (((self.h-1)//t2*t2+t2)//tm,((self.w-1)//t2*t2+t2)//tm)
        else:
            print("tile_mode config error")
            os._exit(233)

        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

        if self.tile_mode == 0:
            self.ph = ((self.h - 1) // 2 + 1) * 2
            self.pw = ((self.w - 1) // 2 + 1) * 2
        else:
            self.ph = ((self.h - 1) // self.crop_size[0] + 1) * self.crop_size[0]
            self.pw = ((self.w - 1) // self.crop_size[1] + 1) * self.crop_size[1]
        self.padh = 18 + self.ph - self.h
        self.padw = 18 + self.pw - self.w

        tmp = torch.zeros((1, 3, self.h, self.w))
        ref = F.pad(tmp, (18, self.padw, 18, self.padh), 'reflect')
        self.refh, self.refw = ref.shape[2], ref.shape[3]
        del tmp, ref
    
    def forward(self, x:torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        if self.is_pro: x = x / (255 / 0.7) + 0.15
        else: x = x / 255.0
        if self.tile_mode == 0:
            x = F.pad(x, (18, self.padw, 18, self.padh), 'reflect')
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x, self.alpha)
            x = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x)
            if (self.w != self.pw or self.h != self.ph):
                x = x[:, :, :self.h * 2, :self.w * 2]
            return x
        x = F.pad(x, (18, self.padw, 18, self.padh),'reflect')
        n, c, h, w = 1, 3, self.refh, self.refw
        if self.is_half:
            se_mean0=torch.zeros((n,64,1,1),device=x.device,dtype=torch.float16)
        else:
            se_mean0=torch.zeros((n,64,1,1),device=x.device,dtype=torch.float32)
        n_patch  = 0
        tmp_dict = {}
        for i in range(0,h-36,self.crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-36,self.crop_size[1]):
                x_crop=x[:,:,i:i+self.crop_size[0]+36,j:j+self.crop_size[1]+36]
                n,c1,h1,w1=x_crop.shape
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if self.is_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0 /= n_patch
        if self.is_half:
            se_mean1=torch.zeros((n,128,1,1),device=x.device,dtype=torch.float16)
        else:
            se_mean1=torch.zeros((n,128,1,1),device=x.device,dtype=torch.float32)
        for i in range(0,h-36,self.crop_size[0]):
            for j in range(0,w-36,self.crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                opt_unet1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                if self.is_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        if self.is_half:
            se_mean0=torch.zeros((n,128,1,1),device=x.device,dtype=torch.float16)
        else:
            se_mean0=torch.zeros((n,128,1,1),device=x.device,dtype=torch.float32)
        for i in range(0,h-36,self.crop_size[0]):
            for j in range(0,w-36,self.crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x2,tmp_x3=self.unet2.forward_b(tmp_x2)
                if self.is_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        if self.is_half:
            se_mean1=torch.zeros((n,64,1,1),device=x.device,dtype=torch.float16)
        else:
            se_mean1=torch.zeros((n,64,1,1),device=x.device,dtype=torch.float32)
        for i in range(0,h-36,self.crop_size[0]):
            for j in range(0,w-36,self.crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)*self.alpha
                if self.is_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        res = torch.zeros((n, c, h * 2 - 72, w * 2 - 72),dtype=x.dtype,device=x.device)
        for i in range(0,h-36,self.crop_size[0]):
            for j in range(0,w-36,self.crop_size[1]):
                x,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                del tmp_dict[i][j]
                x = torch.add(x0, x)#x0是unet2的最终输出
                res[:, :, i * 2:i * 2 + h1 * 2 - 72, j * 2:j * 2 + w1 * 2 - 72] = x
        del tmp_dict
        if(self.w!=self.ph or self.h!=self.ph):res=res[:,:,:self.h*2,:self.w*2]
        if self.is_pro:
            res = ((res - 0.15) * (255/0.7)).round().clamp_(0, 255)
        else:
            res = (res * 255.0).round().clamp_(0, 255)
        res = res.permute(0, 2, 3, 1)
        return res

from .trt_module import TRTModule
class RealCUGANUpScaler2x(object):
    def __init__(self, weight_path, half, tile, alpha, h, w, accel=False, device="cpu", export_engine=False, export_engine_verbose=False):
        self.half               =   half
        self.device             =   device
        self.tile               =   tile
        self.alpha              =   alpha
        self.h                  =   h
        self.w                  =   w

        self.accel              =   accel
        self.export_engine      =   export_engine
        self.onnx_name          =   None

        weight                  =   torch.load(weight_path, map_location="cpu", weights_only=True)
        self.pro                =   "pro" in weight
        if(self.pro):
            del weight["pro"]

        if export_engine:
            if device == "cpu":
                raise Exception("Cannot export engine on CPU")

            folder = os.path.join(".", "models")
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.onnx_name = os.path.join(".", "models", f"model_2x_{str(self.half)}_{str(self.pro)}_{tile}_{alpha}_{h}_{w}.onnx")
            self.engine_name = os.path.join(".", "models", f"model_2x_{str(self.half)}_{device}_{str(self.pro)}_{tile}_{alpha}_{h}_{w}.trt")

            import subprocess
            import multiprocessing
            device_index = int(device.split(":")[-1])

            def subproc():
                if os.path.exists(self.onnx_name) and os.path.exists(self.engine_name):
                    return
                import pycuda.driver as cuda
                cuda.init()
                cuda.Device(device_index).make_context()
                if not os.path.exists(self.onnx_name):
                    temp_model = UpCunet2x(tile, alpha, h, w, False, self.pro)
                    temp_model.load_state_dict(weight, strict=True)
                    with torch.no_grad():
                        arr = torch.zeros((1, self.h, self.w, 3))
                        torch.onnx.export(model=temp_model,
                                        args=arr, 
                                        f=self.onnx_name, 
                                        verbose=export_engine_verbose, 
                                        opset_version=11, 
                                        do_constant_folding=False,
                                        input_names=['input:0'], 
                                        output_names=['output:0'])
                
                if not os.path.exists(self.engine_name):
                    logger = trt.Logger(trt.Logger.WARNING)
                    builder = trt.Builder(logger)
                    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    network = builder.create_network(network_flags)

                    parser = trt.OnnxParser(network, logger)
                    success = parser.parse_from_file(self.onnx_name)
                    if not success:
                        raise Exception("ONNX parsing failed")
                
                    config = builder.create_builder_config()
                    if builder.platform_has_fast_fp16 and self.half:
                        config.set_flag(trt.BuilderFlag.FP16)
                    else:
                        if self.half:
                            raise Exception("FP16 not supported on this platform")

                    engine_bytes = builder.build_serialized_network(network, config)
                    with open(self.engine_name, "wb") as f:
                        f.write(engine_bytes)
                cuda.Context.pop()
                
            proc = multiprocessing.Process(target=subproc)
            proc.start()
            proc.join()

            if proc.exitcode != 0:
                raise Exception(f"Failed to export engine {proc.exitcode}")
                
            self.model = TRTModule(self.engine_name, device_index)
        else:
            self.model = UpCunet2x(tile, alpha, h, w, half, is_pro=self.pro)
            if(half==True): self.model=self.model.half().to(device)
            else: self.model=self.model.to(device)
            self.model.load_state_dict(weight, strict=True)

    def tensor2np(self, tensor: torch.Tensor):
        return tensor.cpu().byte().numpy()

    def np2input(self, frame):
        if self.half and not self.export_engine:
            return frame.astype(np.float16)
        else:
            return frame.astype(np.float32)

    def __call__(self, frame):
        with torch.no_grad():
            inputx = self.np2input(frame)
            if self.export_engine:
                model_result = self.model(inputx)
            else:
                inputx = torch.from_numpy(inputx).to(self.device)
                model_result = self.model(inputx)
            result = self.tensor2np(model_result)
        return result
