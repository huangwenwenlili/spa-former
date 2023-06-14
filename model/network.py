from torch import nn
import torch
import math
from model.swish import Swish
from torch.nn import functional as F
from model.base_function import init_net


def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=64)
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)


class Generator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.conv0 = FeedForward(in_ch=4, out_ch=ngf, kernel_size=5, stride=1, padding=2)  # B *ngf * 256, 256
        self.trane256 = TransformerEncoder(in_ch=ngf)
        self.down128 = FeedForward(in_ch=ngf, out_ch=ngf*2, kernel_size=3, stride=2, padding=1) # B *2ngf * 128, 128
        self.trane128 = TransformerEncoder(in_ch=ngf*2)
        self.down64 = FeedForward(in_ch=ngf*2, out_ch=ngf*4, kernel_size=3, stride=2, padding=1) # B *4ngf * 64, 64
        self.trane64 = TransformerEncoder(in_ch=ngf*4)
        self.down32 = FeedForward(in_ch=ngf * 4, out_ch=ngf * 8, kernel_size=3, stride=2, padding=1)  # B *8ngf * 32, 32
        self.trane32 = TransformerEncoder(in_ch=ngf*8)

        self.up64 = FeedForward(in_ch=ngf * 8, out_ch=ngf * 4, kernel_size=3, stride=1, padding=1)  # B *4ngf * 64, 64
        self.trand64 = TransformerEncoder(in_ch=ngf*4)

        self.up128 = FeedForward(in_ch=ngf * 4, out_ch=ngf * 2, kernel_size=3, stride=1, padding=1) # B *2ngf * 128, 128
        self.trand128 = TransformerEncoder(in_ch=ngf * 2)

        self.up256 = FeedForward(in_ch=ngf * 2, out_ch=ngf, kernel_size=3, stride=1, padding=1) # B *ngf * 256, 256
        self.trand256 = TransformerEncoder(in_ch=ngf)

        self.out = FeedForward(in_ch=ngf, out_ch=3, kernel_size=5, stride=1, padding=2)



    def forward(self, x, mask=None):
        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)
        feature = self.conv0(feature)
        #m = F.interpolate(mask, size=feature.size()[-2:], mode='nearest')
        feature = self.trane256(feature)
        feature = self.down128(feature)
        feature = self.trane128(feature)
        feature = self.down64(feature)
        feature = self.trane64(feature)
        feature = self.down32(feature)
        feature = self.trane32(feature)

        feature = torch.nn.functional.interpolate(feature, scale_factor=2, mode='bilinear',
                          align_corners=True)
        feature = self.up64(feature)
        feature = self.trand64(feature)
        feature = torch.nn.functional.interpolate(feature, scale_factor=2, mode='bilinear',
                          align_corners=True)
        feature = self.up128(feature)
        feature = self.trand128(feature)
        feature = torch.nn.functional.interpolate(feature, scale_factor=2, mode='bilinear',
                          align_corners=True)
        feature = self.up256(feature)
        feature = self.trand256(feature)
        out = torch.tanh(self.out(feature))
        return out


class GeneratorU(nn.Module):
    def __init__(self, ngf=64, max_ngf=256):
        super().__init__()
        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )
        self.conv1 = Convblock(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)  # B *ngf * 256, 256
        self.trane256 = TransformerEncoder(in_ch=ngf)
        self.down128 = FeedForward(in_ch=ngf, out_ch=ngf*2, kernel_size=3, stride=2, padding=1) # B *2ngf * 128, 128
        self.trane128 = TransformerEncoder(in_ch=ngf*2)
        self.down64 = FeedForward(in_ch=ngf*2, out_ch=ngf*4, kernel_size=3, stride=2, padding=1) # B *4ngf * 64, 64
        self.trane64 = TransformerEncoder(in_ch=ngf*4)
        self.down32 = FeedForward(in_ch=ngf * 4, out_ch=ngf * 8, kernel_size=3, stride=2, padding=1)  # B *8ngf * 32, 32
        self.trane32 = TransformerEncoder(in_ch=ngf*8)

        self.up64 = FeedForward(in_ch=ngf * 8, out_ch=ngf * 4, kernel_size=3, stride=1, padding=1)  # B *4ngf * 64, 64
        self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1)
        self.trand64 = TransformerEncoder(in_ch=ngf*4)

        self.up128 = FeedForward(in_ch=ngf * 4, out_ch=ngf * 2, kernel_size=3, stride=1, padding=1) # B *2ngf * 128, 128
        self.fuse128 = nn.Conv2d(in_channels=ngf * 2 * 2, out_channels=ngf * 2, kernel_size=1, stride=1)
        self.trand128 = TransformerEncoder(in_ch=ngf * 2)

        self.up256 = FeedForward(in_ch=ngf * 2, out_ch=ngf, kernel_size=3, stride=1, padding=1) # B *ngf * 256, 256
        self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
        self.trand256 = TransformerEncoder(in_ch=ngf)

        self.conv2 = Convblock(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        )


    def forward(self, x, mask=None):
        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)
        feature = self.start(feature)
        feature256 = self.conv1(feature)
        #m = F.interpolate(mask, size=feature.size()[-2:], mode='nearest')
        feature256 = self.trane256(feature256)
        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)
        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)
        feature32 = self.down32(feature64)
        feature32 = self.trane32(feature32)

        out64 = torch.nn.functional.interpolate(feature32, scale_factor=2, mode='bilinear',
                          align_corners=True)
        out64 = self.up64(out64)
        out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
        out64 = self.trand64(out64)
        out128 = torch.nn.functional.interpolate(out64, scale_factor=2, mode='bilinear',
                          align_corners=True)
        out128 = self.up128(out128)
        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
        out128 = self.trand128(out128)
        out256 = torch.nn.functional.interpolate(out128, scale_factor=2, mode='bilinear',
                          align_corners=True)
        out256 = self.up256(out256)
        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
        out256 = self.trand256(out256)
        out256 = self.conv2(out256)
        out = torch.tanh(self.out(out256))
        return out



class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]



class Decoder(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.decoder1 = ResBlock(in_ch=ngf*4, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.decoder12 =ResBlock(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.decoder2 = ResBlock(in_ch=ngf*2, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.decoder22 = ResBlock(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder1(x)
        x = self.decoder12(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder2(x)
        x = self.decoder22(x)
        x = self.decoder3(x)
        return x


class ResBlock0(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)


    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

    def forward(self, x):
        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + residual

        return out


class Convblock(nn.Module):
    def __init__(self, in_ch=256, out_ch=None, kernel_size=3, padding=1, stride=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.norm = nn.InstanceNorm2d(num_features=out_ch, track_running_stats=False)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU()
        )
        self.linear = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.projection(x)
        x1 = self.conv(x)
        x2 = self.gate(x)
        out = x1 * x2
        out = self.norm(out)
        out = self.linear(out)
        out = out + residual
        return out


# (H * W) * C -> (H/2 * C/2) * (4C) -> (H/4 * W/4) * 16C -> (H/8 * W/8) * 64C
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=256):
        super().__init__()
        #self.attn = MultiPatchAttention(patchsizes, num_hidden)
        self.attn = GAttn(in_ch=in_ch)
        self.feed_forward = FeedForward(in_ch=in_ch, out_ch=in_ch)

    def forward(self, x):
        x = self.attn(x)
        x = self.feed_forward(x)
        return x



class FeedForward(nn.Module):
    def __init__(self, in_ch=256, out_ch=None, kernel_size=3, padding=1, stride=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.norm = nn.InstanceNorm2d(num_features=in_ch, track_running_stats=False)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU()
        )
        self.linear = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.projection(x)
        x = self.norm(x)
        x1 = self.conv(x)
        x2 = self.gate(x)
        out = x1 * x2
        out = self.linear(out)
        out = out + residual
        return out


class GAttn(nn.Module):
    def __init__(self, in_ch=256):
        super().__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
            #nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0)
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """
        residual = x
        x = self.norm(x)
        B, C, H, W = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        g = self.gate(x)

        q = q.view(B, C, H * W).contiguous().permute(0,2,1).contiguous()  # b * N * C
        k = k.view(B, C, H * W).contiguous()                              # b * C * N
        v = v.view(B, C, H * W).contiguous().permute(0,2,1).contiguous()  # B * N * C
        kv = torch.einsum('bcn, bnd -> bcd', k, v)
        z = torch.einsum('bnc,bc -> bn', q, k.sum(dim=-1)) / math.sqrt(C)
        z = 1.0 / (z + H * W)
        out = torch.einsum('bnc, bcd-> bnd', q, kv)
        out = out / math.sqrt(C)
        out = out + v
        out = torch.einsum('bnc, bn -> bnc', out, z)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = out * g
        out = self.output_linear(out)
        out = out + residual
        return out



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module