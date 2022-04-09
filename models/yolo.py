import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from carafe import CARAFEPack
from models.common import *
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)

import ipdb


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), strides=None):  # detection layer
        super(Detect, self).__init__()
        if strides is None:
            strides = [8, 16, 32]
        self.stride = strides  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        # a=[3, 3, 2] anchors以[w, h]对的形式存储，3个feature map，每个feature map上有三个anchor(w,h)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # 模型中需要保存的参数一般有两种: 一种是反向传播需要被optimizer更新的，称为parameter;
        #                             另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # output conv，对每个输出的feature map都要调用一次conv1x1
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        # ----------decoupled head -------------------#

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(ch)):
            self.stems.append(
                Conv(c1=ch[i], c2=256, k=1, s=1, act=True, )
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(c1=256, c2=256, k=3, s=1, act=True, ),
                        Conv(c1=256, c2=256, k=3, s=1, act=True, ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(c1=256, c2=256, k=3, s=1, act=True, ),
                        Conv(c1=256, c2=256, k=3, s=1, act=True, ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(256, self.na * self.nc, 1, 1, padding=0,)
            )
            self.reg_preds.append(
                nn.Conv2d(256, self.na * 4, 1, 1, padding=0, )
            )
            self.obj_preds.append(
                nn.Conv2d(256, self.na * 1, 1, 1, padding=0, )
            )

        # -------decoupled head -----------------#

        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export

        # ------- decoupled head  ----------------- #

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.stride, x)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # ipdb.set_trace()
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            bs, _, ny, nx = output.shape
            output = output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:

                if self.grid[k].shape[2:4] != output.shape[2:4]:
                    self.grid[k] = self._make_grid(nx, ny).to(output.device)

                y = output.sigmoid()
                y[..., 0:2] = (output[..., 0:2] * 2. - 0.5 + self.grid[k].to(output.device)) * self.stride[k]
                y[..., 2:4] = (output[..., 2:4] * 2) ** 2 * self.anchor_grid[k]
                z.append(y.view(bs, -1, self.no))

            # if self.training:
            #     output = torch.cat([reg_output, obj_output, cls_output], 1)
            #
            #     output, grid = self.get_output_and_grid(output, k, stride_this_level, x[0].dtype)
            #
            #     x_shifts.append(grid[:, :, 0])
            #     y_shifts.append(grid[:, :, 1])
            #     expanded_strides.append(
            #         torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(x[0])
            #     )
            # else:
            #     output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        return outputs if self.training else (torch.cat(z, 1), outputs)

        # -------decoupled head -----------------#

        # ipdb.set_trace()
        # for i in range(self.nl):
        #     x[i] = self.m[i](x[i])  # conv
        #     bs, _, ny, nx = x[i].shape  # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
        #     x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        #
        #     if not self.training:  # inference
        #         # 构造网格
        #         # 因为推理返回的不是归一化后的网格偏移量，需要再加上网格的位置，得到最终的推理坐标后再送入NMS
        #         # 所以这里构建网格就是为了记录每个grid的网格坐标，方面后面使用
        #         if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        #             self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
        #
        #         y = x[i].sigmoid()
        #         y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
        #         y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        #         z.append(y.view(bs, -1, self.no))
        #
        # return x if self.training else (torch.cat(z, 1), x)

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grid[k]

        batch_size = output.shape[0]

        n_ch = self.no

        h_size, w_size = output.shape[-2:]

        if grid.shape[2:4] != output.shape[2:4]:
            grid = self._make_grid(h_size, w_size).to(output.device)
            self.grid[k] = grid

        output = output.view(batch_size, self.na, n_ch, h_size, w_size)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.na * h_size * w_size, -1).contiguous()

        grid = grid.view(1, -1, 2)

        output[..., :2] = (output[..., :2] + grid.to(output.device)) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov4-p5.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # 如果配置文件里有中文，打开时要加encoding参数
            with open(cfg, encoding='utf-8') as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # 这里是因为下面求下采样率时前向传播到carafe时会出错，故设置了这个
        tmp_device = select_device()
        self.model.to(tmp_device)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s).to(tmp_device))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1).to(tmp_device)
            check_anchor_order(m)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            self.initialize_biases()  # decoupled head
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:].data += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def initialize_biases(self, prior_prob=1e-2):
        m = self.model[-1]
        for conv in m.cls_preds:
            b = conv.bias.view(m.na, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in m.obj_preds:
            b = conv.bias.view(m.na, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from不是-1的层结构序号
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # i: 当前层索引
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # eval(string) 得到当前层的真实类名。例如: m=Conv -> <class 'models.common.Conv'>
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # depth gain控制深度。  n: 当前模块的(间接控制深度)
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
            # c1:当前层的输入通道数  c2: 当前层的输出通道数 ch: 所有层的输出通道数
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            # 最后一层不用控制宽度，直接输出
            # width gain 控制宽度 c2: 当前层的最终输出的channel数(间接控制宽度)
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            args = [c1, c2, *args[1:]]
            # 如果当前层是如下, 则需要在args中加入bottleneck的个数
            # [in_channel, out_channel, Bottleneck的个数n, bool(True表示有shortcut 默认，反之无)]
            if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:
                args.insert(2, n)  # 在第2个位置插入bottleneck个数n
                n = 1  # 恢复默认值
        elif m in [HarDBlock, HarDBlock2]:
            c1 = ch[f]
            args = [c1, *args[:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            # 将f中所有的输出累加得到输出channel
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        # m_: 得到当前层module。如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type 'modules.common.Conv'
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        # 把所有层结构中from不是-1的值记下
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if m in [HarDBlock, HarDBlock2]:
            c2 = m_.get_out_ch()
            ch.append(c2)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov4-p5.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
