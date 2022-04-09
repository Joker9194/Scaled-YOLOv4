import math
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def init_seeds(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    benchmark模式会自动寻找最优配置，但由于计算的随机性，每次网络进行前向反馈时会有差异
    避免这样差异的方式就是将deterministic设置为True（该设置表明每次卷积的高效算法均相同）
    """
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()  # the number of GPU
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    """
    精确计算当前时间，并返回
    先进行torch.cuda.synchronize()添加同步操作 再返回time.time()当前时间
    先执行同步操作，再取时间的原因:
       在pytorch里面，程序的执行都是异步的。
       如果time.time(), 测试的时间会很短，因为执行完end=time.time()程序就退出了
       而先加torch.cuda.synchronize()会先同步cuda的操作，等待gpu上的操作都完成了再继续运行end = time.time()
       这样子测试时间会准确一点
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model):
    # Return True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # 返回key在da和db中且da和db中key对应的value的shape相同，且key不在exclude中
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:  # 如果是二维卷积就跳过，或者可以使用何凯明初始化
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:  # 设置BN层eps和momentum参数
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            # 如果是这几类激活函数 inplace插值就赋为True
            # inplace = True指进行原地操作，对于上层网络传递下来的tensor直接进行修改，不需要另外赋值变量
            # 这样可以节省运算内存，不用多储存变量
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    # a: 初始化模型的总参数个数(前向+反向). b: 模型参数中值为0的参数个数b
    a, b = 0., 0.
    # model.parameters()返回模型model的参数，返回一个生成器，需要用for循环或者next()来获取参数
    # for循环取出每一层的前向传播和反向传播的参数
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    # b / a 即可以反应模型的稀疏程度
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    # 对模型model进行剪枝操作, 以增加模型的稀疏性. 使用prune工具将参数稀疏化
    import torch.nn.utils.prune as prune  # 导入工具包
    print('Pruning model... ', end='')
    # 模型的迭代器，返回的是所有模块的迭代器，同时产生模块的名称(name)以及模块本身(m)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # 对当前层结构m, 随机裁剪(总参数量 x amount)数量的权重(weight)参数
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            # 彻底移除被裁剪的的权重参数
            prune.remove(m, 'weight')  # make permanent
    # 输出模型的稀疏度，调用sparsity函数计算当前模型的稀疏度
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    """
     融合卷积层和BN层(测试推理使用)   Fuse convolution and batchnorm layers
    方法: 卷积层还是正常定义, 但是卷积层的参数w,b要改变   通过只改变卷积参数, 达到CONV+BN的效果
          w = w_bn * w_conv   b = w_bn * b_conv + b_bn   (可以证明)
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    https://zhuanlan.zhihu.com/p/94138640
    """
    #
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        # w_conv: 卷积层的w参数 直接clone conv的weight即可
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        # w_bn: bn层的w参数  torch.diag: 返回一个以input为对角线元素的2D/1D 方阵/张量
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        # w = w_bn * w_conv      torch.mm: 对两个矩阵相乘
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        # b_conv: 卷积层的b参数，如果不为None就直接读取conv.bias即可
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        # b_bn: bn层的b参数
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        #  b = w_bn * b_conv + b_bn
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)  # 640x640 FLOPS
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    # 用于检测结束后可能需要第二次分类，直接修改torchvision中的预训练

    # 加载torchvision中已经写好的pretrained模型  reshape为n类输出
    model = models.__dict__[name](pretrained=True)

    # Display model properties
    input_size = [3, 224, 224]
    input_space = 'RGB'
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        print(x + ' =', eval(x))

    # Reshape output to n classes
    # 将加载的预训练模型的最后一层的分类类别数改为n
    # 总体的过程 = 将fc层的权重和偏置清0 + 修改类别个数为n
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16, 3, 256, 416), r=ratio
    """ scales img(bs, 3, y, x) by ratio
    实现对图片的缩放操作
    @param img: 原图
    @param ratio: 缩放比例
    @param same_shape: 保持缩放后的图像尺寸？
    """

    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 128  # 64 # 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    模型的指数加权平均方法，是一种给予近期数据更高权重的平均方法，利用滑动平均的参数来提高模型在测试数据上的健壮性/鲁棒性
    一般用于测试集
    https://www.bilibili.com/video/BV1FT4y1E74V?p=63
    https://www.cnblogs.com/wuliytTaotao/p/9479958.html
    https://zhuanlan.zhihu.com/p/68748778
    https://zhuanlan.zhihu.com/p/32335746
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        @param decay: 衰减函数参数，默认0.9999，即考虑过去10000次的真实值
        @param updates: ema更新次数
        """
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            # 随机更新次数，更新参数beta(d)
            d = self.decay(self.updates)

            # msd: 模型配置的字典 model state_dict，msd中的数据保持不变 用于训练
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            # 遍历模型配置字典 如: k=linear.bias  v=[0.32, 0.25]  ema中的数据发生改变，用于测试
            for k, v in self.ema.state_dict().items():
                # 这里得到的v: 预测值
                if v.dtype.is_floating_point:
                    v *= d
                    # .detach() 使对应的Variables与网络隔开而不参与梯度更新
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
