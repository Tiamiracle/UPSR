import torch
import torch.amp as amp
import torch.nn as nn
import functools
import math
import lpips
import os
import os.path as osp
import pyiqa
import numpy as np
import random
import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt

# from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_obj_from_str, get_root_logger, ImageSpliterTh, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from .base_model import BaseModel
from .sr_model import SRModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torchvision import transforms
from PIL import Image

# ============ 权重图生成器 ============
"""输入:一张图片,输出:单通道权重图.值为1表示边缘区域要多加噪"""
class WeightGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel边缘检测算子
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        # 可学习参数α (全局标量，初始0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj_48to1 = nn.Conv2d(48, 1, kernel_size=1, bias=False)

    def get_edge_map(self, x):
        """提取边缘图:输入(B, C, H, W)  输出边缘图edge_map: (B, 1, H, W) 范围[0, 1]亮的表示边缘"""
        # 预处理
        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        if gray.min() < 0:
            gray = (gray + 1) / 2
        gray = torch.clamp(gray, 0, 1)
        gray = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        # Sobel算梯度
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        # 归一化到[0,1]
        edge_map = edge_map / (edge_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)
        edge_map = torch.where(edge_map > 0.1, edge_map, torch.zeros_like(edge_map))
        return edge_map

    def get_high_freq_map(self, x):
        """提取高频图：输入彩色图(B,C,H,W)保留高频细节信息输出单通道高频梯度图(B,1,H,W)"""
        # 预处理
        gray = x.mean(dim=1, keepdim=True)
        if gray.min() < 0:
            gray = (gray + 1) / 2
        gray = torch.clamp(gray, 0, 1)
        gray = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        B, C, H, W = gray.shape
        # FFT处理
        with torch.amp.autocast("cuda", enabled=False):
            gray_32 = gray.float()
            fft = torch.fft.fft2(gray_32, norm="ortho")
            fft_shift = torch.fft.fftshift(fft)
            crow, ccol = H // 2, W // 2
            r, c = H // 4, W // 4
            mask = torch.ones_like(fft_shift)
            mask[:, :, crow - r: crow + r, ccol - c: ccol + c] = 0
            fft_high = fft_shift * mask
            fft_high = torch.fft.ifftshift(fft_high)
            high_spatial = torch.fft.ifft2(fft_high, norm="ortho").real
        # 边缘图
        grad_x = F.conv2d(high_spatial, self.sobel_x, padding=1)
        grad_y = F.conv2d(high_spatial, self.sobel_y, padding=1)
        high_freq_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        # 归一化统一值域
        with torch.no_grad():
            high_freq_map = high_freq_map / \
                (high_freq_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)
        high_freq_map = torch.where(
            high_freq_map > 0.15, high_freq_map, torch.zeros_like(high_freq_map))
        return high_freq_map

    def forward(self, x):
        """生成权重图输入: (B, C, H, W)输出weight_map: (B, 1, H, W)，范围[0,1]
        alpha: 可学习参数"""
        if x.size(1) == 48:
            feat = self.proj_48to1(x)
            x = feat.repeat(1, 3, 1, 1)
        edge_map = self.get_edge_map(x)
        high_freq_map = self.get_high_freq_map(x)

        alpha = torch.clamp(torch.sigmoid(self.alpha), 0.1, 0.9)
        # 加权融合+自适应归一化
        weight_map = alpha * edge_map + (1 - alpha) * high_freq_map
        # 自适应归一化
        weight_map = weight_map / (weight_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)
        weight_map = torch.clamp(weight_map, 0., 1.)
        return weight_map, alpha


# ============ 动态权重管理器 ============
class DynamicWeightManager:
    def __init__(self, weight_generator, T=1000, total_iter=300000):
        self.weight_generator = weight_generator
        self.T = T
        self.total_iter = total_iter
        self.current_weight = None
        self.update_counter = 0
        self.first_update_done = False
        self.training_iter = 0

    def set_training_iter(self, training_iter):
        """设置当前训练迭代次数"""
        self.training_iter = training_iter

    def get_update_info(self, t):
        """判断是否需要更新，并返回平滑系数
        Args:
            t: 当前时间步 (从T到0)
        Returns:
            should_update: bool
            smoothing: float 平滑系数 (W_new = smoothing * W_old + (1-smoothing) * W_current)
        """
        T = self.T
        # 三重控制平滑系数 s = 0.4^(t/T) × (0.5 + f_noise) × d_global
        # 1. 指数衰减项
        exp_decay = math.pow(0.4, t / T)
        
        # 2. 噪声抑制项
        f_noise = (t - 0.6 * T) / (0.4 * T) if T > 0 else 0
        f_noise = max(0.0, min(0.5, f_noise))
        noise_term = 0.5 + f_noise
        
        # 3. 全局衰减项
        if self.total_iter > 0:
            d_global = 1 - 0.5 * (self.training_iter / self.total_iter)
            d_global = max(0.5, min(1.0, d_global))
        else:
            d_global = 1.0
        
        smoothing = exp_decay * noise_term * d_global
        smoothing = max(0.1, min(0.95, smoothing))  # 限制范围
        
        # 每一步都更新
        self.update_counter += 1
        interval = 1
        if self.update_counter >= interval:
            self.update_counter = 0
            return True, smoothing
        return False, None

    def init_weight(self, lr):
        """初始化权重（基于LR原始图像，小尺寸）
        Args:
            lr: 原始LR图像 (B, 3, 64, 64)
        Returns:
            weight: 权重图 (B, 1, 64, 64)
        """
        with torch.no_grad():
            self.current_weight, alpha = self.weight_generator(lr)
        return self.current_weight, alpha

    def update_weight(self, blend_img, smoothing,is_train=False):
        """更新权重图（小尺寸）
        Args:
            blend_img: 混合图 (B, 3, 64, 64)
            smoothing: 平滑系数
            is_train: 是否训练阶段，True开启梯度更新WG，False禁用梯度
        Returns:
            new_weight: 更新后的权重图 (B, 1, 64, 64)
        """
        if is_train:
            # 训练：去掉no_grad，允许梯度回传更新alpha
            new_weight, alpha = self.weight_generator(blend_img)
        else:
            # 推理：关闭梯度省显存
            with torch.no_grad():
                new_weight, alpha = self.weight_generator(blend_img)
        # 指数平滑更新
        new_weight = F.interpolate(new_weight, size=(64,64), mode="bilinear", align_corners=False)
        if self.current_weight is None:
            self.current_weight = new_weight
        else:
            self.current_weight = smoothing * self.current_weight + (1 - smoothing) * new_weight
        # 值域截断到[0, 1]
        self.current_weight = torch.clamp(self.current_weight, 0.0, 1.0)
        self.current_weight = F.interpolate(self.current_weight, size=(64,64), mode="bilinear", align_corners=False)

        return self.current_weight, alpha

    def reset(self, full_reset=False):
        """重置状态
        Args:
            full_reset: True=完全重置(新batch开始)，False=部分重置(保留first_update_done)
        """
        self.current_weight = None
        self.update_counter = 0
        if full_reset:
            self.first_update_done = False

# 模型
class UPSRRealModel(SRModel):
    """Diffusion SR model for single image super-resolution."""
    def __init__(self, opt):
        self.opt = opt
        diff_opt = self.opt['diffusion']
        self.scale = opt['scale']
        self.base_diffusion = build_network(diff_opt)
        
        # 新增静态权重生成器
        self.weight_generator = WeightGenerator()
        total_iter = self.opt['train'].get('total_iter', 300000)
        self.weight_manager = DynamicWeightManager(
            self.weight_generator, 
            T=self.base_diffusion.num_timesteps,
            total_iter=total_iter
        )
        super(UPSRRealModel, self).__init__(opt)
        self.weight_generator = self.weight_generator.to(self.device)

        logger = get_root_logger()

        self.sf = self.opt['scale']

        # # define network net_mse g(\cdot)
        # net_mse_opt = self.opt['network_mse']
        # assert net_mse_opt['ckpt']['path'] is not None, 'ckpt_path is required for net_mse'
        # logger.info(f"Restoring network_mse from {net_mse_opt['ckpt']['path']}")

        # self.net_mse = build_network(net_mse_opt)
        # param_key = net_mse_opt['ckpt'].get('param_key_mse', 'params_ema')
        # self.load_network(self.net_mse, net_mse_opt['ckpt']['path'], net_mse_opt['ckpt'].get('strict_load_mse', True), param_key)
        # self.net_mse.eval()
        # for name, param in self.net_mse.named_parameters():
        #     param.requires_grad = False
        # self.net_mse = self.net_mse.to(self.device)

        # define base_diffusion

        # simulate JPEG compression artifacts
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)

        # define lpips loss
        loss_lpips = self.metric_lpips = pyiqa.create_metric(
            'lpips-vgg', as_loss=True, device=self.device)
        self.loss_lpips = loss_lpips

        if self.opt['rank'] == 0:
            self.metrics_fr = {}
            self.metrics_nr = {}

            for metric_name, metric_opt in self.opt['val']['metrics'].items():
                if metric_opt.get('fr', True):
                    self.metrics_fr[metric_name] = pyiqa.create_metric(
                        metric_name, device=self.device)
                else:
                    self.metrics_nr[metric_name] = pyiqa.create_metric(
                        metric_name, device=self.device)

            self.metrics_fr['psnr'] = pyiqa.create_metric(
                'psnr', test_y_channel=True, color_space='ycbcr', device=self.device)
            self.metrics_fr['ssim'] = pyiqa.create_metric(
                'ssim', test_y_channel=True, color_space='ycbcr', device=self.device)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.
        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        unet_params = list(self.net_g.parameters())
        wg_params = list(self.weight_generator.parameters())
        base_lr = train_opt['optim_g']['lr']
        self.optimizer_g = torch.optim.AdamW([
            {"params": unet_params, "lr": base_lr},
            {"params": wg_params, "lr": base_lr * 0.7}
        ], weight_decay=train_opt['optim_g']['weight_decay'])
        self.optimizers = [self.optimizer_g]

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get(
                    'strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = True
            self.perceptual_weight = train_opt['perceptual_opt']['lpips_weight']
        else:
            self.cri_perceptual = False

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.amp_scaler = amp.GradScaler() if self.opt['train'].get(
            'use_fp16', False) else None

    def backward_step(self, dif_loss_wrapper, micro_lq,
                      micro_gt, num_grad_accumulate, tt):
        loss_dict = OrderedDict()

        context = amp.autocast if self.opt['train'].get(
            'use_fp16', False) else nullcontext
        with context(device_type="cuda"):
            losses, x_t, x0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            l_pix = losses['loss'].mean() / num_grad_accumulate

            l_total = l_pix
            loss_dict['l_pix'] = l_pix

            if self.cri_perceptual:
                rgb_pred = F.pixel_shuffle(x0_pred.clamp(-1., 1.), self.sf)
                rgb_gt = micro_gt
                l_lpips = self.loss_lpips(rgb_pred, rgb_gt).to(x0_pred.dtype).view(-1)
                if torch.any(torch.isnan(l_lpips)):
                    l_lpips = torch.nan_to_num(l_lpips, nan=0.0)
                l_lpips = l_lpips.mean() / num_grad_accumulate * self.perceptual_weight

                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips

        if self.amp_scaler is None:
            l_total.backward()
        else:
            self.amp_scaler.scale(l_total).backward()

        return loss_dict, x_t, x0_pred

    def optimize_parameters(self, current_iter):
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer_g.zero_grad()

        loss_dict = OrderedDict()
        loss_dict['l_pix'] = 0
        if self.cri_perceptual:
            loss_dict['l_lpips'] = 0

        self.weight_manager.reset(full_reset=True)
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj + micro_batchsize,]
            micro_gt = self.gt[jj:jj + micro_batchsize,]
            last_batch = (jj + micro_batchsize >= current_batchsize)
            if self.opt['diffusion'].get('one_step', False):
                tt = torch.ones(
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                    dtype=torch.int32,
                ) * (self.base_diffusion.num_timesteps - 1)
            else:
                tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                )

            # n
            # noise = torch.randn_like(micro_lq_bicubic)
            # lq_cond = micro_lq_bicubic
            # model_kwargs = {'lq': lq_cond,
            #                 } if self.opt['network_g']['params']['cond_lq'] else None
            # compute_losses = functools.partial(
            #     self.base_diffusion.training_losses,
            #     self.net_g,
            #     micro_gt,
            #     micro_lq_bicubic,
            #     micro_lq_bicubic,
            #     micro_uncertainty,
            #     tt,
            #     model_kwargs=model_kwargs,
            #     noise=noise,
            # )
            micro_lq_bicubic = F.interpolate(micro_lq, scale_factor=self.scale, mode='bicubic', align_corners=False)
            noise = torch.randn_like(micro_lq_bicubic)
            # 关键：下采样到64×64，和UNet中间特征尺寸匹配
            lq_cond = F.interpolate(micro_lq_bicubic, size=(micro_lq_bicubic.shape[2]//self.sf, micro_lq_bicubic.shape[3]//self.sf), mode='bilinear')
            # UNet仅需要lq条件
            net_kwargs = {"lq": lq_cond} if self.opt['network_g']['params']['cond_lq'] else {}
            # 扩散损失函数需要的额外参数
            loss_extra_kwargs = {
                "lr_bicubic": micro_lq_bicubic,
                "weight_manager": self.weight_manager,
                "net_kwargs": net_kwargs
            }
            self.base_diffusion.training_iter = current_iter
            self.weight_manager.set_training_iter(current_iter)
            model_kwargs_train = {
                "raw_lr_image": micro_lq
            }
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.net_g,
                micro_gt,
                micro_lq_bicubic,
                micro_lq_bicubic,
                tt,
                model_kwargs=model_kwargs_train,
                loss_extra_kwargs=loss_extra_kwargs,
                noise=noise,
            )

            if last_batch or self.opt['num_gpu'] <= 1:
                losses, x_t, x0_pred = self.backward_step(
                    compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
            else:
                with self.net_g.no_sync():
                    losses, x_t, x0_pred = self.backward_step(
                        compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)

            loss_dict['l_pix'] += losses['l_pix']
            if self.cri_perceptual:
                loss_dict['l_lpips'] += losses['l_lpips']

        if self.opt['train'].get('use_fp16', False):
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            self.optimizer_g.step()

        self.net_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def sample_func(self, y0, noise_repeat=False):
        desired_min_size = self.opt['val']['desired_min_size']
        ori_h, ori_w = y0.shape[2:]
        flag_pad = False
        pad_h = 0
        pad_w = 0
        if not (ori_h % desired_min_size == 0):
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            # 兜底：填充不能超过原图高度
            pad_h = min(pad_h, ori_h - 1)
            flag_pad = True
        if not (ori_w % desired_min_size == 0):
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            pad_w = min(pad_w, ori_w - 1)
            flag_pad = True

        if flag_pad:
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')

        y_bicubic = torch.nn.functional.interpolate(
            y0, scale_factor=self.sf, mode='bicubic', align_corners=False,
        )

        # if self.opt['diffusion']['un'] > 0:
        #     diff = (y_hat - y_bicubic) / 2
        #     un_max = self.opt['diffusion']['un']
        #     b_un = self.opt['diffusion']['min_noise']
        #     un = torch.abs(diff).clamp_(0., un_max) / un_max
        #     un = b_un + (1 - b_un) * un
        # else:
        #     un = torch.ones_like(y_hat)
        lq_cond = F.interpolate(y_bicubic, size=(y_bicubic.shape[2]//self.sf, y_bicubic.shape[3]//self.sf), mode='bilinear')
        # 推理额外传入weight_manager给diffusion采样函数
        self.weight_manager.reset(full_reset=True)
        model_kwargs = {"lq": lq_cond, "weight_manager": self.weight_manager, "raw_lr_image": y0} if self.opt['network_g']['params']['cond_lq'] else {"weight_manager": self.weight_manager, "raw_lr_image": y0}
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        results = self.base_diffusion.ddim_sample_loop(
            y=y_bicubic,
            y_hat=y_bicubic,
            model=net,
            first_stage_model=None,
            noise=None,
            noise_repeat=noise_repeat,
            # clip_denoised=(self.autoencoder is None),
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
            one_step=self.opt['diffusion'].get('one_step', False),
        )

        if flag_pad:
            results = results[:, :, :ori_h * self.sf, :ori_w * self.sf]

        return results.clamp_(-1.0, 1.0)

    def test(self):
        def _process_per_image(im_lq_tensor):
            h, w = im_lq_tensor.shape[2], im_lq_tensor.shape[3]
            chop_size = self.opt['val']['chop_size']
            # 图片尺寸小于分块尺寸，不分块，直接推理
            if h <= chop_size or w <= chop_size:
                im_sr_tensor = self.sample_func(
                    (im_lq_tensor - 0.5) / 0.5,
                    noise_repeat=self.opt['val']['noise_repeat'],
                )
            else:
                im_spliter = ImageSpliterTh(
                    im_lq_tensor,
                    self.opt['val']['chop_size'],
                    stride=self.opt['val']['chop_stride'],
                    sf=self.opt['scale'],
                    extra_bs=self.opt['val']['chop_bs'],
                )
                for im_lq_pch, index_infos in im_spliter:
                    ph, pw = im_lq_pch.shape[2], im_lq_pch.shape[3]
                    im_sr_pch = self.sample_func(
                        (im_lq_pch - 0.5) / 0.5,
                        noise_repeat=self.opt['val']['noise_repeat'],
                    )
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor
        self.output = _process_per_image(self.lq)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.
        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data, training=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if training and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM sharpen the GT images
            if self.opt['degradation']['use_sharp'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(
                ['up', 'down', 'keep'], self.opt['degradation']['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.opt['degradation']['jpeg_range'])
            # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['degradation']['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(
                ['up', 'down', 'keep'], self.opt['degradation']['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['degradation']['scale'] * scale), int(ori_w / self.opt['degradation']['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob2']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will
            # introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(
                    ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(
                    ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['degradation']['gt_size']
            self.gt, self.lq = paired_random_crop(
                self.gt, self.lq, gt_size, self.opt['degradation']['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # for the warning: grad and param do not obey the gradient layout contract
            self.lq = self.lq.contiguous()
            # normalize
            # if self.mean is not None or self.std is not None:
            self.lq = (self.lq - 0.5) / 0.5
            self.gt = (self.gt - 0.5) / 0.5
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            else:
                self.gt = None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting
            # multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        num_img = 0

        for idx, val_data in enumerate(dataloader):
            num_img += len(val_data['lq_path'])
            self.feed_data(val_data, training=False)

            self.test()

            metric_data['img'] = self.output.clamp(0, 1)
            # metric_data['img'] = torch.clamp((self.output * 255.0).round(), 0, 255) / 255.
            metric_data['img2'] = self.gt

            if with_metrics:
                # calculate metrics
                if metric_data['img2'] is not None:
                    for name, metric in self.metrics_fr.items():
                        self.metric_results[name] += metric(metric_data['img'],
                                                           metric_data['img2']).sum().item()
                for name, metric in self.metrics_nr.items():
                    self.metric_results[name] += metric(metric_data['img']).sum().item()

            visuals = self.get_current_visuals()

            sr_img = [tensor2img(visuals['result'][ii])
                      for ii in range(self.output.shape[0])]
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                # metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output

            torch.cuda.empty_cache()

            for ii in range(len(val_data['lq_path'])):
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][ii]))[0]

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')

                    imwrite(sr_img[ii], save_img_path)

            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_img
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict