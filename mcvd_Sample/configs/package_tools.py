import numpy as np
import torch
import torch.nn as nn
from models.better.ncsnpp_more import  NCSNpp
from functools import partial
from torch.distributions.gamma import Gamma
import logging
from datasets import get_dataset, data_transform, inverse_data_transform

def get_sigmas(config):

    T = getattr(config.model, 'num_classes')

    if config.model.sigma_dist == 'geometric':

        return torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end),
                              T).to(config.device)

    elif config.model.sigma_dist == 'linear':
        print(len(torch.logspace(np.log10(config.model.sigma_begin), np.log10(config.model.sigma_end),
                              T).to(config.device)))
        return torch.linspace(config.model.sigma_begin, config.model.sigma_end,
                              T).to(config.device)

    elif config.model.sigma_dist == 'cosine':
        t = torch.linspace(T, 0, T+1)/T
        s = 0.008
        f = torch.cos((t + s)/(1 + s) * np.pi/2)**2
        return f[:-1]/f[-1]

    else:
        raise NotImplementedError('sigma distribution not supported')

@torch.no_grad()
def revise_model(config,model,  gen_samples):
    gen_samples = gen_samples.reshape(gen_samples.shape[0], -1, config.data.channels,
                                      config.data.image_height, config.data.image_width).permute(0, 1, 3, 4, 2)
    gen_samples = torch.squeeze(reshape_patch_back_torch(gen_samples, config.data.patch))
    gen_samples = inverse_data_transform(config, gen_samples)
    gen_samples = model(gen_samples)
    gen_samples = data_transform(config, gen_samples)
    gen_samples = gen_samples.unsqueeze(-1)
    gen_samples = reshape_patch_torch(gen_samples, config.data.patch)
    gen_samples = gen_samples.permute(0, 1, 4, 2, 3)
    gen_samples = gen_samples.reshape(gen_samples.shape[0], gen_samples.shape[1] * gen_samples.shape[2],
                                      gen_samples.shape[3], gen_samples.shape[4])
    return gen_samples

@torch.no_grad()
def ddpm_sampler(x_mod, scorenet, config=None, cond=None, just_beta=False, final_only=False, denoise=True, subsample_steps=None,
                 same_noise=False, noise_val=None, frac_steps=None, verbose=False, log=False, clip_before=True,
                 t_min=-1, gamma=False, **kwargs):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    schedule = getattr(config.model, 'sigma_dist', 'linear')
    if schedule == 'linear':
        betas = get_sigmas(config)
        alphas = torch.cumprod(1 - betas.flip(0), 0).flip(0)
        alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
    elif schedule == 'cosine':
        alphas = get_sigmas(config)
        alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
        betas = (1 - alphas/alphas_prev).clip_(0, 0.999)
    # alphas, alphas_prev, betas = net.alphas, net.alphas_prev, net.betas
    steps = np.arange(len(betas))
    if gamma:
        ks_cum, thetas = net.k_cum, net.theta_t

    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    # New ts (see https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py)
    if subsample_steps is not None:
        if subsample_steps < len(alphas):
            skip = len(alphas) // subsample_steps
            steps = range(0, len(alphas), skip)
            steps = torch.tensor(steps, device=alphas.device)
            # new alpha, beta, alpha_prev
            alphas = alphas.index_select(0, steps)
            alphas_prev = torch.cat([alphas[1:], torch.tensor([1.0]).to(alphas)])
            betas = 1.0 - torch.div(alphas, alphas_prev) # for some reason we lose a bit of precision here
            if gamma:
                ks_cum = ks_cum.index_select(0, steps)
                thetas = thetas.index_select(0, steps)

    if frac_steps is not None:
        steps = steps[int((1 - frac_steps)*len(steps)):]
        alphas = alphas[steps]
        alphas_prev = alphas_prev[steps]
        betas = betas[steps]
        if gamma:
            ks_cum = ks_cum[steps]
            thetas = thetas[steps]

    if same_noise and noise_val is None:
        noise_val = x_mod.detach().clone()

    images = []
    scorenet = partial(scorenet, cond=cond)
    x_transf = False

    L = len(steps)
    for i, step in enumerate(steps):
        if step < t_min*len(alphas): # wait until it happens
            continue
        if not x_transf and t_min > 0: # we must add noise to the previous frame
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                z = (z - ks_cum[i]*thetas[i]) / (1 - alphas[i]).sqrt()
            else:
                z = torch.randn_like(x_mod)
            x_mod = alphas[i].sqrt() * x_mod + (1 - alphas[i]).sqrt() * z
        x_transf = True
        c_beta, c_alpha, c_alpha_prev = betas[i], alphas[i], alphas_prev[i]
        labels = (step * torch.ones(x_mod.shape[0], device=x_mod.device)).long()

        grad = scorenet(x_mod, labels)


        # x_mod = 1 / (1 - c_beta).sqrt() * (x_mod + c_beta / (1 - c_alpha).sqrt() * grad)
        x0 = (1 / c_alpha.sqrt()) * (x_mod - (1 - c_alpha).sqrt() * grad)
        if clip_before:
            x0 = x0.clip_(-1, 1)
        x_mod = (c_alpha_prev.sqrt() * c_beta / (1 - c_alpha)) * x0 + ((1 - c_beta).sqrt() * (1 - c_alpha_prev) / (1 - c_alpha)) * x_mod

        if not final_only:
            images.append(x_mod.to('cpu'))

        if i == 0 or (i+1) % max(L//10, 1) == 0:
            if verbose or log:
                grad = -1/(1 - c_alpha).sqrt() * grad
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x_mod.reshape(x_mod.shape[0], -1), dim=-1).mean()
                grad_mean_norm = torch.norm(grad.mean(dim=0).reshape(-1)) ** 2 * (1 - c_alpha)


            # if log:
            #     logging.info("{}: {}/{}, grad_norm: {}, image_norm: {}, grad_mean_norm: {}".format(
            #         "DDPM gamma" if gamma else "DDPM", i+1, L, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))

        # If last step, don't add noise
        last_step = i + 1 == L
        if last_step:
            continue

        # Add noise
        if same_noise:
            noise = noise_val
        else:
            if gamma:
                z = Gamma(torch.full(x_mod.shape[1:], ks_cum[i]),
                          torch.full(x_mod.shape[1:], 1 / thetas[i])).sample((x_mod.shape[0],)).to(x_mod.device)
                noise = (z - ks_cum[i]*thetas[i])/((1 - alphas[i]).sqrt())
            else:
                noise = torch.randn_like(x_mod)
        if just_beta:
            x_mod += c_beta.sqrt() * noise
        else:
            x_mod += ((1 - c_alpha_prev) / (1 - c_alpha) * c_beta).sqrt() * noise
    # Denoise
    if denoise:
        last_noise = ((L - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)).long()
        x_mod = x_mod - (1 - alphas[-1]).sqrt() * scorenet(x_mod, last_noise)
        if not final_only:
            images.append(x_mod.to('cpu'))
    if final_only:
        return x_mod.unsqueeze(0)
    else:
        return torch.stack(images)

class UNetMore_DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.unet = NCSNpp(config)
    self.schedule = getattr(config.model, 'sigma_dist', 'linear')
    if self.schedule == 'linear':
      self.register_buffer('betas', get_sigmas(config))
      self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
    elif self.schedule == 'cosine':
      self.register_buffer('alphas', get_sigmas(config))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
      self.register_buffer('betas', 1 - self.alphas/self.alphas_prev)
    self.gamma = getattr(config.model, 'gamma', False)
    if self.gamma:
        self.theta_0 = 0.001
        self.register_buffer('k', self.betas/(self.alphas*(self.theta_0 ** 2))) # large to small, doesn't match paper, match code instead
        self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0)) # flip for small-to-large, then flip back
        self.register_buffer('theta_t', torch.sqrt(self.alphas)*self.theta_0)

    self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

  def forward(self, x, y, cond=None, cond_mask=None):
    if self.noise_in_cond and cond is not None: # We add noise to cond
      alphas = self.alphas
      # if labels is None:
      #     labels = torch.randint(0, len(alphas), (cond.shape[0],), device=cond.device)
      labels = y
      used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
      if self.gamma:
        used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        z = torch.distributions.gamma.Gamma(used_k, 1 / used_theta).sample()
        z = (z - used_k*used_theta)/(1 - used_alphas).sqrt()
      else:
        z = torch.randn_like(cond)
      cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z
    return self.unet(x, y, cond, cond_mask=cond_mask)

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def reshape_patch_back(patch_tensor, patch_size):
    if patch_tensor.ndim == 4:
        patch_tensor = patch_tensor[np.newaxis, ...]
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels / (patch_size * patch_size)

    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  int(img_channels)])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                int(img_channels)])
    return img_tensor

def reshape_patch_torch(img_tensor, patch_size):
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(0)
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    num_channels = img_tensor.shape[4]

    a = img_tensor.reshape(batch_size, seq_length,
                           int(img_height / patch_size), patch_size,
                           int(img_width / patch_size), patch_size,
                           num_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    patch_tensor = b.reshape(batch_size, seq_length,
                             int(img_height / patch_size),
                             int(img_width / patch_size),
                             patch_size * patch_size * num_channels)
    return patch_tensor.squeeze()

def reshape_patch_back_torch(patch_tensor, patch_size):
    if patch_tensor.ndim == 4:
        patch_tensor = patch_tensor.unsqueeze(0)

    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    channels = patch_tensor.shape[4]
    img_channels = channels // (patch_size * patch_size)

    a = patch_tensor.reshape(batch_size, seq_length, patch_height, patch_width,
                             patch_size, patch_size, img_channels)
    b = a.permute(0, 1, 2, 4, 3, 5, 6)
    img_tensor = b.reshape(batch_size, seq_length, patch_height * patch_size,
                           patch_width * patch_size, img_channels)

    return img_tensor