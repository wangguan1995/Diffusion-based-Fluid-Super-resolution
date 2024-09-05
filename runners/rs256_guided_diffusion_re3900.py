import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.diffusion_new import ConditionalModel as CModel
from models.diffusion_new import Model
from train_ddpm.models.diffusion_spatial_temperal import Model_Spatial_Temperal

from functions.process_data import *
from functions.denoising_step import guided_ddpm_steps, guided_ddim_steps, ddpm_steps, ddim_steps

import matplotlib.pyplot as plt
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import pickle

from copy import deepcopy



def plot_data(u_2d, u_2d_downsampled):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.imshow(u_2d, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d.png')
    plt.clf()
    plt.imshow(u_2d_downsampled, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d_downsampled.png')
    plt.clf()


class MetricLogger(object):
    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @torch.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=''):
        with open(os.path.join(outdir, f'metric_log_{postfix}.pkl'), 'wb') as f:
            pickle.dump(self.metric_dict, f)


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


@torch.no_grad()
def patchify_new(data_input):
    data = data_input.cpu().numpy()
    block_size = 40
    # 补0， data.shape = [time, dim, nx, ny]
    n_x = int(np.ceil(data.shape[2] / block_size) * block_size)
    n_y = int(np.ceil(data.shape[3] / block_size) * block_size)
    expanded_matrix = np.zeros((data.shape[0], data.shape[1], n_x, n_y), dtype=np.float)  
    expanded_matrix[:, :, :data.shape[2], :data.shape[3]] = data  
    # split image into 4*10 patches and concatenate them
    # 切分图像为4*10的子图像，并拼接
    sub_patches_list = []
    for row_blocks in np.split(expanded_matrix, int(n_x / block_size), axis=2):   # 2 rows
        for block in np.split(row_blocks, int(n_y / block_size), axis=3):       # 2 cols  
            sub_patches_list.append(block)
    
    sub_patches = np.stack(sub_patches_list, axis=0)
    return sub_patches

@torch.no_grad()
def patchify_old(data):
    data = data.cpu().numpy()
    block_size = 40
    # 补0， data.shape = [time, dim, nx, ny]
    n_x = int(np.ceil(data.shape[2] / block_size) * block_size)
    n_y = int(np.ceil(data.shape[3] / block_size) * block_size)
    expanded_matrix = np.zeros((data.shape[0], data.shape[1], n_x, n_y), dtype=np.float)  
    expanded_matrix[:, :, :data.shape[2], :data.shape[3]] = data  
    # split image into 4*10 patches and concatenate them
    # 切分图像为4*10的子图像，并拼接
    sub_matrices = []
    for row_blocks in np.split(expanded_matrix, int(n_x / block_size), axis=2):   # 2 rows
        for block in np.split(row_blocks, int(n_y / block_size), axis=3):       # 2 cols  
            sub_matrices.append(block)
    return np.concatenate(sub_matrices, axis=0)


def load_recons_data(ref_path, sample_data_dir, stat_path, smoothing, smoothing_scale, train_ratio=0.9):
    flattened_ref_data = np.load(ref_path)
    flattened_sampled_data = np.load(sample_data_dir)
    num_train = flattened_sampled_data.shape[0]
    num_train = int(train_ratio * num_train)
    flattened_ref_data = flattened_ref_data[num_train:]
    flattened_sampled_data = flattened_sampled_data[num_train:]
    stat = np.load(stat_path)
    data_mean, data_scale = np.mean(flattened_ref_data), np.std(flattened_ref_data)
    data_mean, data_scale = stat["mean"], stat["scale"]
    print(f"Data statistics, mean: {data_mean}, scale: {data_scale}")
    flattened_ref_data = torch.Tensor(flattened_ref_data)
    flattened_sampled_data = torch.Tensor(flattened_sampled_data)

    if smoothing:
        arr = flattened_sampled_data
        ker_size = smoothing_scale
        # peridoic padding
        arr = F.pad(arr,
                    pad=((ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2),
                    mode='circular', )
        arr = transforms.GaussianBlur(kernel_size=ker_size, sigma=ker_size)(arr)# F.avg_pool2d(arr, (ker_size, ker_size), stride=1, count_include_pad=False)
        flattened_sampled_data = arr[..., (ker_size - 1) // 2:-(ker_size - 1) // 2, (ker_size - 1) // 2:-(ker_size - 1) // 2]

    # ref_data, blur_data, data_mean, data_std
    return flattened_ref_data, flattened_sampled_data, data_mean.item(), data_scale.item()



class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) #/ (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def nearest_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = data[:, :, ::scale, ::scale]

    return blur_data


def gaussian_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = transforms.GaussianBlur(kernel_size=scale, sigma=2*scale+1)(data)

    return blur_data


def random_square_hole_mask(data, hole_size):
    # generate a random square hole mask
    h, w = data.shape[2:]
    mask = torch.zeros(data.shape, dtype=torch.int64).to(data.device)
    hole_x = np.random.randint(0, w - hole_size)
    hole_y = np.random.randint(0, h - hole_size)
    mask[..., hole_y:hole_y+hole_size, hole_x:hole_x+hole_size] = 1

    return mask


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def slice2sequence(data):
    #这段代码是使用einops库的rearrange函数来重新排列多维数组data。
    # 首先，data[:, 1:2]是将data数组的第二维（纵向）取一个切片，得到一个新的数组。
    # 然后，rearrange(data[:, 1:2], 't f h w -> (t f) h w')将这个新数组重新排列。 't f h w -> (t f) h w'是一个操作字符串，表示原来数组的维度是t, f, h, w四个维度，然后将t和f两个维度合并为一个新的维度（即在这两个维度上做了flatten操作），h和w两个维度保持不变。
    # 比如，如果原来data的维度是(10, 2, 32, 32)，那么data[:, 1:2]得到的新数组维度是(10, 1, 32, 32)，然后经过rearrange操作后，最终得到的数组data的维度会是(10, 32, 32)。
    data = rearrange(data[:, 0:1], 't f h w -> (t f) h w')
    return data


def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y):
    l2_list = []
    for i in range(x.shape[0]):
        l2 = ((x[i] - y[i])**2).mean((-1, -2)).sqrt().mean()
        l2_list.append(float(l2))
    return np.mean(l2_list)


def voriticity_residual(w, re=1000.0, dt=1/32, calc_grad=True):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    residual_loss = (residual**2).mean()
    if calc_grad:
        dw = torch.autograd.grad(residual_loss, w)[0]
        return dw, residual_loss
    else:
        return residual_loss


class Diffusion_Re3900(object):
    def __init__(self, args, config, logger, log_dir, device=None):
        self.args = args
        self.config = config
        self.logger = logger
        self.image_sample_dir = log_dir

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def log(self, info):
        self.logger.info(info)

    def make_image_grid(self, images, out_path, ncols=4, batch_index=1):
        bs, t, c, h, w = 0, 1, 0, 0, 0
        if len(images.shape) == 4:
            bs, c, h, w = images.shape
            images = images[:,:,None,:,:]
        elif len(images.shape) == 5:
            bs, t, c, h, w = images.shape
        else:
            raise NotImplementedError
        images = images[:, :, 0, :, :] # plot velocity in x direction
        for i in range(t):
            image = images[:, i, :, :] # plot velocity in x direction int time step i
            time = int(batch_index * t + i)
            out_path_new = out_path.replace('.png', '_t_{}.png'.format(time))
            b = bs // ncols
            fig = plt.figure(figsize=(8., 8.))
            grid = ImageGrid(fig, 111, nrows_ncols=(b, ncols))
            for ax, im_no in zip(grid, np.arange(b*ncols)):
                ax.imshow(image[im_no, :, :], cmap='twilight', vmin=self.ref_data_min, vmax=self.ref_data_max)
                ax.axis('off')

            plt.savefig(out_path_new, bbox_inches='tight', dpi=100)
            plt.close()

    def reconstruct(self):
        if self.config.model.type == 'conditional':
            model = CModel(self.config)
            raise NotImplementedError
        elif self.config.model.type == 'spatial_temperal':
            model = Model_Spatial_Temperal(self.config)
            patchify = patchify_new
        else:
            patchify = patchify_old
            model = Model(self.config)

        model.load_state_dict(torch.load(self.config.model.ckpt_path)[-1])
        model.to(self.device)
        model.eval()
        ref_data, blur_data, data_mean, data_std = load_recons_data(self.config.data.data_dir,
                                                                    self.config.data.sample_data_dir,
                                                                    self.config.data.stat_path,
                                                                    smoothing=self.config.data.smoothing,
                                                                    smoothing_scale=self.config.data.smoothing_scale)
        patch_row_num = int(np.ceil(ref_data.shape[2] / 40))
        patch_col_num = int(np.ceil(ref_data.shape[3] / 40))
        scaler = StdScaler(data_mean, data_std)
        self.ref_data_min = ref_data.min()
        self.ref_data_max = ref_data.max()

        # pack data loader
        testset = torch.utils.data.TensorDataset(blur_data, ref_data)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.config.sampling.batch_size,
                                                  shuffle=False, num_workers=self.config.data.num_workers, drop_last=True)

        l2_loss_predict_all = np.zeros((ref_data.shape[0], self.args.sample_step))
        l2_loss_reference_all = np.zeros((ref_data.shape[0], self.args.sample_step))
        residual_loss_all = np.zeros((ref_data.shape[0], self.args.sample_step))

        for batch_index, (blur_data, data) in enumerate(test_loader):
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))
            x0 = blur_data.to(self.device)  # x0 : 低精度 输入数据， batch_size=20
            gt = data.to(self.device)       # gt : 高精度 参考数据， batch_size=20
            x0_masked = x0.clone()
            # self.make_image_grid(patchify(x0_masked), self.image_sample_dir + '/input_image.png', patch_col_num, batch_index)
            # self.make_image_grid(patchify(gt), self.image_sample_dir + '/reference_image.png', patch_col_num, batch_index)
            l2_loss_init = l2_loss(x0, gt)                   
            l2_loss_reference_all[batch_index] = (l2_loss_init.item())         
            self.log('L2 loss init: {}'.format(l2_loss_init))
            x0 =  (x0)
            
            # prepare loss function
            if self.config.sampling.log_loss:
                l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to(gt.device), gt)
                # equation_loss_fn = lambda x: voriticity_residual(scaler.inverse(x),
                                                                #  calc_grad=False)

                logger = MetricLogger({
                    'l2 loss': l2_loss_fn,
                    # 'residual loss': equation_loss_fn
                })

            def model_forward(x):
                x = patchify(x.clone())
                x = torch.Tensor(x).cuda()
                logger = None
                if self.config.model.type == 'conditional':
                    xs, _ = guided_ddim_steps(x, seq, model, betas,
                                            w=self.config.sampling.guidance_weight,
                                            dx_func=physical_gradient_func, cache=False, logger=logger)
                elif self.config.sampling.lambda_ > 0:
                    xs, _ = ddim_steps(x, seq, model, betas,
                                    dx_func=physical_gradient_func, cache=False, logger=logger)
                elif self.config.model.type == 'spatial_temperal':
                    x = x.permute((0,2,1,3,4)) # time and frame id in 3rd channel
                    xs, _ = ddim_steps(x, seq, model, betas, cache=False, logger=logger)
                    xs[-1] = xs[-1].permute((0,2,1,3,4)) # time and frame id in 3rd channel
                    xs = torch.cat([xs[-1][i:i+patch_col_num] for i in range(0, patch_row_num * patch_col_num, patch_col_num)], dim=3) 
                    xs = torch.cat([xs[i:i+1] for i in range(patch_col_num)], dim=4)
                    xs = xs[0]
                else:
                    xs, _ = ddim_steps(x, seq, model, betas, cache=False, logger=logger)
                    xs = torch.cat([xs[-1][i:i+patch_col_num] for i in range(0, patch_row_num * patch_col_num, patch_col_num)], dim=2) 
                    xs = torch.cat([xs[i:i+1] for i in range(patch_col_num)], dim=3)
                return xs

            for it in range(self.args.sample_step):  # we run the sampling for three times
                e = torch.randn_like(x0)
                total_noise_levels = int(self.args.t * (0.7 ** it)) 
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                num_of_reverse_steps = int(self.args.reverse_steps * (0.7 ** it ))
                if num_of_reverse_steps == 0:
                    num_of_reverse_steps = 1
                betas = self.betas.to(self.device)
                skip = total_noise_levels // num_of_reverse_steps
                seq = range(0, total_noise_levels, skip)
                xs = model_forward(x)
                x = xs[:,:,:ref_data.shape[2],:ref_data.shape[3]]
                x0 = xs.cuda()
                l2_loss_f = l2_loss(scaler.inverse(x.clone()).to(gt.device), gt)
                l2_loss_predict_all[batch_index * x.shape[0]:(batch_index + 1) * x.shape[0], it] = l2_loss_f.item()
                self.log('L2 loss it{}: {}'.format(it, l2_loss_f))
            self.make_image_grid(scaler.inverse(patchify(x)), self.image_sample_dir + f"/predict.png", patch_col_num, batch_index)
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
        self.log(f'mean l2 reference loss: {l2_loss_reference_all[..., -1].mean()}')
        self.log(f'mean l2 predict   loss: {l2_loss_predict_all[..., -1].mean()}')
