import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model, ConditionalModel
from models.diffusion_spatial_temperal import Model_Spatial_Temperal
from models.DIT import DiT_models
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.denoising import generalized_steps, ddim_steps
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

from tensorboardX import SummaryWriter

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datasets.utils import KMFlowTensorDataset, KMFlowTensorDataset_ST, KMFlowTensorDataset_DIT, KMFlowTensorDataset_DiT

torch.manual_seed(0)
np.random.seed(0)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()


def load_recons_data(ref_path, sample_data_dir, data_kw, smoothing, smoothing_scale):
    flattened_ref_data = np.load(ref_path)
    flattened_sampled_data = np.load(sample_data_dir)

    data_mean, data_scale = np.mean(flattened_ref_data), np.std(flattened_ref_data)
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


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self.logger = args.logger

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if config.model.type == "simple":
            train_dataset = KMFlowTensorDataset
        elif config.model.type == "spatial_temperal":
            train_dataset = KMFlowTensorDataset_ST
        elif config.model.type == "DIT":
            train_dataset = KMFlowTensorDataset_DiT
        else:
            raise NotImplementedError(f"{config.model.type} not implemented!")

        # Load training and test datasets
        if os.path.exists(config.data.stat_path):
            print("Loading dataset statistics from {}".format(config.data.stat_path))
            train_data = train_dataset(config, config.data.data_dir, stat_path=config.data.stat_path)
        else:
            print("No dataset statistics found. Computing statistics...")
            train_data = train_dataset(config, config.data.data_dir, )
            train_data.save_data_stats(config.data.stat_path)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.training.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers)
        if config.model.type == "simple":
            model = Model(config)
        elif config.model.type == "spatial_temperal":
            model = Model_Spatial_Temperal(config)
        elif config.model.type == "DIT":
            model = DiT_models['DiT-S/8']()
        else:
            raise NotImplementedError(f"{config.model.type} not implemented!")

        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.config.training.n_epochs//1, gamma=0.1)
        lrs = []

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        writer = SummaryWriter()
        num_iter = 0
        log_freq = 100
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss = []
            for i, x in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)  # size: [32, 3, 256, 256]
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                epoch_loss.append(loss.item())

                tb_logger.add_scalar("loss", loss, global_step=step)

                if num_iter % log_freq == 0:
                    logging.info(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
                    )
                #
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('data_time', data_time / (i + 1), step)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
                num_iter = num_iter + 1
            scheduler.step()
            lrs.append([group['lr'] for group in optimizer.param_groups])
            text_message = "Epoch: {}/{}, Loss: {}, Lr: {}".format(epoch, self.config.training.n_epochs, np.mean(epoch_loss), lrs[-1][0])
            writer.add_text('Training_Info', text_message, epoch)
            print(text_message)
        torch.save(
            states,
            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
        )
        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        print("Model saved at: ", self.args.log_path + "ckpt_{}.pth".format(step))

        writer.export_scalars_to_json("./runs/all_scalars.json")
        writer.close()

    def sample(self):
        # do nothing
        # leave the sampling procedure to sdeit
        pass

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                    torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                    + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i: i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass

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

            plt.savefig(out_path_new, bbox_inches='tight', dpi=h)
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
                                                                    self.config.data.data_kw,
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

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step))
        residual_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step))


        for batch_index, (blur_data, data) in enumerate(test_loader):
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))
            x0 = blur_data.to(self.device)  # x0 : 低精度 输入数据， batch_size=20
            gt = data.to(self.device)       # gt : 高精度 参考数据， batch_size=20
            x0_masked = x0.clone()
            self.make_image_grid(patchify(x0_masked), self.image_sample_dir + '/input_image.png', patch_col_num, batch_index)
            self.make_image_grid(patchify(gt), self.image_sample_dir + '/reference_image.png', patch_col_num, batch_index)

            # calculate initial loss
            #l1_loss_init = l1_loss(x0, gt)
            # l2_loss_init : 进入神经网络前，【低精度输入数据 x0】和【高精度参考数据 gt】之间的l2误差
            l2_loss_init = l2_loss(x0, gt)                          

            self.log('L2 loss init: {}'.format(l2_loss_init))
            # gt_residual : 进入神经网络前，【高精度参考数据 gt】的NS方程残差
            # gt_residual = voriticity_residual(gt)[1].detach()
            # init_residual : 进入神经网络前，【低精度参考数据 x0】的NS方程残差
            # init_residual = voriticity_residual(x0)[1].detach()

            x0 = scaler(x0)
            xinit = x0.clone()
            
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

            # we repeat the sampling for multiple times
            for repeat in range(self.args.repeat_run):
                self.log(f'Run No.{repeat}:')
                x0 = xinit.clone()
                for it in range(self.args.sample_step):  # we run the sampling for three times

                    e = torch.randn_like(x0)

                    # [self.args.t] means denosing step number
                    # total_noise_levels.max = int(0.7 * self.args.t)
                    total_noise_levels = int(self.args.t * (0.7 ** it)) 

                    a = (1 - self.betas).cumprod(dim=0)
                    # a[total_noise_levels - 1] means alpha_t_hat in paper, a.shape = (1000,1)
                    # 根据扩散模型的公式，计算当前步骤的噪声数据。x0是原始数据，e是添加的噪声，a[total_noise_levels - 1]表示当前步骤数据保留的比例。
                    x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                    # if self.config.model.type == 'conditional':
                    #     physical_gradient_func = lambda x: voriticity_residual(scaler.inverse(x))[0] / scaler.scale()
                    # elif self.config.sampling.lambda_ > 0:
                    #     physical_gradient_func = lambda x: \
                    #         voriticity_residual(scaler.inverse(x))[0] / scaler.scale() * self.config.sampling.lambda_
                    num_of_reverse_steps = int(self.args.reverse_steps * (0.7 ** it ))
                    if num_of_reverse_steps == 0:
                        num_of_reverse_steps = 1
                    betas = self.betas.to(self.device)
                    # 计算每一步去噪时跳过的噪声水平数，以便减少计算量。
                    skip = total_noise_levels // num_of_reverse_steps
                    # 生成一个序列，表示去噪过程中每一步的噪声水平索引。
                    seq = range(0, total_noise_levels, skip)
                    xs = model_forward(x)
                    x = xs[:,:,:ref_data.shape[2],:ref_data.shape[3]]
                    x0 = xs.cuda()
                    l2_loss_f = l2_loss(scaler.inverse(x.clone()).to(gt.device), gt)
                    self.log('L2 loss it{}: {}'.format(it, l2_loss_f))
                    # residual_loss_f = voriticity_residual(scaler.inverse(x.clone()), calc_grad=False).detach()
                    # self.log('Residual it{}: {}'.format(it, residual_loss_f))

                    # l2_loss_log[f'run_{repeat}'].append(l2_loss_f.item())
                    # residual_loss_log[f'run_{repeat}'].append(residual_loss_f.item())
                    l2_loss_all[batch_index * x.shape[0]:(batch_index + 1) * x.shape[0], repeat, it] = l2_loss_f.item()
                    # residual_loss_all[batch_index * x.shape[0]:(batch_index + 1) * x.shape[0], repeat,
                    # it] = residual_loss_f.item()

                    if self.config.sampling.log_loss:
                        logger.log(self.image_sample_dir, f'run_{repeat}_it{it}')
                        logger.reset()
                    x_split = patchify(x)
                self.make_image_grid(scaler.inverse(x_split), self.image_sample_dir + f"/predict.png", patch_col_num, batch_index)
            # with open(os.path.join(self.image_sample_dir, sample_folder, f'log.pkl'), 'wb') as f:
            #     pickle.dump({'l2 loss': l2_loss_log, 'residual loss': residual_loss_log, 'bicubic': bicubic_log}, f)
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
        self.log(f'mean l2 loss: {l2_loss_all[..., -1].mean()}')
        self.log(f'std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}')
        self.log(f'mean residual loss: {residual_loss_all[..., -1].mean()}')
        self.log(f'std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}')


class ConditionalDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # Load training and test datasets
        if os.path.exists(config.data.stat_path):
            print("Loading dataset statistics from {}".format(config.data.stat_path))
            train_data = KMFlowTensorDataset(config.data.data_dir, stat_path=config.data.stat_path)
        else:
            print("No dataset statistics found. Computing statistics...")
            train_data = KMFlowTensorDataset(config.data.data_dir, )
            train_data.save_data_stats(config.data.stat_path)
        x_offset, x_scale = train_data.stat['mean'], train_data.stat['scale']
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.training.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers)

        model = ConditionalModel(config)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(num_params)
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        writer = SummaryWriter()
        num_iter = 0
        log_freq = 100
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            epoch_loss = []
            for i, x in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)  # size: [32, 3, 256, 256]
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b, x_offset.item(), x_scale.item())

                epoch_loss.append(loss.item())

                tb_logger.add_scalar("loss", loss, global_step=step)

                if num_iter % log_freq == 0:
                    logging.info(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
                    )
                #
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('data_time', data_time / (i + 1), step)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
                num_iter = num_iter + 1
            print("==========================================================")
            print("Epoch: {}/{}, Loss: {}".format(epoch, self.config.training.n_epochs, np.mean(epoch_loss)))
        print("Finished training")

        torch.save(
            states,
            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
        )
        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        print("Model saved at: ", self.args.log_path + "ckpt_{}.pth".format(step))

        writer.export_scalars_to_json("./runs/all_scalars.json")
        writer.close()

    def sample(self):
        # do nothing
        # leave the sampling procedure to sdeit
        pass

    def sample_sequence(self, model):
        pass

    def sample_interpolation(self, model):
        pass

    def sample_image(self, x, model, last=True):
        pass

    def test(self):
        pass
