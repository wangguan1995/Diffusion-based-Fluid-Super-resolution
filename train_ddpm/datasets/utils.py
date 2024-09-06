import os
import os.path
import hashlib
import errno
from torch.utils.model_zoo import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import glob as glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()

def normalize_array(x):
    # x: input numpy array with shape (W, H)
    x_min = np.amin(x)
    x_max = np.amax(x)
    y = (x - x_min) / (x_max - x_min)
    return y, x_min, x_max

def unnormalize_array(y, x_min, x_max):
    return y * (x_max - x_min) + x_min

def data_blurring(data_sample, us_size):
    # data_sample: torch tensor, size: [128, 128, 3]
    # us_size: image upscale size: 64
    # return: blurred torch tensor, size: [64, 64, 3]

    ds_size = 16
    resample_method = Image.NEAREST

    x_array, x_min, x_max = normalize_array(data_sample.numpy())
    im = Image.fromarray((x_array*255).astype(np.uint8))
    im_ds = im.resize((ds_size, ds_size))
    im_us = im_ds.resize((us_size, us_size), resample=resample_method)
    x_array_blur = np.asarray(im_us)

    # inverse-transform to the original value range
    x_array_blur = x_array_blur.astype(np.float32)/255.0
    x_array_blur = unnormalize_array(x_array_blur, x_min, x_max)

    return torch.from_numpy(x_array_blur)

def data_preprocessing(target, Image_Size):
    # Convert 128 x 128 target data to 64 x 64
    # reduce the resolution of target data to create 64 x 64 img data
    # target: torch tensor, size: [BS, 3, 128, 128]
    # Image_Size: 64 by default
    # img: torch tensor, size: [BS, 3, Image_Size, Image_Size]
    # output_target: torch tensor, size: [BS, 3, Image_Size, Image_Size]

    img = torch.zeros(target.size(0), target.size(1), Image_Size, Image_Size) # size: [32, 3, 64, 64]
    for idx in range(target.size(0)):
        x = target[idx]
        x = torch.permute(x, [1,2,0]) # x size: [128, 128, 3]
        x = data_blurring(x, Image_Size) # x size: [64, 64, 3]
        img[idx] =  torch.permute(x, [2,0,1])

    down_scale_transform = transforms.Resize(Image_Size)
    output_target = down_scale_transform(target)

    return img, output_target

# Create a customized dataset class for fno dataset

import torch
from torch.utils.data import Dataset

class FNO_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data = torch.load(data_dir)
        self.data = torch.permute(self.data, [0,3,1,2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KMFlowDataset(Dataset):
    def __init__(self, data_dir, resolution=256, max_cache_len=3200,
                 inner_steps=32, outer_steps=10, train_ratio=0.9, test=False,
                 stat_path=None):
        fname_lst = glob.glob(data_dir + '/seed*')
        np.random.seed(1)
        num_of_training_samples = int(train_ratio*len(fname_lst))
        np.random.shuffle(fname_lst)
        self.train_fname_lst = fname_lst[:num_of_training_samples]
        self.test_fname_lst = fname_lst[num_of_training_samples:]

        if not test:
            self.fname_lst = self.train_fname_lst[:]
        else:
            self.fname_lst = self.test_fname_lst[:]

        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.resolution = resolution
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
            self.scaler = StandardScaler()
            self.scaler.mean_ = self.stat['mean']
            self.scaler.scale_ = self.stat['scale']
        else:
            self.prepare_data()
        self.cache = {}

    def __len__(self):
        return len(self.fname_lst) * (self.inner_steps * self.outer_steps - 2)

    def prepare_data(self):
        # load all training data and calculate their statistics
        self.scaler = StandardScaler()
        for data_dir in tqdm(self.fname_lst):
            for i in range(self.outer_steps):
                for j in range(0, self.inner_steps, 4):
                    fname = os.path.join(data_dir,
                                         f'sol_t{i}_step{j}.npy')
                    data = np.load(fname, mmap_mode='r')[::4, ::4]
                    data = data.reshape(-1, 1)
                    self.scaler.partial_fit(data)
                    del data

        print(f'Data statistics, mean: {self.scaler.mean_}, standard deviation: {self.scaler.scale_}')

    def preprocess_data(self, data):
        # normalize data

        s = data.shape[0]
        sub = int(s // self.resolution)
        data = data[::sub, ::sub]

        data = self.scaler.transform(data.reshape(-1, 1)).reshape((self.resolution, self.resolution))
        return data

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.scaler.mean_, scale=self.scaler.scale_)

    def __getitem__(self, idx):
        seed = idx // (self.inner_steps * self.outer_steps - 2)
        frame_idx = idx % (self.inner_steps * self.outer_steps - 2)
        if frame_idx % self.inner_steps == 31:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step + 1
            next_next_outer_step = next_outer_step
            next_inner_step = 0
            next_next_inner_step = 1
        elif frame_idx % self.inner_steps == 30:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step
            next_next_outer_step = next_outer_step + 1
            next_inner_step = inner_step + 1
            next_next_inner_step = 0
        else:
            inner_step = frame_idx % self.inner_steps
            outer_step = frame_idx // self.inner_steps
            next_outer_step = outer_step
            next_next_outer_step = next_outer_step
            next_inner_step = inner_step + 1
            next_next_inner_step = next_inner_step + 1

        id = f'seed{seed}_t{outer_step}_step{inner_step}'

        if id in self.cache.keys():
            return self.cache[id]
        else:
            data_dir = self.fname_lst[seed]
            fname0 = os.path.join(data_dir,
                                 f'sol_t{outer_step}_step{inner_step}.npy')
            frame0 = np.load(fname0, mmap_mode='r')
            frame0 = self.preprocess_data(frame0)

            fname1 = os.path.join(data_dir,
                                    f'sol_t{next_outer_step}_step{next_inner_step}.npy')
            frame1 = np.load(fname1, mmap_mode='r')
            frame1 = self.preprocess_data(frame1)

            fname2 = os.path.join(data_dir,
                                    f'sol_t{next_next_outer_step}_step{next_next_inner_step}.npy')
            frame2 = np.load(fname2, mmap_mode='r')
            frame2 = self.preprocess_data(frame2)

            frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)
            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                self.cache.pop(np.random.choice(self.cache.keys()))
            return frame


class KMFlowTensorDataset(Dataset):
    def __init__(self, config, data_path, train_ratio=0.9, test=False, stat_path=None, max_cache_len=4000,):
        np.random.seed(1)
        all_data_list = []
        if isinstance(data_path, list):
            for path in data_path:
                data = np.load(path)
                num_train = data.shape[0]
                num_train = int(train_ratio * num_train)
                data = data[:num_train]
                all_data_list.append(data)
            all_data = np.concatenate(all_data_list, axis=0)
            self.all_data = self.patchify(all_data)
        else:
            self.all_data = np.load(data_path)
            num_train = self.all_data.shape[0]
            num_train = int(train_ratio * num_train)
            self.all_data = self.all_data[:num_train]
            self.all_data = self.patchify(self.all_data)
        print('Data set shape: ', self.all_data.shape)
        self.stat = self.prepare_data()

    def __len__(self):
        return self.all_data.shape[0]

    def prepare_data(self):
        stat = {
            'mean': np.mean(self.all_data.reshape(-1, 1)),
            'scale': np.std(self.all_data.reshape(-1, 1)),
        }
        print(f"Data statistics, mean: {stat['mean']}, scale: {stat['scale']}")
        return stat

    def patchify(self, data):
        data = data
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

    def preprocess_data(self, data):
        # normalize data
        s = data.shape[-1]
        data = (data - self.stat['mean']) / (self.stat['scale'])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        frame0 = self.preprocess_data(self.all_data[idx, 0])
        frame1 = self.preprocess_data(self.all_data[idx, 1])
        frame2 = self.preprocess_data(self.all_data[idx, 2])
        frame = np.concatenate((frame0[None, ...], frame1[None, ...], frame2[None, ...]), axis=0)
        return frame


class KMFlowTensorDataset_ST(Dataset):
    def __init__(self, config, data_path, test=False, stat_path=None, max_cache_len=4000):
        np.random.seed(1)
        self.frame_number = config.data.frame_number
        self.block_size = config.data.block_size
        self.all_data = np.load(data_path)
        self.all_data = self.patchify_spatial_temperal(self.all_data)
        print('Data set shape: ', self.all_data.shape)
        idxs = np.arange(self.all_data.shape[1]-self.frame_number)
        train_number = int(config.data.train_ratio*len(idxs))
        self.train_time_index_list = idxs[:train_number]
        self.test_time_index_list = idxs[train_number:]
        self.spatial_index_list = np.arange(self.all_data.shape[0])
        if not test:
            self.idx_lst = self.train_time_index_list[:]
        else:
            self.idx_lst = self.test_time_index_list[:]
        self.cache = {}
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        return len(self.idx_lst) * len(self.spatial_index_list)

    def prepare_data(self):
        # load all training data and calculate their statistics
        self.stat['mean'] = np.mean(self.all_data[:,self.train_time_index_list[:]].reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[:,self.train_time_index_list[:]].reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')

    def patchify_spatial_temperal(self, data):
        data = data
        # data.shape = [time, dim, nx, ny]
        n_x = int(np.ceil(data.shape[2] / self.block_size) * self.block_size)
        n_y = int(np.ceil(data.shape[3] / self.block_size) * self.block_size)
        expanded_matrix = np.zeros((data.shape[0], data.shape[1], n_x, n_y), dtype=np.float)  
        expanded_matrix[:, :, :data.shape[2], :data.shape[3]] = data  
        sub_patches_list = []
        for row_blocks in np.split(expanded_matrix, int(n_x / self.block_size), axis=2):   # 2 rows
            for block in np.split(row_blocks, int(n_y / self.block_size), axis=3):       # 2 cols  
                sub_patches_list.append(block)
        
        
        sub_patches = np.stack(sub_patches_list)
        # data.shape = [bs, time, dim, nx, ny]
        return sub_patches

    def preprocess_data(self, data):
        # normalize data

        s = data.shape[-1]

        data = (data - self.stat['mean']) / (self.stat['scale'])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        frame_idx = self.idx_lst[idx // len(self.spatial_index_list)]
        seed = idx % len(self.spatial_index_list)
        id = idx

        if id in self.cache.keys():
            return self.cache[id]
        else:
            frame_list = []
            for i in range(self.frame_number):
                frame_i = self.all_data[seed, frame_idx + i]
                frame_i = self.preprocess_data(frame_i)
                frame_list.append(frame_i)
            frame = np.stack(frame_list)
            # [time, uvw, x, y] -> [uvw, time, x, y]
            frame = np.transpose(frame, (1, 0, 2, 3)) 
            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                self.cache.pop(list(self.cache.keys())[np.random.choice(len(self.cache.keys()))])
            return frame


class KMFlowTensorDataset_DIT(Dataset):
    def __init__(self, config, data_path,
                 train_ratio=0.9, test=False,
                 stat_path=None,
                 max_cache_len=4000,
                 patch_size=40,
                 frame_number=10,
                 ):
        np.random.seed(1)
        self.patch_size=patch_size
        self.frame_number=frame_number
        self.all_data = np.load(data_path)
        self.all_data = self.pad_with_zeros(self.all_data, self.patch_size)
        print("all_data.shape", self.all_data.shape)
        idxs = np.arange(self.all_data.shape[0])
        num_of_training_seeds = int(train_ratio*len(idxs))
        # np.random.shuffle(idxs)
        self.train_time_index_list = idxs[:num_of_training_seeds]
        self.test_time_index_list = idxs[num_of_training_seeds:]
        if not test:
            self.idx_lst = self.train_time_index_list[:]
        else:
            self.idx_lst = self.test_time_index_list[:]
        self.cache = {}
        self.max_cache_len = max_cache_len

        if stat_path is not None:
            self.stat_path = stat_path
            self.stat = np.load(stat_path)
        else:
            self.stat = {}
            self.prepare_data()

    def __len__(self):
        return len(self.idx_lst)

    def prepare_data(self):
        # load all training data and calculate their statistics
        self.stat['mean'] = np.mean(self.all_data[self.train_time_index_list[:]].reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[self.train_time_index_list[:]].reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']
        print(f'Data statistics, mean: {data_mean}, scale: {data_scale}')

    def pad_with_zeros(self, data, patch_size):
        # 补0， data.shape = [time, dim, nx, ny]
        n_x = int(np.ceil(data.shape[2] / patch_size) * patch_size)
        n_y = int(np.ceil(data.shape[3] / patch_size) * patch_size)
        expanded_matrix = np.zeros((data.shape[0], data.shape[1], n_x, n_y), dtype=np.float)  
        expanded_matrix[:, :, :data.shape[2], :data.shape[3]] = data  
        return expanded_matrix

    def preprocess_data(self, data):
        # normalize data

        s = data.shape[-1]

        data = (data - self.stat['mean']) / (self.stat['scale'])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        id = idx
        if id in self.cache.keys():
            return self.cache[id]
        else:
            frame_list = []
            for i in range(self.frame_number):
                frame_i = self.all_data[idx + i]
                frame_i = self.preprocess_data(frame_i)
                frame_list.append(frame_i)
            frame = np.stack(frame_list)
            # [time, uvw, x, y] -> [uvw, time, x, y]
            frame = np.transpose(frame, (1, 0, 2, 3))
            self.cache[id] = frame

            if len(self.cache) > self.max_cache_len:
                self.cache.pop(list(self.cache.keys())[np.random.choice(len(self.cache.keys()))])
            return frame


class KMFlowTensorDataset_DiT(Dataset):
    def __init__(self, config, data_path, train_ratio=0.9, test=False, stat_path=None, max_cache_len=4000,):
        np.random.seed(1)
        self.all_data = np.load(data_path)
        num_train = self.all_data.shape[0]
        num_train = int(train_ratio * num_train)
        self.all_data = self.all_data[:num_train]
        print('Data set shape: ', self.all_data.shape)
        self.stat = self.prepare_data()

    def __len__(self):
        return self.all_data.shape[0]

    def prepare_data(self):
        stat = {
            'mean': np.mean(self.all_data.reshape(-1, 1)),
            'scale': np.std(self.all_data.reshape(-1, 1)),
        }
        print(f"Data statistics, mean: {stat['mean']}, scale: {stat['scale']}")
        return stat

    def preprocess_data(self, data):
        # normalize data
        s = data.shape[-1]
        data = (data - self.stat['mean']) / (self.stat['scale'])
        return data.astype(np.float32)

    def save_data_stats(self, out_dir):
        # save data statistics to out_dir
        np.savez(out_dir, mean=self.stat['mean'], scale=self.stat['scale'])

    def __getitem__(self, idx):
        return self.all_data[idx]
