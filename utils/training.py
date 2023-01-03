import argparse
import logging

import numpy as np
import torch

from utils.data import to_torch, to_np, load_data, sample_metapath

from typing import Tuple, Dict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--command', type=str, default='')
    ap.add_argument('--tag', type=str)
    ap.add_argument('--description', type=str)
    ap.add_argument('--seed_list', nargs='+', type=int)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--sample_limit', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--patience', type=int)
    # LNL
    ap.add_argument('--noise_p', type=float)
    ap.add_argument('--noise_u', type=float)
    # SFT
    ap.add_argument('--sft_filter_memory', type=int)
    ap.add_argument('--sft_filter_warmup', type=int)
    ap.add_argument('--sft_loss_threshold', type=float)
    ap.add_argument('--sft_loss_weights', nargs='+', type=float)
    # MLC
    ap.add_argument('--mlc_virtual_lr', type=float)
    ap.add_argument('--mlc_T_lr', type=float)
    args = ap.parse_args()
    return args


def override_cfg(args: argparse.Namespace, exp_cfg: dict, train_cfg: dict, model_cfg: dict, data_cfg: dict):
    if hasattr(args, 'tag') and args.tag is not None:
        exp_cfg['tag'] = args.tag
    if hasattr(args, 'description') and args.description is not None:
        exp_cfg['description'] = args.description
    if hasattr(args, 'seed_list') and args.seed_list is not None:
        exp_cfg['seed_list'] = args.seed_list
    if hasattr(args, 'dataset') and args.dataset is not None:
        data_cfg['dataset'] = args.dataset
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if hasattr(args, 'sample_limit') and args.sample_limit is not None:
        train_cfg['sample_limit'] = args.sample_limit
    if hasattr(args, 'lr') and args.lr is not None:
        train_cfg['optim_cfg']['lr'] = args.lr
    if hasattr(args, 'patience') and args.patience is not None:
        train_cfg['patience'] = args.patience
    # LNL
    if hasattr(args, 'noise_p') and args.noise_p is not None:
        data_cfg['noise_cfg']['pair_flip_rate'] = args.noise_p
        data_cfg['noise_cfg']['apply'] = True
    if hasattr(args, 'noise_u') and args.noise_u is not None:
        data_cfg['noise_cfg']['uniform_flip_rate'] = args.noise_u
        data_cfg['noise_cfg']['apply'] = True
    # SFT
    if hasattr(args, 'sft_filter_memory') and args.sft_filter_memory is not None:
        train_cfg['sft_cfg']['filtering_cfg']['memory'] = args.sft_filter_memory
        train_cfg['sft_cfg']['apply_filtering'] = True
    if hasattr(args, 'sft_filter_warmup') and args.sft_filter_warmup is not None:
        train_cfg['sft_cfg']['filtering_cfg']['warmup'] = args.sft_filter_warmup
        train_cfg['sft_cfg']['apply_filtering'] = True
    if hasattr(args, 'sft_loss_threshold') and args.sft_loss_threshold is not None:
        train_cfg['sft_cfg']['loss_cfg']['threshold'] = args.sft_loss_threshold
        train_cfg['sft_cfg']['apply_loss'] = True
    if hasattr(args, 'sft_loss_weights') and args.sft_loss_weights is not None:
        train_cfg['sft_cfg']['loss_cfg']['weight'] = args.sft_loss_weights
        train_cfg['sft_cfg']['apply_loss'] = True
    # MLC
    if hasattr(args, 'mlc_virtual_lr') and args.mlc_virtual_lr is not None:
        train_cfg['mlc_cfg']['virtual_lr'] = args.mlc_virtual_lr
        train_cfg['mlc_cfg']['apply'] = True
    if hasattr(args, 'mlc_T_lr') and args.mlc_T_lr is not None:
        train_cfg['mlc_cfg']['T_lr'] = args.mlc_T_lr
        train_cfg['mlc_cfg']['apply'] = True

    dataset = data_cfg['dataset']
    if dataset == 'DBLP':
        train_cfg['patience'] = 5
        train_cfg['optim_cfg']['lr'] = 5e-3
    elif dataset == 'IMDB':
        train_cfg['patience'] = 10
        train_cfg['optim_cfg']['lr'] = 1e-3
    else:
        assert False

    exp_cfg['command'] = args.command
    return


def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return


def get_logger(name, file, verbose=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # %(asctime)s %(levelname)s
    formatter = logging.Formatter('%(name)s %(message)s')

    file_handle = logging.FileHandler(file, mode='w+', encoding='utf-8')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    if verbose:
        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(formatter)
        logger.addHandler(stream_handle)
    return logger


def build_model(
        model_type, model_cfg: dict,
        node_feature_dim_list, num_metapaths, num_node_types,
        device
):
    if model_type == 'HAN':
        from models.HAN import HAN

        model = HAN(
            node_feature_dim_list=node_feature_dim_list,
            num_metapaths=num_metapaths,
            num_cls=num_node_types,
            device=device,
            **model_cfg['cfg']
        )
    elif model_type == 'MLP':
        from models.MLP import MLP

        model = MLP(
            node_raw_feature_dim_list=node_feature_dim_list,
            num_cls=num_node_types,
            device=device,
            **model_cfg
        )
    else:
        assert False
    return model


class Namespace:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        return

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return


class EarlyStopping:
    def __init__(self, patience, criterion: Tuple[str, int], margin, ckpt_file):
        self.patience = patience
        self.criterion = criterion
        self.margin = margin
        self.ckpt_file = ckpt_file

        self.counter = 0
        self.best_record = None
        return

    def record(self, record: Dict[str, float], model):
        flag_improve = True
        if self.best_record is not None:
            if self.criterion[1] > 0 and \
                    record[self.criterion[0]] < self.best_record[self.criterion[0]] - self.margin:
                flag_improve = False
            elif self.criterion[1] < 0 and \
                    record[self.criterion[0]] > self.best_record[self.criterion[0]] + self.margin:
                flag_improve = False
        if flag_improve:
            torch.save(model.state_dict(), self.ckpt_file)
            self.counter = 0
            if self.best_record is None:
                prev_best = -np.inf if self.criterion[1] > 0 else np.inf
            else:
                prev_best = self.best_record[self.criterion[0]]
            new_best = record[self.criterion[0]]
            self.best_record = record
            record_msg = f'{self.criterion[0]} improved {prev_best:.6f} --> {new_best:.6f}. Checkpoint saved.'
        else:
            self.counter += 1
            record_msg = f'EarlyStopping counter {self.counter} / {self.patience}.'
            if self.counter >= self.patience:
                record_msg = 'EarlyStopping. Best record is:'
                return self.best_record, record_msg
        return None, record_msg


class Sampler:
    def __init__(self, metapath_list, all_indices, batch_size, sample_limit, device, shuffle, loop=False):
        self.metapath_list = metapath_list
        self.all_indices = all_indices
        self.num_indices = all_indices.shape[0]
        self.num_batch = batch_size
        self.sample_limit = sample_limit
        self.device = device
        self.shuffle = shuffle
        self.loop = loop
        self.pointer = 0

        self._reset()
        return

    def _reset(self):
        self.pointer = 0
        if self.shuffle:
            np.random.shuffle(self.all_indices)
        return

    def num_iterations(self):
        return int(np.ceil(self.num_indices / self.num_batch))

    def sample(self):
        def sample_indice():
            selected_data = np.copy(self.all_indices[self.pointer:self.pointer + self.num_batch])
            self.pointer += self.num_batch
            if self.pointer >= self.num_indices:
                self._reset()
                if self.loop:
                    self.pointer = self.num_batch - selected_data.shape[0]
                    compensate = np.copy(self.all_indices[:self.pointer])
                    selected_data = np.concatenate([selected_data, compensate], axis=0)

            return selected_data

        indices = sample_indice()
        metapath_sampled_list = sample_metapath(
            indices, self.metapath_list, self.sample_limit, keep_intermediate=False
        )
        indices = to_torch(indices, self.device, indices=True)
        metapath_sampled_list = to_torch(metapath_sampled_list, self.device, indices=True)
        return indices, metapath_sampled_list
