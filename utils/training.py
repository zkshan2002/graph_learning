
import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils.data import to_np, MetapathDataset, Sampler
from utils.evaluate import evaluate_multiclass

from typing import Tuple, Dict




def get_cfg(arg: dict):
    from config import cfg

    def override(dst: dict, src: dict):
        for key, value in src.items():
            if isinstance(value, dict):
                override(dst[key], src[key])
            else:
                dst[key] = value

    cfg = copy.deepcopy(cfg)
    override(cfg, arg)

    dataset = cfg['data']['dataset']
    if dataset == 'DBLP':
        cfg['train']['early_stop_cfg']['warmup'] = 5
        cfg['train']['early_stop_cfg']['patience'] = 5
        cfg['train']['optim_cfg']['lr'] = 5e-3
    elif dataset == 'IMDB':
        cfg['train']['early_stop_cfg']['warmup'] = 10
        cfg['train']['early_stop_cfg']['patience'] = 10
        cfg['train']['optim_cfg']['lr'] = 1e-3
    else:
        assert False

    return cfg


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


class EarlyStopping:
    def __init__(self, warmup, patience, criterion: Tuple[str, int], margin, ckpt_file):
        self.warmup = warmup
        self.patience = patience
        self.criterion = criterion
        self.margin = margin
        self.ckpt_file = ckpt_file

        self.counter = 0
        self.best_record = None
        return

    def record(self, record: Dict[str, float], epoch, model):
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
            if epoch < self.warmup:
                record_msg = f'EarlyStopping counter blocked by warmup {epoch}/{self.warmup}.'
            else:
                self.counter += 1
                record_msg = f'EarlyStopping counter {self.counter} / {self.patience}.'
                if self.counter >= self.patience:
                    record_msg = 'EarlyStopping. Best record is:'
                    return self.best_record, record_msg
        return None, record_msg


def test(model, data: MetapathDataset, sampler: Sampler, all_indices: np.array):
    model.eval()
    log_prob_list = []
    for iteration in range(sampler.num_iterations()):
        indices, metapath_list = sampler.sample()
        with torch.no_grad():
            log_prob = model.forward(
                indices, metapath_list, data.id2type, data.feature_list
            )
        log_prob_list.append(log_prob)
    with torch.no_grad():
        log_prob = torch.cat(log_prob_list, dim=0)
        labels = data.labels[all_indices]
        loss = F.nll_loss(log_prob, labels)
        pred = torch.argmax(log_prob, dim=-1)
    loss = loss.item()
    labels = to_np(labels)
    pred = to_np(pred)
    macro_f1, micro_f1 = evaluate_multiclass(labels, pred)
    return loss, macro_f1, micro_f1

