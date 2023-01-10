import argparse

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


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
    ap.add_argument('--warmup', type=int)
    ap.add_argument('--patience', type=int)
    # LNL
    ap.add_argument('--noise_type', type=str)
    ap.add_argument('--flip_rate', type=float)
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
    if hasattr(args, 'warmup') and args.warmup is not None:
        train_cfg['early_stop_cfg']['warmup'] = args.warmup
    if hasattr(args, 'patience') and args.patience is not None:
        train_cfg['early_stop_cfg']['patience'] = args.patience
    # LNL
    if hasattr(args, 'noise_type') and args.noise_type is not None:
        data_cfg['noise_cfg']['noise_type'] = args.noise_type
        data_cfg['noise_cfg']['apply'] = True
    if hasattr(args, 'flip_rate') and args.flip_rate is not None:
        data_cfg['noise_cfg']['flip_rate'] = args.flip_rate
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
        train_cfg['early_stop_cfg']['warmup'] = 5
        train_cfg['early_stop_cfg']['patience'] = 5
        train_cfg['optim_cfg']['lr'] = 5e-3
    elif dataset == 'IMDB':
        train_cfg['early_stop_cfg']['warmup'] = 10
        train_cfg['early_stop_cfg']['patience'] = 10
        train_cfg['optim_cfg']['lr'] = 1e-3
    else:
        assert False

    exp_cfg['command'] = args.command
    return


def indices_mapping(local2global: torch.tensor):
    num_nodes = local2global.shape[0]
    num_size = torch.max(local2global) + 1
    device = local2global.device
    global2local = -1 * torch.ones(num_size, dtype=torch.int64, device=device)
    global2local[local2global] = torch.arange(num_nodes, dtype=torch.int64, device=device)
    return global2local


class Namespace:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        return

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return


def svm_test(embeddings, labels, train_ratio, repeat, seed):
    macro_f1_list = []
    micro_f1_list = []
    confusion_mat_list = []

    for i in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, train_size=train_ratio, shuffle=True, random_state=seed + i
        )
        svm = LinearSVC(dual=False)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        macro_f1_list.append(
            f1_score(y_test, y_pred, average='macro')
        )
        micro_f1_list.append(
            f1_score(y_test, y_pred, average='micro')
        )
        confusion_mat_list.append(
            confusion_matrix(y_test, y_pred).reshape(-1)
        )

    macro_f1_mean = np.mean(macro_f1_list)
    macro_f1_std = np.std(macro_f1_list)
    macro_f1_msg = f'Macro-F1: {macro_f1_mean:.6f} ~ {macro_f1_std:.6f}'
    micro_f1_mean = np.mean(micro_f1_list)
    micro_f1_std = np.std(micro_f1_list)
    micro_f1_msg = f'Micro-F1: {micro_f1_mean:.6f} ~ {micro_f1_std:.6f}'
    confusion_mats = np.stack(confusion_mat_list)
    confusion_mat_mean = np.mean(confusion_mats, axis=0)
    confusion_mat_std = np.std(confusion_mats, axis=0)
    num_cls = np.max(labels) + 1
    confusion_mat_msg = f'Confusion Matrix:\n'
    for i in range(num_cls * num_cls):
        confusion_mat_msg += f'{confusion_mat_mean[i]:.2f},'
        if (i + 1) % num_cls == 0:
            confusion_mat_msg += '\n'

    result_dict = dict(
        macro_f1=dict(
            mean=macro_f1_mean,
            std=macro_f1_std,
            msg=macro_f1_msg,
        ),
        micro_f1=dict(
            mean=micro_f1_mean,
            std=micro_f1_std,
            msg=micro_f1_msg,
        ),
        confusion_mat=dict(
            mean=confusion_mat_mean,
            std=confusion_mat_std,
            msg=confusion_mat_msg,
        )
    )
    return result_dict


def evaluate_results_raw(embeddings, labels, num_classes):
    def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
        random_states = [182318 + i for i in range(repeat)]
        result_macro_f1_list = []
        result_micro_f1_list = []
        for test_size in test_sizes:
            macro_f1_list = []
            micro_f1_list = []
            for i in range(repeat):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
                svm = LinearSVC(dual=False)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                micro_f1 = f1_score(y_test, y_pred, average='micro')
                macro_f1_list.append(macro_f1)
                micro_f1_list.append(micro_f1)
            result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
            result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
        return result_macro_f1_list, result_micro_f1_list

    def kmeans_test(X, y, n_clusters, repeat=10):
        nmi_list = []
        ari_list = []
        for _ in range(repeat):
            kmeans = KMeans(n_clusters=n_clusters)
            y_pred = kmeans.fit_predict(X)
            nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
            ari_score = adjusted_rand_score(y, y_pred)
            nmi_list.append(nmi_score)
            ari_list.append(ari_score)
        return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)

    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std
