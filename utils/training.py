import os.path as osp
import argparse
import logging

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

from typing import Tuple, Dict


class Namespace:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        return

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='debug')
    ap.add_argument('--description', type=str, default='')
    ap.add_argument('--seed_list', nargs='+', type=int)
    ap.add_argument('--batch_size', type=int)
    ap.add_argument('--sample_limit', type=int)
    ap.add_argument('--noise_p', type=float)
    ap.add_argument('--noise_u', type=float)
    ap.add_argument('--sft_mb_warmup', type=int)
    ap.add_argument('--sft_loss_threshold', type=float)
    ap.add_argument('--sft_loss_weights', nargs='+', type=float)
    args = ap.parse_args()
    return args


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


def get_logger(name, file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # %(asctime)s %(levelname)s
    formatter = logging.Formatter('%(name)s %(message)s')

    file_handle = logging.FileHandler(file, mode='w+', encoding='utf-8')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    stream_handle = logging.StreamHandler()
    stream_handle.setFormatter(formatter)
    logger.addHandler(stream_handle)
    return logger


def evaluate_results_nc(embeddings, labels, num_classes):
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


class IndicesSampler:
    def __init__(self, data, batch_size, shuffle, loop=False):
        self.data = data
        self.num_data = data.shape[0]
        self.num_batch = batch_size
        self.shuffle = shuffle
        self.loop = loop
        self.pointer = 0

        self._reset()
        return

    def _reset(self):
        self.pointer = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        return

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.num_batch))

    def __call__(self):
        selected_data = np.copy(self.data[self.pointer:self.pointer + self.num_batch])
        self.pointer += self.num_batch
        if self.pointer >= self.num_data:
            self._reset()
            if self.loop:
                self.pointer = self.num_batch - selected_data.shape[0]
                compensate = np.copy(self.data[:self.pointer])
                selected_data = np.concatenate([selected_data, compensate], axis=0)

        return selected_data
