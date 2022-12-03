import os.path as osp
import argparse
import logging

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

from typing import Optional, Dict


class Namespace:
    def __init__(self, **kwargs):
        self.update(**kwargs)
        return

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return


def get_logger(name, workdir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')

    file = osp.join(workdir, f'{name}.log')
    file_handle = logging.FileHandler(file, mode='w+', encoding='utf-8')
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

    stream_handle = logging.StreamHandler()
    stream_handle.setFormatter(formatter)
    logger.addHandler(stream_handle)
    return logger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default='debug')
    ap.add_argument('--seed', type=int, default=0)
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


def svm_test(embeddings, labels, seed, train_ratio_list=(0.8, 0.6, 0.4, 0.2), num_repeat=10):
    macro_f1_mean = []
    macro_f1_std = []
    micro_f1_mean = []
    micro_f1_std = []
    random_seeds = np.arange(num_repeat, dtype=np.int32) + seed
    for train_ratio in train_ratio_list:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(num_repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=1 - train_ratio, shuffle=True, random_state=random_seeds[i]
            )
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        macro_f1_mean.append((np.mean(macro_f1_list)))
        macro_f1_std.append((np.std(macro_f1_list)))
        micro_f1_mean.append((np.mean(micro_f1_list)))
        micro_f1_std.append((np.std(micro_f1_list)))
    macro_f1_msg = 'Macro-F1: ' + '|'.join([
        f'{macro_f1_mean[i]:,6f}~{macro_f1_std[i]:.6f} ({train_ratio_list[i]:.1f})'
        for i in range(len(train_ratio_list))
    ])
    micro_f1_msg = 'Micro-F1: ' + '|'.join([
        f'{micro_f1_mean[i]:,6f}~{micro_f1_std[i]:.6f} ({train_ratio_list[i]:.1f})'
        for i in range(len(train_ratio_list))
    ])
    result_dict = dict(
        macro_f1=dict(
            mean=macro_f1_mean,
            std=macro_f1_std,
            msg=macro_f1_msg,
        ),
        micro_f1=dict(
            mean=micro_f1_mean,
            std=micro_f1_std,
            msg=micro_f1_msg
        )
    )
    return result_dict

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
    def __init__(self, patience, criterion, delta, ckpt_file, logger: Optional[logging.Logger] = None):
        self.patience = patience
        self.criterion = criterion
        self.delta = delta
        self.logger = logger
        self.ckpt_file = ckpt_file

        self.counter = 0
        self.best_record = None
        return

    def record(self, record: Dict[str, float], model):
        if self.best_record is None:
            self.best_record = record
            self._save_checkpoint(record, model)
        elif record[self.criterion] < self.best_record[self.criterion] - self.delta:
            self.counter += 1
            if self.logger is not None:
                self.logger.info(f'EarlyStopping counter {self.counter} / {self.patience}.')
            if self.counter >= self.patience:
                record_msg = '|'.join([f'{key}: {value:.6f}' for key, value in self.best_record.items()])
                return self.best_record, record_msg
        else:
            self.best_record = record
            self._save_checkpoint(record, model)
            self.counter = 0
        return None

    def _save_checkpoint(self, record, model):
        if self.logger is not None:
            prev_best = self.best_record[self.criterion] if self.best_record is not None else -np.Inf
            new_best = record[self.criterion]

            self.logger.info(
                f'{self.criterion} increased {prev_best:.6f} --> {new_best:.6f}. Checkpoint saved.'
            )
        torch.save(model.state_dict(), self.ckpt_file)
        return


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
