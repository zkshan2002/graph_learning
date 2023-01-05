import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


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
