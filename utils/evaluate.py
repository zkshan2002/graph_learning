import numpy as np

from sklearn.metrics import f1_score


def evaluate_multiclass(label: np.array, pred: np.array):
    macro_f1 = f1_score(label, pred, average='macro')
    micro_f1 = f1_score(label, pred, average='micro')
    return macro_f1, micro_f1

