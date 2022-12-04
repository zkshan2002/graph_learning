import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def svm_test(embeddings, labels, seed, train_ratio_list, num_repeat=10):
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
            macro_f1_list.append(macro_f1)
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            micro_f1_list.append(micro_f1)
        macro_f1_mean.append((np.mean(macro_f1_list)))
        macro_f1_std.append((np.std(macro_f1_list)))
        micro_f1_mean.append((np.mean(micro_f1_list)))
        micro_f1_std.append((np.std(micro_f1_list)))
    macro_f1_msg = 'Macro-F1: ' + ' | '.join([
        f'{macro_f1_mean[i]:.6f} ~ {macro_f1_std[i]:.6f} ({train_ratio_list[i]:.1f})'
        for i in range(len(train_ratio_list))
    ])
    micro_f1_msg = 'Micro-F1: ' + ' | '.join([
        f'{micro_f1_mean[i]:.6f} ~ {micro_f1_std[i]:.6f} ({train_ratio_list[i]:.1f})'
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
            msg=micro_f1_msg,
        ),
    )
    return result_dict
