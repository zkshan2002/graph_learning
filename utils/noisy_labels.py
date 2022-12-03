import numpy as np

def apply_label_noise(labels: np.ndarray, pair_flip_rate, uniform_flip_rate):
    num_labels = labels.shape[0]
    noisy_labels = np.copy(labels)

    # apply pair noise
    # 0: Database; 1: Data Mining; 2: AI; 3: Information Retrieval;
    pairs = np.array([1, 0, 3, 2], dtype=np.int32)
    mask = (np.random.rand(num_labels) < pair_flip_rate)
    noisy_labels[mask] = pairs[noisy_labels[mask]]

    # apply uniform noise
    mask = (np.random.rand(num_labels) < uniform_flip_rate)
    num_cls = pairs.shape[0]
    num_corrupted = np.where(mask)[0].shape[0]
    shift = np.random.randint(1, num_cls, size=num_corrupted)
    noisy_labels[mask] = (labels[mask] + shift) % num_cls

    is_corrupted = (noisy_labels != labels)

    return noisy_labels, is_corrupted
