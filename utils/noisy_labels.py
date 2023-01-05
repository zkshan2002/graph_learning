import numpy as np
import torch

from utils.data import to_torch
from utils.training import set_seed


def apply_label_noise(dataset: str, labels: torch.tensor, noise_type, flip_rate, seed, fix_num, all_train_indices):
    set_seed(seed)
    flip_indices = _get_flip_indices(all_train_indices.shape[0], flip_rate, fix_num)
    flip_indices = all_train_indices[flip_indices]
    if noise_type == 'uniform':
        noisy_labels = _apply_uniform_noise(labels, flip_indices)
    elif noise_type == 'pair':
        noisy_labels = _apply_pair_noise(dataset, labels, flip_indices)
    else:
        assert False
    actual_flip_rate = torch.mean((noisy_labels != labels)[all_train_indices].to(torch.float32)).item()
    return noisy_labels, actual_flip_rate


def _get_flip_indices(num_total, flip_rate, fix_num) -> np.array:
    if fix_num:
        num_flip = int(num_total * flip_rate)
        flip_indices = np.random.choice(num_total, num_flip, replace=False)
    else:
        mask = (np.random.rand(num_total) < flip_rate)
        flip_indices = np.where(mask)[0]
    return flip_indices


def _apply_uniform_noise(labels: torch.tensor, flip_indices: np.array):
    noisy_labels = torch.clone(labels)
    num_cls = torch.max(labels) + 1
    shift = torch.randint(1, num_cls, size=flip_indices.shape, dtype=torch.int64, device=labels.device)
    noisy_labels[flip_indices] = (labels[flip_indices] + shift) % num_cls
    return noisy_labels


_pairs_dict = {
    'DBLP': [1, 0, 3, 2],  # 0: Database; 1: Data Mining; 2: AI; 3: Information Retrieval
    'IMDB': [1, 2, 0],  # 0: Action; 1: Comedy; 2: Drama
}


def _apply_pair_noise(dataset: str, labels: torch.tensor, flip_indices: np.array):
    noisy_labels = torch.clone(labels)
    pairs = to_torch(_pairs_dict[dataset], device=labels.device, indices=True)
    noisy_labels[flip_indices] = pairs[noisy_labels[flip_indices]]
    return noisy_labels


class MemoryBank:
    def __init__(self, num_nodes, memory, warmup, device, **kwargs):
        self.num_nodes = num_nodes
        self.memory = memory
        self.warmup = warmup
        self.device = device

        self.memory_bank = []
        self.last_epoch = 0
        self.buffer = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        return

    def filter(self, epoch, indices, accurate_mask):
        mask = torch.ones(indices.shape[0], dtype=torch.bool, device=self.device)
        if epoch == self.last_epoch + 1:
            if len(self.memory_bank) == self.memory:
                self.memory_bank = self.memory_bank[1:]
            self.memory_bank.append(torch.clone(self.buffer))
            self.buffer[:] = 0
        self.buffer[indices] = accurate_mask

        if epoch >= self.warmup:
            for index in range(len(self.memory_bank) + 1):
                correct = torch.zeros_like(mask)
                for memory in self.memory_bank[:index]:
                    correct = torch.logical_or(correct, memory[indices])
                miss = torch.zeros_like(mask)
                for memory in self.memory_bank[index:]:
                    miss = torch.logical_or(miss, torch.logical_not(memory[indices]))
                miss = torch.logical_or(miss, torch.logical_not(accurate_mask))
                mask = torch.logical_and(mask, torch.logical_not(torch.logical_and(correct, miss)))

        selected_indices = torch.where(mask)[0]
        fluctuating_indices = torch.where(torch.logical_not(mask))[0]
        self.last_epoch = epoch
        return selected_indices, fluctuating_indices


def debug_sft():
    sft = MemoryBank(16, 2, 1, 'cpu')

    indice_list = list(torch.arange(16, dtype=torch.int64).view(4, 4))

    memory0 = torch.zeros(16, dtype=torch.bool)
    memory0[[8, 9, 10, 11, 12, 13, 14, 15]] = 1
    memory1 = torch.zeros(16, dtype=torch.bool)
    memory1[[4, 5, 6, 7, 12, 13, 14, 15]] = 1
    memory2 = torch.zeros(16, dtype=torch.bool)
    memory2[[2, 3, 6, 7, 10, 11, 14, 15]] = 1
    memory3 = torch.zeros(16, dtype=torch.bool)
    memory3[[1, 3, 5, 7, 9, 11, 13, 15]] = 1

    for indices in indice_list:
        _, _ = sft.filter(0, indices, memory0[indices])

    for indices in indice_list:
        _, _ = sft.filter(1, indices, memory1[indices])

    for indices in indice_list:
        _, _ = sft.filter(2, indices, memory2[indices])

    for indices in indice_list:
        selected, fluctuated = sft.filter(3, indices, memory3[indices])
        mb = torch.cat([memory.view(1, -1) for memory in sft.memory_bank], dim=0).to(torch.int32)
        print(mb)
        print(memory3.to(torch.int32))
        import pdb
        pdb.set_trace()

