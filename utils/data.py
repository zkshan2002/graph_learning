import os
import os.path as osp
import pickle
from typing import List

import numpy as np
import torch
import scipy


def to_torch(data, device, indices=False):
    if isinstance(data, list):
        return [to_torch(entry, device, indices) for entry in data]
    if isinstance(data, dict):
        return {key: to_torch(value, device, indices) for key, value in data.items()}
    assert isinstance(data, np.ndarray)
    data = np.copy(data)
    data = torch.from_numpy(data).to(device)
    if indices:
        data = data.to(torch.int64)
    else:
        data = data.to(torch.float32)
    return data


def to_np(data: torch.Tensor) -> np.array:
    assert isinstance(data, torch.Tensor)
    data = torch.clone(data)
    data = data.detach().cpu().numpy()
    return data


class MetapathDataset:
    def __init__(self, id2type: np.array, adj_mat: np.array, feature_list: List[np.array], labels: np.array,
                 all_metapath_instances: List[List[np.array]], device):
        self.id2type = to_torch(id2type, device=device)
        self.adj_mat = adj_mat
        self.feature_list = to_torch(feature_list, device=device)
        self.labels = to_torch(labels, device=device, indices=True)
        self.all_metapath_instances = all_metapath_instances

        self.num_node_types = np.max(id2type) + 1
        self.num_metapath_schemes = len(all_metapath_instances)
        self.num_target_nodes = labels.shape[0]
        self.feature_dim_list = [node_feature.shape[1] for node_feature in feature_list]
        self.num_cls = np.max(labels) + 1
        return

    def sample(self, target_nodes: np.array, sample_limit, keep_intermediate=False):
        metapath_sampled_list = []
        for metapath_id in range(self.num_metapath_schemes):
            metapath_sampled = []
            for target_node in target_nodes:
                metapath_given_node = self.all_metapath_instances[metapath_id][target_node]
                _, neighbor_cnt = np.unique(metapath_given_node[:, 0], return_counts=True)
                # sample with prob ~ cnt ^ 0.75
                prob = []
                for cnt in neighbor_cnt:
                    prob += [cnt ** 0.75 / cnt] * cnt
                prob = np.array(prob)
                prob /= np.sum(prob)
                num_available = metapath_given_node.shape[0]
                num_sample = min(num_available, sample_limit)
                if num_sample == 0:
                    if keep_intermediate:
                        length = metapath_given_node.shape[1]
                    else:
                        length = 2
                    selected_metapath = np.zeros((0, length), dtype=np.int32)
                else:
                    selected_indices = np.sort(np.random.choice(num_available, num_sample, replace=False, p=prob))
                    selected_metapath = metapath_given_node[selected_indices]
                    if not keep_intermediate:
                        selected_metapath = selected_metapath[:, [0, -1]]
                metapath_sampled.append(selected_metapath)
            metapath_sampled = np.concatenate(metapath_sampled, axis=0)
            metapath_sampled_list.append(metapath_sampled)

        return metapath_sampled_list


def load_data(dataset: str, device):
    project_root = osp.realpath('..')
    if dataset == 'DBLP':
        return _load_DBLP(project_root, device)
    elif dataset == 'IMDB':
        return _load_IMDB(project_root, device)
    else:
        assert False


def _load_DBLP(project_root, device):
    data_root = osp.join(project_root, 'data/preprocessed/DBLP_processed')

    # (26128,)
    id2type = np.load(osp.join(data_root, 'node_types.npy'))
    # (26128, 26128) (sparse)
    adj_mat = scipy.sparse.load_npz(osp.join(data_root, 'adjM.npz'))

    feature_list = [
        # (4057, 334) (sparse) author
        scipy.sparse.load_npz(osp.join(data_root, 'features_0.npz')).toarray(),
        # (14328, 4231) (sparse) paper
        scipy.sparse.load_npz(osp.join(data_root, 'features_1.npz')).toarray(),
        # (7723, 50) term
        np.load(osp.join(data_root, 'features_2.npy')),
        # (20, 20) conf
        np.eye(20, dtype=np.float32)
    ]

    # (4057,) author labels, {0, 1, 2, 3}
    labels = np.load(osp.join(data_root, 'labels.npy'))

    # [3][4057] (cnt, len)
    all_metapath_instances = []
    for tag in ['0/0-1-0', '0/0-1-2-1-0', '0/0-1-3-1-0']:
        pkl_file = osp.join(data_root, f'{tag}_idx.pickle')
        with open(pkl_file, 'rb') as f:
            metapath_dict = pickle.load(f)
        # list of arrays (b, len_path)
        metapath_list = list(metapath_dict.values())
        all_metapath_instances.append(metapath_list)

    # train/val/test: 400, 400, 3257
    # all_indices = np.load(osp.join(data_root, 'train_val_test_idx.npz'))
    # train_indices = np.sort(all_indices['train_idx'])
    # val_indices = np.sort(all_indices['val_idx'])
    # test_indices = np.sort(all_indices['test_idx'])

    return MetapathDataset(id2type, adj_mat, feature_list, labels, all_metapath_instances, device)


def _load_IMDB(project_root, device):
    data_root = osp.join(project_root, 'data/preprocessed/IMDB_processed')

    # (11616,)
    id2type = np.load(osp.join(data_root, 'node_types.npy'))
    # (11616, 11616) (sparse)
    adj_mat = scipy.sparse.load_npz(osp.join(data_root, 'adjM.npz'))

    feature_list = [
        # (4278, 3066) (sparse) movie
        scipy.sparse.load_npz(osp.join(data_root, 'features_0.npz')).toarray(),
        # (2081, 3066) (sparse) director
        scipy.sparse.load_npz(osp.join(data_root, 'features_1.npz')).toarray(),
        # (5257, 3066) (sparse) actor
        scipy.sparse.load_npz(osp.join(data_root, 'features_2.npz')).toarray(),
    ]

    # (4278,) author labels, {0, 1, 2}
    labels = np.load(osp.join(data_root, 'labels.npy'))

    # load control type only) ['1/1-0-1', '1/1-0-2-0-1', '2/2-0-2', '2/2-0-1-0-2']
    # [2][4278] (cnt, len)
    all_metapath_instances = []
    for tag in ['0/0-1-0', '0/0-2-0']:
        npy_file = osp.join(data_root, f'{tag}_idx.npy')
        metapath_instances = np.load(npy_file)
        metapath_list = []
        for control_node in range(4278):
            mask = np.where(metapath_instances[:, -1] == control_node)
            metapath_list.append(metapath_instances[mask])
        all_metapath_instances.append(metapath_list)

    # # train/val/test: 400, 400, 3478
    # all_indices = np.load(osp.join(data_root, 'train_val_test_idx.npz'))
    # train_indices = np.sort(all_indices['train_idx'])
    # val_indices = np.sort(all_indices['val_idx'])
    # test_indices = np.sort(all_indices['test_idx'])

    return MetapathDataset(id2type, adj_mat, feature_list, labels, all_metapath_instances, device)


def split_data(data: MetapathDataset, data_cfg: dict):
    all_indices = np.arange(data.num_target_nodes)
    split_ratio = data_cfg['split_cfg']['split_ratio']
    num_total = all_indices.shape[0]
    num_train = int(num_total * split_ratio[0])
    num_val = int(num_total * split_ratio[1])
    np.random.seed(data_cfg['split_cfg']['seed'])
    np.random.shuffle(all_indices)
    all_train_indices = all_indices[:num_train]
    all_val_indices = all_indices[num_train:num_train + num_val]
    all_test_indices = all_indices[num_train + num_val:]
    data_cfg['split_cfg']['node_type_cnt'] = {}
    for indices, name in zip([all_train_indices, all_val_indices, all_test_indices], ['train', 'val', 'test']):
        label = data.labels[indices]
        type_cnt = []
        for i in range(data.num_node_types):
            type_cnt.append(torch.where(label == i)[0].shape[0])
        data_cfg['split_cfg']['node_type_cnt'][name] = type_cnt

    return all_train_indices, all_val_indices, all_test_indices


class Sampler:
    def __init__(self, data: MetapathDataset, all_indices: np.array, batch_size, sample_limit, device, shuffle=False,
                 loop=False):
        self.data = data
        self.all_indices = all_indices
        self.num_batch = batch_size
        self.sample_limit = sample_limit
        self.device = device
        self.shuffle = shuffle
        self.loop = loop

        self.num_indices = all_indices.shape[0]
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
        indices = np.copy(self.all_indices[self.pointer:self.pointer + self.num_batch])
        self.pointer += self.num_batch
        if self.pointer >= self.num_indices:
            self._reset()
            if self.loop:
                self.pointer = self.num_batch - indices.shape[0]
                compensate = np.copy(self.all_indices[:self.pointer])
                indices = np.concatenate([indices, compensate], axis=0)
        metapath_sampled_list = self.data.sample(indices, self.sample_limit, keep_intermediate=False)
        indices = to_torch(indices, self.device, indices=True)
        metapath_sampled_list = to_torch(metapath_sampled_list, self.device, indices=True)
        return indices, metapath_sampled_list
