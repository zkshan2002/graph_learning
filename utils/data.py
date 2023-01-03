import os
import os.path as osp
import pickle
from typing import Optional, List

import numpy as np
import torch
import scipy
import networkx as nx


def to_torch(data, device, indices=False):
    if isinstance(data, list):
        return [to_torch(entry, device, indices) for entry in data]
    if isinstance(data, dict):
        return {key: to_torch(value, device, indices) for key, value in data.items()}
    assert isinstance(data, np.ndarray)
    data = torch.from_numpy(data).to(device)
    if indices:
        data = data.to(torch.int64)
    else:
        data = data.to(torch.float32)
    return data


def to_np(data):
    assert isinstance(data, torch.Tensor)
    data = data.detach().cpu().numpy()
    return data


def load_data(dataset: str, project_root):
    if dataset == 'DBLP':
        return _load_DBLP(project_root)
    elif dataset == 'IMDB':
        return _load_IMDB(project_root)
    else:
        assert False


def _load_DBLP(project_root):
    data_root = osp.join(project_root, 'data/preprocessed/DBLP_processed')

    # load metapaths
    metapath_list = []
    for tag in ['0/0-1-0', '0/0-1-2-1-0', '0/0-1-3-1-0']:
        pkl_file = osp.join(data_root, f'{tag}_idx.pickle')
        with open(pkl_file, 'rb') as f:
            metapaths_dict = pickle.load(f)
        # list of arrays (b, len_path)
        metapaths = list(metapaths_dict.values())
        metapath_list.append(metapaths)

    # load node features
    # (4057, 334) (sparse) author
    node_features_0 = scipy.sparse.load_npz(osp.join(data_root, 'features_0.npz')).toarray()
    # (14328, 4231) (sparse) paper
    node_features_1 = scipy.sparse.load_npz(osp.join(data_root, 'features_1.npz')).toarray()
    # (7723, 50) term
    node_features_2 = np.load(osp.join(data_root, 'features_2.npy'))
    # (20, 20) conf
    node_features_3 = np.eye(20, dtype=np.float32)
    node_feature_list = [
        node_features_0, node_features_1, node_features_2, node_features_3
    ]

    # load structures
    # (26128,) node type mapping
    node_type_mapping = np.load(osp.join(data_root, 'node_types.npy'))
    # (26128, 26128) (sparse) full adjacency matrix
    adjacency_matrix = scipy.sparse.load_npz(osp.join(data_root, 'adjM.npz'))

    # (4057,) author labels, {0, 1, 2, 3}
    labels = np.load(osp.join(data_root, 'labels.npy'))
    # train/val/test: 400, 400, 3257
    all_indices = np.load(osp.join(data_root, 'train_val_test_idx.npz'))
    train_indices = np.sort(all_indices['train_idx'])
    val_indices = np.sort(all_indices['val_idx'])
    test_indices = np.sort(all_indices['test_idx'])

    return (
        metapath_list,
        node_feature_list,
        node_type_mapping,
        adjacency_matrix,
        labels,
        (train_indices, val_indices, test_indices)
    )


def _load_IMDB(project_root):
    data_root = osp.join(project_root, 'data/preprocessed/IMDB_processed')

    num_control_nodes = 4278
    # load metapaths(control type only)
    # ['1/1-0-1', '1/1-0-2-0-1', '2/2-0-2', '2/2-0-1-0-2']
    metapath_list = []
    for tag in ['0/0-1-0', '0/0-2-0']:
        npy_file = osp.join(data_root, f'{tag}_idx.npy')
        all_metapaths = np.load(npy_file)
        metapaths = []
        for control_node in range(num_control_nodes):
            mask = np.where(all_metapaths[:, -1] == control_node)
            metapaths.append(all_metapaths[mask])
        # list of arrays (b, len_path)
        metapath_list.append(metapaths)

    # load node features
    # (4278, 3066) (sparse) movie
    node_features_0 = scipy.sparse.load_npz(osp.join(data_root, 'features_0.npz')).toarray()
    # (2081, 3066) (sparse) director
    node_features_1 = scipy.sparse.load_npz(osp.join(data_root, 'features_1.npz')).toarray()
    # (5257, 3066) (sparse) actor
    node_features_2 = scipy.sparse.load_npz(osp.join(data_root, 'features_2.npz')).toarray()
    node_feature_list = [
        node_features_0, node_features_1, node_features_2
    ]

    # load structures
    # (11616,) node type mapping
    node_type_mapping = np.load(osp.join(data_root, 'node_types.npy'))
    # (11616, 11616) (sparse) full adjacency matrix
    adjacency_matrix = scipy.sparse.load_npz(osp.join(data_root, 'adjM.npz'))

    # (4278,) author labels, {0, 1, 2}
    labels = np.load(osp.join(data_root, 'labels.npy'))
    # train/val/test: 400, 400, 3257
    all_indices = np.load(osp.join(data_root, 'train_val_test_idx.npz'))
    train_indices = np.sort(all_indices['train_idx'])
    val_indices = np.sort(all_indices['val_idx'])
    test_indices = np.sort(all_indices['test_idx'])

    return (
        metapath_list,
        node_feature_list,
        node_type_mapping,
        adjacency_matrix,
        labels,
        (train_indices, val_indices, test_indices)
    )


def sample_metapath(
        target_nodes: np.ndarray, metapath_list: List[List[np.ndarray]], sample_limit: int, keep_intermediate=False
):
    """
    Sample up to <sample_limit> metapaths for each node in <target_nodes> from <metapath_list>.
    If not <keep_intermediate>, return metapath-based neighbors only
    """
    metapath_sampled_list = []
    num_metapaths = len(metapath_list)
    for metapath_id in range(num_metapaths):
        metapath_sampled = []
        for target_node in target_nodes:
            metapaths_this_node = metapath_list[metapath_id][target_node]
            neighbor_indices = metapaths_this_node[:, 0]
            _, neighbor_cnt = np.unique(neighbor_indices, return_counts=True)
            # sample with prob ~ cnt ^ 0.75
            prob = []
            for cnt in neighbor_cnt:
                prob += [cnt ** 0.75 / cnt] * cnt
            prob = np.array(prob)
            prob /= np.sum(prob)
            num_available = neighbor_indices.shape[0]
            num_sample = min(num_available, sample_limit)
            if num_sample == 0:
                if keep_intermediate:
                    len_metapath = metapaths_this_node.shape[1]
                    selected_metapaths = np.zeros((0, len_metapath), dtype=np.int32)
                else:
                    selected_metapaths = np.zeros((0, 2), dtype=np.int32)
            else:
                selected_indices = np.sort(np.random.choice(num_available, num_sample, replace=False, p=prob))
                selected_metapaths = metapaths_this_node[selected_indices]
                if not keep_intermediate:
                    selected_metapaths = selected_metapaths[:, [0, -1]]
            metapath_sampled.append(selected_metapaths)
        metapath_sampled = np.concatenate(metapath_sampled, axis=0)
        metapath_sampled_list.append(metapath_sampled)

    return metapath_sampled_list


def indices_mapping(local2global: torch.tensor):
    num_nodes = local2global.shape[0]
    num_size = torch.max(local2global) + 1
    device = local2global.device
    global2local = -1 * torch.ones(num_size, dtype=torch.int64, device=device)
    global2local[local2global] = torch.arange(num_nodes, dtype=torch.int64, device=device)
    return global2local


def __get_DBLP_statistics(project_dir):
    _, _, node_type_mapping, AM, _, _ = _load_DBLP(project_dir)
    A_cnt = np.where(node_type_mapping == 0)[0].shape[0]
    P_cnt = np.where(node_type_mapping == 1)[0].shape[0]
    T_cnt = np.where(node_type_mapping == 2)[0].shape[0]
    C_cnt = np.where(node_type_mapping == 3)[0].shape[0]
    A_slice = slice(0, A_cnt)
    P_slice = slice(A_cnt, A_cnt + P_cnt)
    T_slice = slice(A_cnt + P_cnt, A_cnt + P_cnt + T_cnt)
    C_slice = slice(A_cnt + P_cnt + T_cnt, A_cnt + P_cnt + T_cnt + C_cnt)
    AA = AM[A_slice, A_slice].todense()
    assert np.sum(AA) == 0
    AP = AM[A_slice, P_slice].todense()
    PA = AM[P_slice, A_slice].todense()
    assert (AP == PA.T).all()

    import pdb
    pdb.set_trace()

    return


class MetapathDataset:
    def __init__(self, id2type: np.array, adj_mat, feature_list, labels, metapath_list, device):
        self.id2type = to_torch(id2type, device=device)
        self.adj_mat = adj_mat
        self.feature_list = to_torch(feature_list, device=device)
        self.labels = labels
        self.metapath_list = metapath_list

        self.num_node_types = np.max(id2type) + 1
        self.num_metapath_schemes = len(metapath_list)
        self.num_target_nodes = labels.shape[0]
        self.feature_dim_list = [node_feature.shape[1] for node_feature in feature_list]
        self.num_cls = np.max(labels) + 1
        return

    def sample(self, target_nodes: np.array, sample_limit: int, keep_intermediate=False):
        metapath_sampled_list = []
        for metapath_id in range(self.num_metapath_schemes):
            metapath_sampled = []
            for target_node in target_nodes:
                metapath_given_node = self.metapath_list[metapath_id][target_node]
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
                    selected_metapaths = np.zeros((0, length), dtype=np.int32)
                else:
                    selected_indices = np.sort(np.random.choice(num_available, num_sample, replace=False, p=prob))
                    selected_metapaths = metapath_given_node[selected_indices]
                    if not keep_intermediate:
                        selected_metapaths = selected_metapaths[:, [0, -1]]
                metapath_sampled.append(selected_metapaths)
            metapath_sampled = np.concatenate(metapath_sampled, axis=0)
            metapath_sampled_list.append(metapath_sampled)

        return metapath_sampled_list

class Sampler:
    def __init__(self, dataset: MetapathDataset, batch_size, sample_limit, device, shuffle, loop=False):
        self.dataset = dataset
        self.num_indices = dataset.num_target_nodes
        self.all_indices = np.arange(self.num_indices, dtype=np.int64)
        self.num_batch = batch_size
        self.sample_limit = sample_limit
        self.device = device
        self.shuffle = shuffle
        self.loop = loop
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
        def sample_indice():
            indices = np.copy(self.all_indices[self.pointer:self.pointer + self.num_batch])
            self.pointer += self.num_batch
            if self.pointer >= self.num_indices:
                self._reset()
                if self.loop:
                    self.pointer = self.num_batch - indices.shape[0]
                    compensate = np.copy(self.all_indices[:self.pointer])
                    indices = np.concatenate([indices, compensate], axis=0)

            return indices

        indices = sample_indice()
        metapath_sampled_list = self.dataset.sample(indices, self.sample_limit, keep_intermediate=False)
        indices = to_torch(indices, self.device, indices=True)
        metapath_sampled_list = to_torch(metapath_sampled_list, self.device, indices=True)
        return indices, metapath_sampled_list
