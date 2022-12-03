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
    data = torch.from_numpy(data)
    if indices:
        data = data.to(torch.int64)
    else:
        data = data.to(torch.float32)
    return data.to(device)


def load_DBLP(project_root):
    data_root = osp.join(project_root, 'data/preprocessed/DBLP_processed')

    # load metapath indices
    metapath_list = []
    for tag in ['0-1-0', '0-1-2-1-0', '0-1-3-1-0']:
        pkl_file = osp.join(data_root, '0', f'{tag}_idx.pickle')
        with open(pkl_file, 'rb') as f:
            metapaths_dict = pickle.load(f)
        # list of arrays (b, len_path)
        metapaths = list(metapaths_dict.values())
        metapath_list.append(metapaths)

    # load node features
    # (4057, 334)（sparse) author
    node_features_0 = scipy.sparse.load_npz(osp.join(data_root, 'features_0.npz')).toarray()
    # (14328, 4231)（sparse) paper
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

def sample_metapath_local(
        target_nodes: np.ndarray, metapath_list: List[List[np.ndarray]], sample_limit: int, keep_intermediate=False
):
    """
    Sample up to <sample_limit> metapaths for each node in <target_nodes> from <metapath_list>.
    All seen nodes are transformed to local indices, with <target_nodes> starting from 0.
    If not <keep_intermediate>, return metapath-based neighbors only
    """
    max_node = -1
    for metapath in metapath_list:
        for target_node in target_nodes:
            max_node = max(max_node, np.max(metapath[target_node]))
    max_node += 1

    global2local = -1 * np.ones(max_node, dtype=np.int32)
    local_id = 0
    local2global = []
    for target_node in target_nodes:
        global2local[target_node] = local_id
        local_id += 1
        local2global.append(target_node)
    metapath_local_list = []
    num_metapaths = len(metapath_list)
    for metapath_id in range(num_metapaths):
        metapath_local = []
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
            selected_indices = np.sort(np.random.choice(num_available, num_sample, replace=False, p=prob))
            selected_metapaths = metapaths_this_node[selected_indices]

            if keep_intermediate:
                seen_node_set = np.unique(selected_metapaths[:, :-1])
            else:
                seen_node_set = np.unique(selected_metapaths[:, 0])
            for seen_node in seen_node_set:
                if global2local[seen_node] == -1:
                    global2local[seen_node] = local_id
                    local_id += 1
                    local2global.append(seen_node)
            if keep_intermediate:
                metapath_local_this_node = global2local[selected_metapaths]
            else:
                metapath_local_this_node = global2local[selected_metapaths[:, [0, -1]]]
            metapath_local.append(metapath_local_this_node)
        metapath_local = np.concatenate(metapath_local, axis=0)
        metapath_local_list.append(metapath_local)
    local2global = np.array(local2global, dtype=np.int32)

    return (
        metapath_local_list,
        global2local,
        local2global,
    )


def sample_minibatch_local(
        indices: np.ndarray, metapath_list: List[List[np.ndarray]], node_type_mapping: np.array,
        node_feature_list: List[np.array],
        sample_limit: int, keep_intermediate=False, device='cpu'
):
    metapath_local_list, global2local, local2global = \
        sample_metapath_local(indices, metapath_list, sample_limit, keep_intermediate=keep_intermediate)

    target_nodes = np.arange(indices.shape[0], dtype=np.int32)
    node_type_mapping_local = node_type_mapping[local2global]
    # for now
    assert np.where(node_type_mapping_local != 0)[0].shape[0] == 0
    node_feature_local_list = []
    for node_feature in node_feature_list:
        node_feature_local_list.append(np.empty((0, node_feature.shape[1])))
    node_feature_local_list[0] = node_feature_list[0][local2global]

    # target_nodes = to_torch(target_nodes, device, indices=True)
    metapath_local_list = [to_torch(metapath, device, indices=True) for metapath in metapath_local_list]
    # node_type_mapping_local = to_torch(node_type_mapping_local, device, indices=True)
    node_feature_local_list = [to_torch(node_feature_local, device) for node_feature_local in node_feature_local_list]

    return target_nodes, metapath_local_list, node_type_mapping_local, node_feature_local_list
