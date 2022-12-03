import os
import os.path as osp
import json
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils.training import parse_args, set_seed, get_logger
from utils.training import EarlyStopping, IndicesSampler, Namespace
from utils.evaluate import svm_test
from utils.data import to_torch, to_np, load_DBLP, sample_metapath
from utils.noisy_labels import apply_label_noise

from config import train_cfg, model_cfg, data_cfg

if __name__ == '__main__':

    set_seed(0)

    args = Namespace(**train_cfg)
    if args.device_id == -1:
        device = 'cpu'
    else:
        device = f'cuda:{args.device_id}'
    args.update(**dict(device=device))

    # manage workdir
    project_root = osp.realpath(osp.join(osp.realpath(__file__), '..'))
    workdir = osp.join(project_root, 'exp', args.tag)
    os.makedirs(workdir, exist_ok=(args.tag == 'debug'))
    cfg_file = osp.join(workdir, 'cfg.json')

    log_file = osp.join(workdir, 'avg_result.log')
    logger_result = get_logger('exp_result', log_file)

    # load dataset
    dataset = data_cfg['dataset']
    assert dataset == 'DBLP'
    metapath_list, node_feature_list, node_type_mapping, adjacency_matrix, labels, (
        train_indices, val_indices, test_indices) = load_DBLP(project_root)

    num_metapaths = len(metapath_list)
    num_node_types = np.unique(node_type_mapping).shape[0]
    node_feature_dim_list = [node_feature.shape[1] for node_feature in node_feature_list]
    node_feature_list = to_torch(node_feature_list, device)

    # apply label noise
    LNL = data_cfg['noise_cfg'].pop('apply')
    if LNL:
        noisy_labels, label_corruption_mask = apply_label_noise(labels=labels, **data_cfg['noise_cfg'])
        noisy_labels = to_torch(noisy_labels, device, indices=True)

        corruption_ratio = np.mean(label_corruption_mask[train_indices].astype(np.int32))
        data_cfg['noise_cfg']['corruption_ratio'] = corruption_ratio
    else:
        data_cfg.pop('noise_cfg')

    labels = to_torch(labels, device, indices=True)

    # save additional information to cfg
    data_cfg['num_metapaths'] = num_metapaths
    data_cfg['num_node_types'] = num_node_types
    data_cfg['indices'] = dict(
        train=train_indices.shape[0],
        val=val_indices.shape[0],
        test=test_indices.shape[0],
    )

    all_cfg = dict(
        train=train_cfg,
        model=model_cfg,
        data=data_cfg,
    )
    with open(cfg_file, 'w') as f:
        json.dump(all_cfg, f)

    model_type = model_cfg.pop('type')
    assert model_type == 'HAN'

    exp_results = {}
    for key in [
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_acc',
        'macro_f1', 'micro_f1'
    ]:
        exp_results[key] = []

    display = dict(
        train_loss='Train_Loss',
        train_acc='Train_Accuracy',
        val_loss='Val_Loss',
        val_acc='Val_Accuracy',
        test_acc='Test_Accuracy',
        macro_f1='Macro-F1',
        micro_f1='Micro-F1',
    )

    num_repeat = len(args.seed_list)
    for exp_id, seed in enumerate(args.seed_list):

        set_seed(seed)
        log_file = osp.join(workdir, f'{seed}.log')
        logger = get_logger(f'{exp_id}_{num_repeat}_seed{seed}', log_file)
        ckpt_file = osp.join(workdir, f'ckpt_seed{seed}.pt')

        # build model
        from model.HAN import HAN

        model = HAN(
            node_raw_feature_dim_list=node_feature_dim_list,
            num_metapaths=num_metapaths,
            num_cls=num_node_types,
            device=device,
            **model_cfg
        )
        optimizer = torch.optim.Adam(model.parameters(), **args.optim_cfg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **args.scheduler_cfg)

        # training utils
        early_stopping = EarlyStopping(
            patience=args.patience, criterion=('val_loss', -1), margin=0, ckpt_file=ckpt_file
        )
        train_sampler = IndicesSampler(train_indices, args.batch_size, shuffle=True)
        val_sampler = IndicesSampler(val_indices, args.batch_size, shuffle=False)

        for epoch in range(args.epoch):
            t_start = time.time()
            model.train()
            train_loss_list = []
            accurate_cnt = 0
            for iteration in range(train_sampler.num_iterations()):
                indices = train_sampler()
                metapath_sampled_list = sample_metapath(indices, metapath_list, args.sample_limit,
                                                        keep_intermediate=False)
                indices = to_torch(indices, device, indices=True)
                metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                logits, embeddings = model(
                    indices, metapath_sampled_list, node_type_mapping, node_feature_list
                )
                logp = F.log_softmax(logits, 1)

                if LNL:
                    train_labels = noisy_labels[indices]
                else:
                    train_labels = labels[indices]

                train_loss = F.nll_loss(logp, train_labels)
                train_loss_list.append(train_loss.item())
                accurate_cnt += torch.sum(torch.argmax(logp, dim=-1) == train_labels).item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss = np.mean(train_loss_list)
            train_acc = accurate_cnt / train_indices.shape[0]

            model.eval()
            val_logp = []
            with torch.no_grad():
                for iteration in range(val_sampler.num_iterations()):
                    indices = val_sampler()
                    metapath_sampled_list = sample_metapath(indices, metapath_list, args.sample_limit,
                                                            keep_intermediate=False)
                    indices = to_torch(indices, device, indices=True)
                    metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                    logits, embeddings = model(
                        indices, metapath_sampled_list, node_type_mapping, node_feature_list
                    )
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                val_logp = torch.cat(val_logp, dim=0)
                val_loss = F.nll_loss(val_logp, labels[val_indices]).item()
                val_acc = torch.mean(torch.argmax(val_logp, dim=-1) == labels[val_indices], dtype=torch.float32).item()
            t_end = time.time()
            time_elapsed = t_end - t_start
            msg = f'Epoch {epoch:03d} | Time {time_elapsed:.4f}' + \
                  f' | Train_Loss {train_loss:.4f} | Train_Accuracy {train_acc:.4f}' + \
                  f' | Val_Loss {val_loss:.4f} | Val_Accuracy {val_acc:.4f}'
            logger.info(msg)
            record = dict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            best_record, msg = early_stopping.record(record=record, model=model)
            logger.info(msg)
            if best_record is not None:
                msg = []
                for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    msg.append(f'{display[term]} {best_record[term]:.6f}')
                msg = ' | '.join(msg)
                # record_msg = f"Train_Loss {best_record['train_loss']:.6f}" +\
                #              f" | Train_Accuracy {best_record['train_acc']:.6f}" + \
                #              f" | Val_Loss {best_record['val_loss']:.6f}" + \
                #              f" | Val_Accuracy {best_record['val_acc']:.6f}"
                logger.info(msg)
                for key, value in best_record.items():
                    exp_results[key].append(value)
                break

        test_sampler = IndicesSampler(test_indices, args.batch_size, shuffle=False)

        model.load_state_dict(torch.load(ckpt_file))
        model.eval()
        test_logits = []
        test_embeddings = []
        with torch.no_grad():
            for iteration in range(test_sampler.num_iterations()):
                indices = test_sampler()
                metapath_sampled_list = sample_metapath(indices, metapath_list, args.sample_limit,
                                                        keep_intermediate=False)
                indices = to_torch(indices, device, indices=True)
                metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                logits, embeddings = model(
                    indices, metapath_sampled_list, node_type_mapping, node_feature_list
                )

                test_logits.append(logits)
                test_embeddings.append(embeddings)
            test_logits = torch.cat(test_logits, dim=0)
            test_embeddings = torch.cat(test_embeddings, dim=0)
            test_acc = torch.mean(torch.argmax(test_logits, dim=-1) == labels[test_indices], dtype=torch.float32).item()

        exp_results['test_acc'].append(test_acc)
        msg = f'Test_Accuracy {test_acc:.6f}'
        logger.info(msg)

        test_embeddings = to_np(test_embeddings)
        test_labels = to_np(labels[test_indices])
        train_ratio_list = [0.8, 0.6, 0.4, 0.2]
        svm_results = svm_test(test_embeddings, test_labels, seed, train_ratio_list=train_ratio_list)

        for key, value in svm_results.items():
            exp_results[key].append(value['mean'])

        record_msg = '\n'.join([value['msg'] for value in svm_results.values()])
        logger.info(record_msg)

    for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_acc']:
        result = exp_results[term]
        msg = ' | '.join([f'{num:.6f}' for num in result])
        msg = f'{display[term]}: {msg}'
        logger_result.info(msg)
        mean = np.mean(result)
        std = np.std(result)
        msg = f'{display[term]} Summary: {mean:.6f} ~ {std:.6f}'
        logger_result.info(msg)

    train_ratio_list = [0.8, 0.6, 0.4, 0.2]
    for term in ['macro_f1', 'micro_f1']:
        results = np.array(exp_results[term])
        msg = []
        for index, train_ratio in enumerate(train_ratio_list):
            result = results[:, index]
            msg = ' | '.join([f'{num:.6f}' for num in result])
            msg = f'{display[term]}({train_ratio:.1f}): {msg}'
            logger_result.info(msg)
            mean = np.mean(result)
            std = np.std(result)
            msg = f'{display[term]}({train_ratio:.1f}) Summary: {mean:.6f} ~ {std:.6f}'
            logger_result.info(msg)
