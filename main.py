import os
import os.path as osp
import json
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from utils.training import parse_args, set_seed, evaluate_results_nc
from utils.training import EarlyStopping, IndicesSampler, Namespace, get_logger
from utils.training import svm_test
from utils.data import to_torch, load_DBLP, sample_metapath
from utils.noisy_labels import apply_label_noise

from exp.config import train_cfg, model_cfg, data_cfg

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

    logger_result = get_logger('result', workdir)

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
    # noisy_labels, label_corruption_mask = apply_label_noise(labels=labels, **data_cfg['noise_cfg'])
    # corruption_ratio = np.mean(labels[train_indices] != noisy_labels[train_indices])
    # data_cfg['noise_cfg']['corruption_ratio'] = corruption_ratio

    labels = to_torch(labels, device, indices=True)
    # noisy_labels = to_torch(noisy_labels, device, indices=True)

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
    for key in ['train_loss, train_acc, val_loss, val_acc, test_acc']:
        exp_results[key] = []

    for seed in args.seed:

        set_seed(seed)
        logger = get_logger(f'seed{seed}', workdir)
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
        early_stopping = EarlyStopping(patience=args.patience, criterion='val_loss', delta=0, ckpt_file=ckpt_file, logger=logger)
        train_sampler = IndicesSampler(train_indices, args.batch_size, shuffle=True)
        val_sampler = IndicesSampler(val_indices, args.batch_size, shuffle=False)

        svm_macro_f1_lists = []
        svm_micro_f1_lists = []
        nmi_mean_list = []
        nmi_std_list = []
        ari_mean_list = []
        ari_std_list = []

        for epoch in range(args.epoch):
            t_start = time.time()
            model.train()
            train_loss_list = []
            time_list = []
            accurate_cnt = 0
            for iteration in range(train_sampler.num_iterations()):
                t0 = time.time()
                indices = train_sampler()
                metapath_sampled_list = sample_metapath(indices, metapath_list, args.sample_limit, keep_intermediate=False)
                indices = to_torch(indices, device, indices=True)
                metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                t1 = time.time()
                logits, embeddings = model(
                    indices, metapath_sampled_list, node_type_mapping, node_feature_list
                )
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[indices])
                train_loss_list.append(train_loss.item())
                accurate_cnt += torch.sum(torch.argmax(logp, dim=-1) == labels[indices]).item()

                t2 = time.time()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                time_list.append([
                    t1 - t0, t2 - t1, t3 - t2
                ])
            scheduler.step()
            train_loss = np.mean(train_loss_list)
            train_acc = accurate_cnt / train_indices.shape[0]
            logger.info('Epoch {:05d} | Time {:.4f} | Train_Loss {:.4f} | Train_Accuracy {:.4f}'.format(
                    epoch, np.sum(time_list), train_loss, train_acc))

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
                val_loss = F.nll_loss(val_logp, labels[val_indices])
                val_acc = torch.mean(torch.argmax(val_logp, dim=-1) == labels[val_indices], dtype=torch.float32).item()
            t_end = time.time()
            logger.info('Epoch {:05d} | Time {:.4f} | Val_Loss {:.4f} | Val_Accuracy {:.4f}'.format(
                epoch, t_end - t_start, val_loss.item(), val_acc))
            record = dict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            result = early_stopping.record(record=record, model=model)
            if result is not None:
                best_record, record_msg = result
                for key, value in best_record:
                    exp_results[key].append(value)
                logger.info(f'Early stopping! Best record is {record_msg}')
                break

        # testing with evaluate_results_nc
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
        logger.info(f'test_accuracy {test_acc:.4f}')

        svm_results = svm_test(test_embeddings, labels[test_indices], seed)

        for term, results in svm_results.items():
            print(term)
            for key, value in results.items():
                print(key, value)
        import pdb
        pdb.set_trace()

        # svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
        #     test_embeddings.cpu().numpy(), labels[test_indices].cpu().numpy(), num_classes=num_node_types)
        #
        # svm_macro_f1_lists.append(svm_macro_f1_list)
        # svm_micro_f1_lists.append(svm_micro_f1_list)
        # nmi_mean_list.append(nmi_mean)
        # nmi_std_list.append(nmi_std)
        # ari_mean_list.append(ari_mean)
        # ari_std_list.append(ari_std)
        #
        # # print out a summary of the evaluations
        # svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
        # svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
        # nmi_mean_list = np.array(nmi_mean_list)
        # nmi_std_list = np.array(nmi_std_list)
        # ari_mean_list = np.array(ari_mean_list)
        # ari_std_list = np.array(ari_std_list)
        # print('----------------------------------------------------------------')
        # print('SVM tests summary')
        # print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        #     macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        #     zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
        # print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        #     micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        #     zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
        # print('K-means tests summary')
        # print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
        # print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))
