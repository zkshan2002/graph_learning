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
from utils.data import to_torch, to_np, load_data, sample_metapath
from utils.noisy_labels import apply_label_noise, MemoryBank

from config import exp_cfg, train_cfg, model_cfg, data_cfg

if __name__ == '__main__':

    # override cfg with args
    args = parse_args()
    if hasattr(args, 'tag') and args.tag is not None:
        exp_cfg['tag'] = args.tag
    if hasattr(args, 'description') and args.description is not None:
        exp_cfg['description'] = args.description
    if hasattr(args, 'seed_list') and args.seed_list is not None:
        exp_cfg['seed_list'] = args.seed_list
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if hasattr(args, 'sample_limit') and args.sample_limit is not None:
        train_cfg['sample_limit'] = args.sample_limit
    if hasattr(args, 'noise_p') and args.noise_p is not None:
        data_cfg['noise_cfg']['pair_flip_rate'] = args.noise_p
        data_cfg['noise_cfg']['apply'] = True
    if hasattr(args, 'noise_u') and args.noise_u is not None:
        data_cfg['noise_cfg']['uniform_flip_rate'] = args.noise_u
        data_cfg['noise_cfg']['apply'] = True
    if hasattr(args, 'sft_mb_warmup') and args.sft_mb_warmup is not None:
        train_cfg['sft_cfg']['mb_cfg']['sft_mb_warmup'] = args.sft_mb_warmup
        train_cfg['sft_cfg']['apply'] = True
    if hasattr(args, 'sft_loss_threshold') and args.sft_loss_threshold is not None:
        train_cfg['sft_cfg']['loss_cfg']['threshold'] = args.sft_loss_threshold
        train_cfg['sft_cfg']['apply'] = True
    if hasattr(args, 'sft_loss_weights') and args.sft_loss_weights is not None:
        train_cfg['sft_cfg']['loss_cfg']['penalty_weight'] = args.sft_loss_weights
        train_cfg['sft_cfg']['apply'] = True

    tag = exp_cfg['tag']
    debug = (tag == 'debug')
    if debug:
        exp_cfg['seed_list'] = [100, 200]
        train_cfg['patience'] = 2

    device_id = exp_cfg['device_id']
    if device_id == -1:
        device = 'cpu'
    else:
        device = f'cuda:{device_id}'
    args = Namespace(**train_cfg)
    args.update(**dict(device=device))

    # manage workdir
    project_root = osp.realpath(osp.join(osp.realpath(__file__), '..'))
    workdir = osp.join(project_root, 'exp', tag)
    if not debug and osp.exists(workdir):
        print(f'Workdir exists: {workdir}')
        print('Continue?')
        import pdb
        pdb.set_trace()
    else:
        os.makedirs(workdir, exist_ok=debug)
    cfg_file = osp.join(workdir, 'cfg.json')

    log_file = osp.join(workdir, 'avg_result.log')
    logger_result = get_logger('exp_result', log_file)

    # load dataset
    dataset = data_cfg['dataset']
    metapath_list, node_feature_list, node_type_mapping, adjacency_matrix, labels, (
        train_indices, val_indices, test_indices) = load_data(dataset, project_root)


    # # 1-0
    # metapaths = []
    # for node_id in range(4057):
    #     metapath = np.zeros((10, 2), dtype=np.int32)
    #     metapath[:, 0] = np.arange(10, dtype=np.int32) + 4057
    #     metapath[:, 1] = np.arange(10, dtype=np.int32)
    #     metapaths.append(metapath)
    # metapath_list.append(metapaths)
    # # 0-1: 2 control type(failed)
    # for metapaths in metapath_list:
    #     metapaths.extend([np.zeros((0, 2), dtype=np.int32)] * 14328)
    # metapaths = [np.zeros((0, 2), dtype=np.int32)] * 4057
    # for node_id in range(14328):
    #     metapath = np.zeros((10, 2), dtype=np.int32)
    #     metapath[:, 0] = np.arange(10, dtype=np.int32)
    #     metapath[:, 1] = np.arange(10, dtype=np.int32) + 4057
    #     metapaths.append(metapath)
    # metapath_list.append(metapaths)
    # train_indices = np.concatenate([train_indices, np.arange(14328, dtype=np.int32) + 4057])
    # labels = np.concatenate([labels, np.random.randint(3, size=(14328,))])

    num_metapaths = len(metapath_list)
    node_type_mapping = to_torch(node_type_mapping, device=device)
    num_nodes = labels.shape[0]
    num_node_types = torch.unique(node_type_mapping).shape[0]
    node_feature_dim_list = [node_feature.shape[1] for node_feature in node_feature_list]
    node_feature_list = to_torch(node_feature_list, device)

    # metapath_cnt = np.zeros((num_metapaths, num_nodes), dtype=np.int32)
    # for metapath_id, metapaths in enumerate(metapath_list):
    #     for node, metapath in enumerate(metapaths):
    #         metapath_cnt[metapath_id, node] = metapath.shape[0]

    # split data
    split_seed = data_cfg['split_cfg']['split_seed']
    split = data_cfg['split_cfg']['split']
    # assert np.sum(split) == num_nodes
    if split_seed != -1:
        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        set_seed(split_seed)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:split[0]]
        val_indices = all_indices[split[0]:split[0] + split[1]]
        test_indices = all_indices[split[0] + split[1]:]
    data_cfg['split_cfg']['node_type_cnt'] = {}
    for indices, name in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
        label = labels[indices]
        type_cnt = []
        for i in range(num_node_types):
            type_cnt.append(np.where(label == i)[0].shape[0])
        data_cfg['split_cfg']['node_type_cnt'][name] = type_cnt

    # while 1:
    #     shuffle_seed = np.random.randint(2 ** 31 - 1)
    #     print(f'shuffle seed: {shuffle_seed}')
    #     set_seed(shuffle_seed)
    #     all_indices = np.concatenate([train_indices, val_indices, test_indices])
    #     np.random.shuffle(all_indices)
    #     train_indices = all_indices[0:400]
    #     val_indices = all_indices[400:800]
    #     test_indices = all_indices[800:]
    #     for indices, name in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
    #         label = labels[indices]
    #         type_cnt = []
    #         for i in range(4):
    #             type_cnt.append(np.where(label == i)[0].shape[0])
    #         print(f'{name}: {type_cnt}')
    #     import pdb
    #     pdb.set_trace()

    num_train = train_indices.shape[0]

    # apply label noise
    LNL = data_cfg['noise_cfg'].pop('apply')
    if LNL:
        noisy_labels, label_corruption_mask = apply_label_noise(labels=labels, **data_cfg['noise_cfg'])
        noisy_labels = to_torch(noisy_labels, device, indices=True)

        corruption_ratio = np.mean(label_corruption_mask[train_indices].astype(np.int32))
        data_cfg['noise_cfg']['corruption_ratio'] = corruption_ratio
    else:
        data_cfg.pop('noise_cfg')
        noisy_labels = None

    labels = to_torch(labels, device, indices=True)

    SFT = train_cfg['sft_cfg'].pop('apply')
    if SFT:
        assert LNL
        sft_loss_cfg = train_cfg['sft_cfg']['loss_cfg']
    else:
        train_cfg.pop('sft_cfg')
        sft_loss_cfg = None

    all_cfg = dict(
        exp=exp_cfg,
        train=train_cfg,
        model=model_cfg,
        data=data_cfg,
    )
    with open(cfg_file, 'w') as f:
        json.dump(all_cfg, f)

    model_type = model_cfg.pop('type')
    assert model_type == 'HAN'
    # assert model_type == 'MLP'

    exp_results = {}
    for key in [
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_acc',
        'macro_f1', 'micro_f1', 'time_elapsed'
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
        time_elapsed='Time_Elapsed',
    )

    seed_list = exp_cfg['seed_list']
    num_repeat = len(seed_list)
    for run_id, seed in enumerate(seed_list):

        run_start_timer = time.time()
        set_seed(seed)

        log_file = osp.join(workdir, f'{seed}.log')
        logger = get_logger(f'{run_id}_{num_repeat}_seed{seed}', log_file)
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
        # from model.MLP import MLP
        #
        # model = MLP(
        #     node_raw_feature_dim_list=node_feature_dim_list,
        #     num_cls=num_node_types,
        #     device=device,
        #     **model_cfg
        # )
        optimizer = torch.optim.Adam(model.parameters(), **args.optim_cfg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **args.scheduler_cfg)

        # training utils
        early_stopping = EarlyStopping(
            patience=args.patience, criterion=('val_loss', -1), margin=0, ckpt_file=ckpt_file
        )
        train_sampler = IndicesSampler(train_indices, args.batch_size, shuffle=True)
        val_sampler = IndicesSampler(val_indices, args.batch_size, shuffle=False)

        if SFT:
            num_nodes = np.max(train_indices) + 1
            memory_bank = MemoryBank(num_nodes=num_nodes, device=device, **train_cfg['sft_cfg']['mb_cfg'])
        else:
            memory_bank = None

        for epoch in range(args.epoch):
            model.train()
            train_loss_list = []
            accurate_cnt = 0
            fluctuate_cnt = 0
            epoch_start_timer = time.time()
            for iteration in range(train_sampler.num_iterations()):
                indices = train_sampler()
                metapath_sampled_list = sample_metapath(
                    indices, metapath_list, args.sample_limit, keep_intermediate=False
                )
                indices = to_torch(indices, device, indices=True)
                metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                logits, embeddings = model(
                    indices, metapath_sampled_list, node_type_mapping, node_feature_list
                )
                log_prob = F.log_softmax(logits, 1)

                if LNL:
                    train_labels = noisy_labels[indices]
                else:
                    train_labels = labels[indices]

                accurate_mask = (torch.argmax(log_prob, dim=-1) == train_labels)
                if SFT:
                    selected_indices, fluctuating_indices = memory_bank.filter(epoch, indices, accurate_mask)

                    prob = torch.exp(log_prob)
                    confidence_threshold = sft_loss_cfg['threshold']
                    if epoch < memory_bank.warm_up:
                        ce_loss = F.nll_loss(log_prob, train_labels)
                        top2prob, top2indices = torch.topk(prob, k=2, dim=-1)
                        pseudo_label = top2indices[:, 1]
                        alpha = confidence_threshold - top2prob[:, 1] / top2prob[:, 0]
                        alpha[alpha < 0] = 0
                        confidence_penalty = torch.mean(F.cross_entropy(prob, pseudo_label) * alpha)
                        confidence_penalty = confidence_penalty * sft_loss_cfg['penalty_weight'][0]
                        train_loss = ce_loss + confidence_penalty
                        print(ce_loss.item(), confidence_penalty.item())
                    else:
                        ce_loss = F.nll_loss(log_prob[selected_indices], train_labels[selected_indices])
                        label_prob = prob[train_labels]
                        pseudo_prob = confidence_threshold - prob / label_prob
                        pseudo_prob[pseudo_prob < 0] = 0
                        confidence_penalty = -torch.sum(log_prob * pseudo_prob) / num_node_types
                        confidence_penalty = confidence_penalty * sft_loss_cfg['penalty_weight'][1]
                        train_loss = ce_loss + confidence_penalty
                        print(ce_loss.item(), confidence_penalty.item())

                        # todo: add fix-watch on fluctuating

                        # for debug only, when memory=1
                        if len(memory_bank.memory_bank) > 0:
                            temp = torch.logical_and(
                                memory_bank.memory_bank[0][indices],
                                torch.logical_not(accurate_mask)
                            )
                            assert (torch.where(temp)[0] == fluctuating_indices).all()

                    fluctuate_cnt += len(fluctuating_indices)
                else:
                    train_loss = F.nll_loss(log_prob, train_labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_loss_list.append(train_loss.item())
                accurate_cnt += torch.sum(accurate_mask).item()

            scheduler.step()
            train_loss = np.mean(train_loss_list)
            train_acc = accurate_cnt / num_train

            model.eval()
            val_log_prob = []
            for iteration in range(val_sampler.num_iterations()):
                indices = val_sampler()
                metapath_sampled_list = sample_metapath(
                    indices, metapath_list, args.sample_limit, keep_intermediate=False
                )
                indices = to_torch(indices, device, indices=True)
                metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

                with torch.no_grad():
                    logits, embeddings = model(
                        indices, metapath_sampled_list, node_type_mapping, node_feature_list
                    )
                    log_prob = F.log_softmax(logits, 1)
                val_log_prob.append(log_prob)
            with torch.no_grad():
                val_log_prob = torch.cat(val_log_prob, dim=0)
                val_loss = F.nll_loss(val_log_prob, labels[val_indices])
                val_acc = torch.mean(torch.argmax(val_log_prob, dim=-1) == labels[val_indices], dtype=torch.float32)
            val_loss = val_loss.item()
            val_acc = val_acc.item()
            epoch_end_timer = time.time()
            epoch_time = epoch_end_timer - epoch_start_timer
            fluctuate_ratio = fluctuate_cnt / num_train
            msg = f'Epoch {epoch:03d} | Time {epoch_time:.4f}' + \
                  f' | Train_Loss {train_loss:.4f} | Train_Accuracy {train_acc:.4f}' + \
                  f' | Val_Loss {val_loss:.4f} | Val_Accuracy {val_acc:.4f}'
            if SFT:
                msg += f' | Fluctuate_Ratio {fluctuate_ratio:.3f}'
            logger.info(msg)
            record = dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            best_record, msg = early_stopping.record(record=record, model=model)
            logger.info(msg)
            if best_record is not None:
                msg = [f'Epoch {best_record["epoch"]:03d}']
                best_record.pop('epoch')
                for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    msg.append(f'{display[term]} {best_record[term]:.6f}')
                msg = ' | '.join(msg)
                logger.info(msg)
                for key, value in best_record.items():
                    exp_results[key].append(value)
                break

        test_start_timer = time.time()
        test_sampler = IndicesSampler(test_indices, args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(ckpt_file))
        os.remove(ckpt_file)
        model.eval()
        test_logits = []
        test_embeddings = []
        for iteration in range(test_sampler.num_iterations()):
            indices = test_sampler()
            metapath_sampled_list = sample_metapath(
                indices, metapath_list, args.sample_limit, keep_intermediate=False
            )
            indices = to_torch(indices, device, indices=True)
            metapath_sampled_list = to_torch(metapath_sampled_list, device, indices=True)

            with torch.no_grad():
                logits, embeddings = model(
                    indices, metapath_sampled_list, node_type_mapping, node_feature_list
                )

            test_logits.append(logits)
            test_embeddings.append(embeddings)
        with torch.no_grad():
            test_logits = torch.cat(test_logits, dim=0)
            test_embeddings = torch.cat(test_embeddings, dim=0)
            test_acc = torch.mean(torch.argmax(test_logits, dim=-1) == labels[test_indices], dtype=torch.float32)
        test_acc = test_acc.item()
        test_end_timer = time.time()
        test_time = test_end_timer - test_start_timer
        exp_results['test_acc'].append(test_acc)
        msg = f'Test | Time {test_time:.4f} | Test_Accuracy {test_acc:.6f}'
        logger.info(msg)

        test_embeddings = to_np(test_embeddings)
        test_labels = to_np(labels[test_indices])
        train_ratio_list = exp_cfg['evaluate_cfg']['svm_cfg']['train_ratio_list']

        test_start_timer = time.time()
        svm_results = svm_test(test_embeddings, test_labels, seed, train_ratio_list=train_ratio_list)
        test_end_timer = time.time()
        test_time = test_end_timer - test_start_timer
        msg = f'SVM Test | Time {test_time:.4f}'
        logger.info(msg)

        for key in ['macro_f1', 'micro_f1']:
            value = svm_results[key]
            exp_results[key].append(value['mean'])
            logger.info(value['msg'])

        run_end_timer = time.time()
        time_elapsed = run_end_timer - run_start_timer
        exp_results['time_elapsed'].append(time_elapsed)
        msg = f'Run {run_id} / {num_repeat} Ended | Time_elapsed {time_elapsed:.4f}'
        logger.info(msg)


    def display_result(result, term, logger, display_postfix=''):
        msg = ' | '.join([f'{num:.6f}' for num in result])
        msg = f'{display[term] + display_postfix}: {msg}'
        logger.info(msg)
        mean = np.mean(result)
        std = np.std(result)
        msg = f'{display[term] + display_postfix} Summary: {mean:.4f} ~ {std:.4f}'
        logger.info(msg)
        return


    for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        display_result(exp_results[term], term, logger_result)
    logger_result.info('__________Results__________')
    train_ratio_list = exp_cfg['evaluate_cfg']['svm_cfg']['train_ratio_list']
    for term in ['macro_f1', 'micro_f1']:
        results = np.array(exp_results[term])
        msg = []
        for index, train_ratio in enumerate(train_ratio_list):
            display_result(results[:, index], term, logger_result, display_postfix=f'({train_ratio:.1f})')

    for term in ['test_acc', 'time_elapsed']:
        display_result(exp_results[term], term, logger_result)
