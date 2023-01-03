import copy
import os
import os.path as osp
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.training import parse_args, override_cfg, set_seed, get_logger, build_model
from utils.training import EarlyStopping, Sampler, Namespace
from utils.data import to_torch, to_np, load_data
from utils.evaluate import svm_test
from utils.noisy_labels import apply_label_noise, MemoryBank


def main():
    # override cfg with args
    from config import exp_cfg, train_cfg, model_cfg, data_cfg
    args = parse_args()
    override_cfg(args, exp_cfg, train_cfg, model_cfg, data_cfg)

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

    log_file = osp.join(workdir, 'results.log')
    logger_result = get_logger('exp_result', log_file, exp_cfg['verbose'])

    # load dataset
    dataset = data_cfg['dataset']
    metapath_list, node_feature_list, node_type_mapping, adjacency_matrix, labels, (
        train_indices, val_indices, test_indices) = load_data(dataset, project_root)

    num_cls = np.max(labels) + 1
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
    all_indices = np.concatenate([train_indices, val_indices, test_indices])
    num_total = all_indices.shape[0]
    re_split = data_cfg['split_cfg'].pop('apply')
    if re_split:
        split_seed = data_cfg['split_cfg']['seed']
        split_ratio = data_cfg['split_cfg']['split_ratio']
        num_train = int(num_total * split_ratio[0])
        num_val = int(num_total * split_ratio[1])
        num_test = num_total - num_train - num_val
        set_seed(split_seed)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:num_train]
        val_indices = all_indices[num_train:num_train + num_val]
        test_indices = all_indices[num_train + num_val:]
    else:
        num_train = train_indices.shape[0]
        num_val = val_indices.shape[0]
        num_test = test_indices.shape[0]
        data_cfg['split_cfg'].pop('seed')
        split_ratio = np.array([num_train, num_val, num_test], dtype=np.float32)
        split_ratio /= np.sum(split_ratio)
        data_cfg['split_cfg']['split_ratio'] = list(split_ratio)
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

    # apply label noise
    LNL = data_cfg['noise_cfg'].pop('apply')
    if LNL:
        noisy_labels, label_corruption_mask = apply_label_noise(dataset=dataset, labels=labels, **data_cfg['noise_cfg'])
        noisy_labels = to_torch(noisy_labels, device, indices=True)

        corruption_ratio = np.sum(label_corruption_mask[train_indices]) / num_train
        data_cfg['noise_cfg']['corruption_ratio'] = corruption_ratio
    else:
        data_cfg.pop('noise_cfg')
        noisy_labels = None

    labels = to_torch(labels, device, indices=True)

    # [SFT]
    sft_cfg = train_cfg.pop('sft_cfg')
    apply_sft = {'apply': False}
    for key in ['filtering', 'loss', 'fixmatch']:
        if sft_cfg[f'apply_{key}']:
            apply_sft[key] = True
            apply_sft['apply'] = True
        else:
            apply_sft[key] = False
            sft_cfg.pop(f'{key}_cfg')
    if apply_sft['apply']:
        # assert LNL
        train_cfg['sft_cfg'] = sft_cfg
    if apply_sft['loss'] or apply_sft['fixmatch']:
        assert apply_sft['filtering']

    # [MLC]
    mlc_cfg = train_cfg.pop('mlc_cfg')
    apply_mlc = mlc_cfg['apply']
    if apply_mlc:
        assert LNL
        train_cfg['mlc_cfg'] = mlc_cfg

    model_type = model_cfg.pop('type')
    assert model_type == 'HAN'
    model_cfg = dict(
        type=model_type,
        cfg=model_cfg[f'{model_type}_cfg']
    )

    all_cfg = dict(
        exp=exp_cfg,
        train=train_cfg,
        model=model_cfg,
        data=data_cfg,
    )
    with open(cfg_file, 'w') as f:
        json.dump(all_cfg, f)

    exp_results = {}
    for key in [
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_acc',
        'macro_f1', 'micro_f1', 'time_elapsed', 'confusion_mat',
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
        confusion_mat='Confusion Matrix',
    )

    # build model
    model = build_model(
        model_type, model_cfg,
        node_feature_dim_list, num_metapaths, num_node_types,
        device,
    )

    seed_list = exp_cfg['seed_list']
    num_repeat = len(seed_list)
    for run_id, seed in enumerate(seed_list):

        run_start_timer = time.time()
        set_seed(seed)

        log_file = osp.join(workdir, f'{seed}.log')
        logger = get_logger(f'{run_id}_{num_repeat}_seed{seed}', log_file, exp_cfg['verbose'])
        ckpt_file = osp.join(workdir, f'ckpt_seed{seed}.pt')

        model.init_weights()

        optimizer = torch.optim.Adam(model.params(), **args.optim_cfg)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **args.scheduler_cfg)

        if apply_mlc:
            noise_transition_matrix = torch.eye(num_node_types, device=device)
            noise_transition_matrix = torch.autograd.Variable(noise_transition_matrix, requires_grad=True)
            val_sampler_meta = Sampler(
                metapath_list, val_indices, args.batch_size, args.sample_limit, device, shuffle=True
            )
        else:
            noise_transition_matrix = None
            val_sampler_meta = None

        # training utils
        early_stopping = EarlyStopping(
            patience=args.patience, criterion=('val_loss', -1), margin=0, ckpt_file=ckpt_file
        )
        train_sampler = Sampler(
                metapath_list, train_indices, args.batch_size, args.sample_limit, device, shuffle=True
        )
        val_sampler = Sampler(
                metapath_list, val_indices, args.batch_size, args.sample_limit, device, shuffle=False
        )

        if apply_sft['filtering']:
            memory_bank = MemoryBank(num_nodes=num_nodes, device=device, **sft_cfg['filtering_cfg'])
        else:
            memory_bank = None

        for epoch in range(args.epoch):
            model.train()
            train_loss_list = []
            accurate_cnt = 0
            fluctuate_cnt = 0
            epoch_start_timer = time.time()
            for iteration in range(train_sampler.num_iterations()):

                train_nodes, train_metapath_list = train_sampler.sample()

                if LNL:
                    train_labels = noisy_labels[train_nodes]
                else:
                    train_labels = labels[train_nodes]

                if apply_mlc:
                    meta_model = build_model(
                        model_type, model_cfg,
                        node_feature_dim_list, num_metapaths, num_node_types,
                        device,
                    )
                    meta_model.load_state_dict(model.state_dict())

                    logits_virtual, _ = meta_model(
                        train_nodes, train_metapath_list, node_type_mapping, node_feature_list
                    )
                    log_prob_virtual = F.log_softmax(logits_virtual, 1)
                    log_prob_virtual = torch.matmul(log_prob_virtual, noise_transition_matrix)
                    loss_virtual = F.nll_loss(log_prob_virtual, train_labels)

                    meta_model.zero_grad()
                    grads = torch.autograd.grad(loss_virtual, (meta_model.params()), create_graph=True)
                    meta_model.update_params(grads, mlc_cfg['virtual_lr'])

                    # def set_param(module, name_list, value):
                    #     next_module = getattr(module, name_list[0])
                    #     import pdb
                    #     pdb.set_trace()
                    #     if isinstance(next_module, torch.nn.Parameter):
                    #         assert len(name_list) == 1
                    #         # next_module.data = value
                    #         setattr(next_module, 'data', value)
                    #     else:
                    #         assert len(name_list) > 1
                    #         set_param(next_module, name_list[1:], value)
                    #     return

                    # for index, (name, param) in enumerate(meta_model.named_parameters()):
                    # grad = torch.autograd.Variable(grads[index].detach().data)
                    # new_value = param - grad * 1e-3
                    # new_value = param - grads[index] * 1e-3
                    # name_list = name.split('.')
                    # set_param(meta_model, name_list, new_value)
                    # local_grad = torch.autograd.grad(new_value, noise_transition_matrix)
                    # local_grad_dict[name] = torch
                    # db(new_value)
                    # db(meta_model.node_feature_projector_list[0].weight)

                    # for index, (name, param) in enumerate(meta_model.named_parameters()):
                    #     print(f'{index} {name} {param.mean()}')

                    val_nodes, val_metapath_list = val_sampler_meta.sample()
                    logits_meta, _ = meta_model(
                        val_nodes, val_metapath_list, node_type_mapping, node_feature_list
                    )

                    log_prob_meta = F.log_softmax(logits_meta, 1)
                    loss_meta = F.nll_loss(log_prob_meta, labels[val_nodes])

                    grad = torch.autograd.grad(loss_meta, noise_transition_matrix, only_inputs=True)[0]

                    noise_transition_matrix = noise_transition_matrix - grad * mlc_cfg['T_lr']
                    noise_transition_matrix = torch.clamp(noise_transition_matrix, min=0)
                    noise_transition_matrix = noise_transition_matrix / torch.sum(noise_transition_matrix, dim=0)

                logits, embeddings = model(
                    train_nodes, train_metapath_list, node_type_mapping, node_feature_list
                )
                log_prob = F.log_softmax(logits, 1)

                if apply_mlc:
                    log_prob = torch.matmul(log_prob, noise_transition_matrix.detach())

                accurate_mask = (torch.argmax(log_prob, dim=-1) == train_labels)

                if apply_sft['filtering']:
                    selected_indices, fluctuating_indices = memory_bank.filter(epoch, train_nodes, accurate_mask)

                    fluctuate_cnt += len(fluctuating_indices)
                    # for debug only, when memory=1
                    # if len(memory_bank.memory_bank) > 0 and epoch >= memory_bank.warmup:
                    #     temp = torch.logical_and(
                    #         memory_bank.memory_bank[0][indices],
                    #         torch.logical_not(accurate_mask)
                    #     )
                    #     assert (torch.where(temp)[0] == fluctuating_indices).all()

                    ce_loss_selected = F.nll_loss(log_prob[selected_indices], train_labels[selected_indices])
                    if len(fluctuating_indices) > 0:
                        ce_loss_fluctuating = F.nll_loss(log_prob[fluctuating_indices],
                                                         train_labels[fluctuating_indices])
                    else:
                        ce_loss_fluctuating = 0

                    if apply_sft['loss']:
                        loss_cfg = sft_cfg['loss_cfg']
                        confidence_threshold = loss_cfg['threshold']
                        weight = loss_cfg['weight']

                        prob = torch.exp(log_prob)
                        if epoch < memory_bank.warmup:
                            ce_loss = ce_loss_selected + ce_loss_fluctuating

                            top2prob, top2indices = torch.topk(prob, k=2, dim=-1)
                            pseudo_label = top2indices[:, 1]
                            alpha = confidence_threshold - top2prob[:, 1] / top2prob[:, 0]
                            alpha[alpha < 0] = 0
                            confidence_penalty = torch.mean(F.cross_entropy(prob, pseudo_label) * alpha)
                            confidence_penalty = confidence_penalty * weight[0]
                        else:
                            ce_loss = ce_loss_selected

                            label_prob = prob[train_labels]
                            pseudo_prob = confidence_threshold - prob / label_prob
                            pseudo_prob[pseudo_prob < 0] = 0
                            confidence_penalty = -torch.sum(log_prob * pseudo_prob) / num_node_types
                            confidence_penalty = confidence_penalty * weight[1]

                        train_loss = ce_loss + confidence_penalty
                        # print(ce_loss.item(), confidence_penalty.item())

                        if apply_sft['fixmatch']:
                            fixmatch_loss = 0
                            train_loss = train_loss + fixmatch_loss
                    else:
                        train_loss = ce_loss_selected
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
                val_nodes, val_metapath_list = val_sampler.sample()

                with torch.no_grad():
                    logits, embeddings = model(
                        val_nodes, val_metapath_list, node_type_mapping, node_feature_list
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
            if apply_sft['filtering']:
                msg += f' | Fluctuate_Ratio {fluctuate_ratio:.3f}'
            if apply_mlc:
                msg += f' | Noise_Transition_Matrix'
                for num in list(noise_transition_matrix.view(-1)):
                    msg += f' {num.item(): .4f}'
            logger.info(msg)
            record = dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            if apply_mlc:
                record['T'] = to_np(noise_transition_matrix.clone().reshape(-1))
            best_record, msg = early_stopping.record(record=record, model=model)
            logger.info(msg)
            if best_record is not None:
                msg = [f'Epoch {best_record["epoch"]:03d}']
                best_record.pop('epoch')
                for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
                    msg.append(f'{display[term]} {best_record[term]:.6f}')
                msg = ' | '.join(msg)
                if apply_mlc:
                    msg += ' | Noise_Transition_Matrix'
                    for val in list(best_record['T']):
                        msg += f' {val.item():.4f}'
                logger.info(msg)
                for key, value in best_record.items():
                    if key != 'T':
                        exp_results[key].append(value)
                break

        test_start_timer = time.time()
        test_sampler = Sampler(
                metapath_list, test_indices, args.batch_size, args.sample_limit, device, shuffle=False
        )
        model.load_state_dict(torch.load(ckpt_file))
        os.remove(ckpt_file)
        model.eval()
        test_logits = []
        test_embeddings = []
        for iteration in range(test_sampler.num_iterations()):
            test_nodes, test_metapath_list = test_sampler.sample()

            with torch.no_grad():
                logits, embeddings = model(
                    test_nodes, test_metapath_list, node_type_mapping, node_feature_list
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

        test_start_timer = time.time()
        svm_results = svm_test(test_embeddings, test_labels, seed=seed, **exp_cfg['evaluate_cfg'])
        test_end_timer = time.time()
        test_time = test_end_timer - test_start_timer
        msg = f'SVM Test | Time {test_time:.4f}'
        logger.info(msg)

        for key in ['macro_f1', 'micro_f1', 'confusion_mat']:
            value = svm_results[key]
            exp_results[key].append(value['mean'])
            logger.info(value['msg'])

        run_end_timer = time.time()
        time_elapsed = run_end_timer - run_start_timer
        exp_results['time_elapsed'].append(time_elapsed)
        msg = f'Run {run_id} / {num_repeat} Ended | Time_elapsed {time_elapsed:.4f}'
        logger.info(msg)


    def display_result(result, term, logger):
        msg = ' | '.join([f'{num:.6f}' for num in result])
        msg = f'{display[term]}: {msg}'
        logger.info(msg)
        mean = np.mean(result)
        std = np.std(result)
        msg = f'{display[term]} Summary: {mean:.6f} ~ {std:.6f}'
        logger.info(msg)
        summary = {term: f'{mean:.4f}'}
        return summary

    summary_dict = {}
    for term in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        summary = display_result(exp_results[term], term, logger_result)
        summary_dict.update(summary)
    logger_result.info('__________Results__________')
    for term in ['macro_f1', 'micro_f1', 'test_acc', 'time_elapsed']:
        summary = display_result(exp_results[term], term, logger_result)
        summary_dict.update(summary)

    result = np.stack(exp_results['confusion_mat'])
    result = np.sum(result, axis=0)
    msg = 'Confusion Matrix:\n'
    for i, val in enumerate(result):
        msg += f'{val:.2f},'
        if (i + 1) % num_cls == 0:
            msg += '\n'
    logger_result.info(msg)

    summary_msg = ['', tag]
    for term in ['macro_f1', 'micro_f1', 'test_acc']:
        summary_msg.append(summary_dict[term])
    summary_msg.append('')

    summary_msg = ' | '.join(summary_msg)
    logger_result.info(summary_msg)


if __name__ == '__main__':
    main()
