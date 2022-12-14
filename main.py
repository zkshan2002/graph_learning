import os
import os.path as osp
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.training import parse_args, set_seed, get_logger, build_model, EarlyStopping, test, display_result, \
    get_summary_msg
from utils.data import to_np, load_data, split_data, Sampler
from utils.evaluate import evaluate_multiclass
from utils.noisy_labels import apply_label_noise, MemoryBank


def run_exp():
    args = parse_args()
    workdir = args.workdir
    tag = workdir.split('/')[-1]
    with open(osp.join(workdir, f'../../all_cfg.json'), 'r') as f:
        cfg = json.load(f)[tag]

    # summary_dict = {}
    # for key in [
    #     'Train_Macro_F1', 'Train_Micro_F1',
    #     'Val_Macro_F1', 'Val_Micro_F1',
    #     'Test_Macro_F1', 'Test_Micro_F1',
    # ]:
    #     summary_dict[key] = np.random.randint(0, 10000)
    # with open(osp.join(workdir, 'out.json'), 'w') as f:
    #     json.dump(summary_dict, f)
    # return

    if tag == 'debug':
        cfg['exp']['seed_list'] = cfg['exp']['seed_list'][:2]
        cfg['train']['patience'] = 2

    device_id = cfg['exp']['device_id']
    if device_id == -1:
        device = 'cpu'
    else:
        device = f'cuda:{device_id}'

    # manage workdir
    cfg_file = osp.join(workdir, 'cfg.json')
    log_file = osp.join(workdir, 'summary.log')
    logger_summary = get_logger('summary', log_file, cfg['exp']['verbose'])

    # load dataset
    dataset = cfg['data']['dataset']
    data = load_data(dataset, device)

    # split data
    all_train_indices, all_val_indices, all_test_indices = split_data(data, cfg['data'])
    num_train = all_train_indices.shape[0]

    # apply label noise
    is_noisy_label = cfg['data']['noise_cfg'].pop('apply')
    if is_noisy_label:
        noisy_labels, actual_flip_rate = apply_label_noise(
            dataset, data.labels, all_train_indices=all_train_indices, **cfg['data']['noise_cfg'])
        cfg['data']['noise_cfg']['actual_flip_rate'] = actual_flip_rate
    else:
        cfg['data'].pop('noise_cfg')
        noisy_labels = None

    # [SFT]
    sft_cfg = cfg['train'].pop('sft_cfg')
    apply_sft = {'apply': False}
    for key in ['filter', 'loss', 'fixmatch']:
        if sft_cfg[f'apply_{key}']:
            apply_sft[key] = True
            apply_sft['apply'] = True
        else:
            apply_sft[key] = False
            sft_cfg.pop(f'{key}_cfg')
    if apply_sft['apply']:
        # assert is_noisy_label
        cfg['train']['sft_cfg'] = sft_cfg
    if apply_sft['loss'] or apply_sft['fixmatch']:
        assert apply_sft['filter']

    # [MLC]
    mlc_cfg = cfg['train'].pop('mlc_cfg')
    apply_mlc = mlc_cfg['apply']
    if apply_mlc:
        # assert is_noisy_label
        cfg['train']['mlc_cfg'] = mlc_cfg

    model_type = cfg['model'].pop('type')
    cfg['model'] = dict(
        type=model_type,
        cfg=cfg['model'][f'{model_type}_cfg']
    )

    with open(cfg_file, 'w') as f:
        json.dump(cfg, f)

    exp_results = {}
    for key in [
        'Epoch', 'Time',
        'Train_Loss', 'Train_Macro_F1', 'Train_Micro_F1',
        'Val_Loss', 'Val_Macro_F1', 'Val_Micro_F1',
        'Test_Macro_F1', 'Test_Micro_F1',
    ]:
        exp_results[key] = []

    # build model
    model = build_model(
        model_type, cfg['model'],
        data.feature_dim_list, data.num_metapath_schemes, data.num_node_types,
        device,
    )
    if apply_mlc:
        meta_model = build_model(
            model_type, cfg['model'],
            data.feature_dim_list, data.num_metapath_schemes, data.num_node_types,
            device,
        )
    else:
        meta_model = None

    seed_list = cfg['exp']['seed_list']
    num_repeat = len(seed_list)
    for run_id, seed in enumerate(seed_list):
        run_start_timer = time.time()

        set_seed(seed)
        log_file = osp.join(workdir, f'{seed}.log')
        logger = get_logger(f'{run_id + 1}_{num_repeat}_{seed}', log_file, cfg['exp']['verbose'])
        ckpt_file = osp.join(workdir, f'ckpt_seed{seed}.pt')

        model.init_weights()
        optimizer = torch.optim.Adam(model.params(), **cfg['train']['optim_cfg'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg['train']['scheduler_cfg'])

        batch_size = cfg['train']['batch_size']
        sample_limit = cfg['train']['sample_limit']
        if apply_mlc:
            noise_transition_matrix = torch.eye(data.num_cls, device=device)
            noise_transition_matrix = torch.autograd.Variable(noise_transition_matrix, requires_grad=True)
            meta_sampler = Sampler(data, all_val_indices, batch_size, sample_limit, device, shuffle=True)
        else:
            noise_transition_matrix = None
            meta_sampler = None

        early_stopping = EarlyStopping(margin=0, ckpt_file=ckpt_file, **cfg['train']['early_stop_cfg'])
        train_sampler = Sampler(
            data, all_train_indices, batch_size, sample_limit, device, shuffle=True
        )
        val_sampler = Sampler(
            data, all_val_indices, batch_size, sample_limit, device
        )

        if apply_sft['filter']:
            memory_bank = MemoryBank(num_nodes=data.num_target_nodes, device=device, **sft_cfg['filter_cfg'])
        else:
            memory_bank = None

        for epoch_id in range(cfg['train']['epoch']):
            epoch_start_timer = time.time()

            model.train()
            train_loss_list = []
            train_pred_list = []
            train_label_list = []
            fluctuate_cnt = 0
            for iteration in range(train_sampler.num_iterations()):
                train_indices, train_metapath_list = train_sampler.sample()
                if is_noisy_label:
                    train_labels = noisy_labels[train_indices]
                else:
                    train_labels = data.labels[train_indices]

                if apply_mlc:
                    meta_model.load_state_dict(model.state_dict())

                    log_prob_virtual = meta_model.forward(
                        train_indices, train_metapath_list, data.id2type, data.feature_list
                    )
                    log_prob_virtual = torch.matmul(log_prob_virtual, noise_transition_matrix)
                    loss_virtual = F.nll_loss(log_prob_virtual, train_labels)

                    meta_model.zero_grad()
                    grads = torch.autograd.grad(loss_virtual, (meta_model.params()), create_graph=True)
                    meta_model.update_params(grads, mlc_cfg['v_lr'])

                    val_indices, val_metapath_list = meta_sampler.sample()
                    log_prob_meta = meta_model.forward(
                        val_indices, val_metapath_list, data.id2type, data.feature_list
                    )
                    loss_meta = F.nll_loss(log_prob_meta, data.labels[val_indices])

                    grad = torch.autograd.grad(loss_meta, noise_transition_matrix, only_inputs=True)[0]

                    noise_transition_matrix = noise_transition_matrix - grad * mlc_cfg['T_lr']
                    noise_transition_matrix = torch.clamp(noise_transition_matrix, min=0)
                    noise_transition_matrix = noise_transition_matrix / torch.sum(noise_transition_matrix, dim=0)

                log_prob = model.forward(
                    train_indices, train_metapath_list, data.id2type, data.feature_list
                )

                if apply_mlc:
                    log_prob = torch.matmul(log_prob, noise_transition_matrix.detach())

                train_pred = torch.argmax(log_prob, dim=-1)
                accurate_mask = (train_pred == train_labels)
                train_pred_list.append(to_np(train_pred))
                train_label_list.append(to_np(train_labels))

                if apply_sft['filter']:
                    selected_indices, fluctuating_indices = memory_bank.filter(epoch_id, train_indices, accurate_mask)
                    fluctuate_cnt += len(fluctuating_indices)
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
                        if epoch_id < memory_bank.warmup:
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
                            confidence_penalty = -torch.sum(log_prob * pseudo_prob) / data.num_cls
                            confidence_penalty = confidence_penalty * weight[1]

                        train_loss = ce_loss + confidence_penalty

                        if apply_sft['fixmatch']:
                            fixmatch_loss = 0
                            train_loss = train_loss + fixmatch_loss
                    else:
                        train_loss = ce_loss_selected
                else:
                    train_loss = F.nll_loss(log_prob, train_labels)
                train_loss_list.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            scheduler.step()

            fluctuate_ratio = fluctuate_cnt / num_train
            train_loss = np.mean(train_loss_list)
            train_label = np.concatenate(train_label_list)
            train_pred = np.concatenate(train_pred_list)
            train_macro_f1, train_micro_f1 = evaluate_multiclass(train_label, train_pred)

            val_loss, val_macro_f1, val_micro_f1 = test(model, data, val_sampler, all_val_indices)

            msg = f'Epoch {epoch_id:03d} | Train {train_loss:.4f} {train_macro_f1:.4f} {train_micro_f1:.4f}' + \
                  f' | Val {val_loss:.4f} {val_macro_f1:.4f} {val_micro_f1:.4f}'
            if apply_sft['filter']:
                msg += f' | Fluctuate_Ratio {fluctuate_ratio:.3f}'
            if apply_mlc:
                msg += f' | Noise_Transition_Matrix'
                for num in list(noise_transition_matrix.view(-1)):
                    msg += f' {num.item(): .4f}'
            logger.info(msg)
            record = dict(
                Epoch=epoch_id,
                Train_Loss=train_loss,
                Train_Macro_F1=train_macro_f1,
                Train_Micro_F1=train_micro_f1,
                Val_Loss=val_loss,
                Val_Macro_F1=val_macro_f1,
                Val_Micro_F1=val_micro_f1,
            )
            if apply_mlc:
                record['Noise_Transition_Matrix'] = to_np(noise_transition_matrix.reshape(-1))
            best_record, msg = early_stopping.record(record, epoch_id, model)
            logger.info(msg)
            if best_record is not None:
                msg = f"Epoch {best_record['Epoch']:03d} | Train {best_record['Train_Loss']:.4f}" + \
                      f" {best_record['Train_Macro_F1']:.4f} {best_record['Train_Micro_F1']:.4f}" + \
                      f" | Val {best_record['Val_Loss']:.4f} {best_record['Val_Macro_F1']:.4f}" + \
                      f" {best_record['Val_Micro_F1']:.4f}"
                if apply_mlc:
                    msg += ' | Noise_Transition_Matrix'
                    for val in list(best_record['Noise_Transition_Matrix']):
                        msg += f' {val:.4f}'
                logger.info(msg)
                if apply_mlc:
                    best_record.pop('Noise_Transition_Matrix')
                for key, value in best_record.items():
                    exp_results[key].append(value)
                break

            epoch_end_timer = time.time()
            epoch_time = epoch_end_timer - epoch_start_timer

        test_sampler = Sampler(
            data, all_test_indices, batch_size, sample_limit, device
        )
        model.load_state_dict(torch.load(ckpt_file))
        os.remove(ckpt_file)

        _, test_macro_f1, test_micro_f1 = test(model, data, test_sampler, all_test_indices)

        exp_results['Test_Macro_F1'].append(test_macro_f1)
        exp_results['Test_Micro_F1'].append(test_micro_f1)

        logger.info(f'Test | {test_macro_f1:.4f} {test_micro_f1:.4f}')

        run_end_timer = time.time()
        time_elapsed = run_end_timer - run_start_timer
        exp_results['Time'].append(time_elapsed)
        logger.info(f'Run {run_id + 1} / {num_repeat} Ended | Time_elapsed {time_elapsed:.4f}')

    summary_dict = {}
    for key in [
        # 'Epoch', 'Time', 'Train_Loss', 'Val_Loss',
        'Train_Macro_F1', 'Train_Micro_F1',
        'Val_Macro_F1', 'Val_Micro_F1',
        'Test_Macro_F1', 'Test_Micro_F1',
    ]:
        str_dict = display_result(exp_results[key], key, logger_summary)
        summary_dict.update(str_dict)

    logger_summary.info(get_summary_msg(tag, summary_dict))

    with open(osp.join(workdir, 'out.json'), 'w') as f:
        json.dump(summary_dict, f)

    return


if __name__ == '__main__':
    run_exp()
