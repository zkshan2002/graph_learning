import os
import os.path as osp
import pdb
import copy
import time
import json

import numpy as np

from utils.training import get_logger, get_cfg, get_summary_msg
from typing import List, Dict


def run_script(
        exp_tag,
        tag_list: List[str], arg_list: List[dict], group_tag_list: List[str], id2group: Dict[int, int],
        group2id: Dict[int, List[int]]
):
    project_root = osp.realpath('.')
    exp_dir = osp.join(project_root, 'exp', exp_tag)
    for group_tag in group_tag_list:
        workdir = osp.join(exp_dir, group_tag)
        if osp.exists(workdir):
            print(f'Workdir {workdir} exists. Continue?')
            pdb.set_trace()
        else:
            os.makedirs(workdir, exist_ok=False)

    all_cfg = {arg['exp']['tag']: get_cfg(arg) for arg in arg_list}
    with open(osp.join(exp_dir, 'all_cfg.json'), 'w') as f:
        json.dump(all_cfg, f)

    logger_file = osp.join(exp_dir, 'exp.log')
    logger = get_logger('[exp]', logger_file, verbose=True)

    total = len(all_cfg)
    skipped = 0
    time_total = 0
    results = []
    summary_msg_list = []
    for i, cfg in enumerate(all_cfg):
        tag = tag_list[i]
        logger.info(f'Running: {tag}')

        group_tag = group_tag_list[id2group[i]]
        workdir = osp.join(exp_dir, group_tag, tag)
        if osp.exists(workdir):
            logger.info(f'Tag {tag} done. Skip.')
            skipped += 1
            time_elapsed = 0
        else:
            os.makedirs(workdir, exist_ok=False)
            time_start = time.time()
            os.system(f'python3 main.py --workdir="{workdir}"')
            time_end = time.time()
            time_elapsed = time_end - time_start

        with open(osp.join(workdir, 'out.json'), 'r') as f:
            summary_dict = json.load(f)
        results.append(np.array(list(summary_dict.values())))
        summary_msg = get_summary_msg(tag, summary_dict)
        logger.info(summary_msg)
        summary_msg_list.append(summary_msg)

        done = i + 1
        time_total += time_elapsed
        if done == skipped:
            ETA = 0
        else:
            ETA = (total - done) / (done - skipped) * time_total
        logger.info(
            f'{done}/{total} done, ETA {ETA:.2f}, current {time_elapsed:.2f}, total {time_total:.2f}'
        )
    results = np.array(results)

    logger.info(f'Done. Summary:')
    for msg in summary_msg_list:
        logger.info(msg)

    for group_tag, group in zip(group_tag_list, group2id.values()):
        logger.info(f'Group {group_tag}')
        group_indices = np.array(group)
        for i in group:
            logger.info(summary_msg_list[i])
        best_index = np.argmax(results[group_indices, 3])
        best_index = group_indices[best_index]
        logger.info(f'Best is')
        logger.info(summary_msg_list[best_index])
    return


def exp_args():
    dataset = 'DBLP'
    sample_limit = 64
    repeat = 10

    tag_list = []
    arg_list = []
    group_tag_list = []
    id2group = {}
    group2id = {}

    exp_cnt = 0
    group_cnt = 0
    for i, flip_rate in enumerate([0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        base_tag = f'{dataset}{sample_limit}_{repeat}'
        base_arg: dict = dict(
            data=dict(
                dataset=dataset,
            )
        )
        if flip_rate > 0:
            base_tag += f' noise_u {flip_rate}'
            base_arg['data']['noise_cfg'] = dict(
                apply=True,
                flip_rate=flip_rate,
            )

        group2id[group_cnt] = []
        group_tag = f'exp2 {base_tag} sft_filter'
        group_tag_list.append(group_tag)
        memory = 1
        for warmup in [2, 4, 6, 8]:
            tag = f'{group_tag} {memory} {warmup}'
            arg = copy.deepcopy(base_arg)
            arg['train'] = dict(
                sft_cfg=dict(
                    apply_filter=True,
                    filter_cfg=dict(
                        memory=memory,
                        warmup=warmup,
                    ),
                )
            )
            group2id[group_cnt].append(exp_cnt)
            id2group[exp_cnt] = group_cnt
            exp_cnt += 1
            tag_list.append(tag)
            arg['exp'] = dict(
                tag=tag,
                group_tag=group_tag,
            )
            arg_list.append(arg)
        group_cnt += 1
        continue

        group2id[group_cnt] = []
        group_tag = f'exp3 {base_tag} mlc'
        group_tag_list.append(group_tag)
        v_lr = 5e-3
        for T_lr in [0.1, 0.2, 0.5, 1]:
            tag = f'{group_tag} {v_lr} {T_lr}'
            arg = copy.deepcopy(base_arg)
            arg['train'] = dict(
                mlc_cfg=dict(
                    apply=True,
                    v_lr=v_lr,
                    T_lr=T_lr,
                ),
            )
            group2id[group_cnt].append(exp_cnt)
            id2group[exp_cnt] = group_cnt
            exp_cnt += 1
            tag_list.append(tag)
            arg['exp'] = dict(
                tag=tag,
                group_tag=group_tag,
            )
            arg_list.append(arg)
        group_cnt += 1

    return dict(
        tag_list=tag_list,
        arg_list=arg_list,
        group_tag_list=group_tag_list,
        id2group=id2group,
        group2id=group2id
    )


if __name__ == '__main__':
    result = exp_args()
    run_script(exp_tag='current', **result)
