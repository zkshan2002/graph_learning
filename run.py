import os
import os.path as osp
import time

import numpy as np

from main import run_exp

from typing import List, Dict


def run_script(
        tag_list: List[str], arg_list: List[dict], group_tag_list: List[str], id2group: Dict[int, int],
        group2id: Dict[int, List[int]]
):
    def print_result(tag, summary_dict: Dict[str, float]):
        summary_msg = f"| {tag} | {summary_dict['Train_Macro_F1']} {summary_dict['Train_Micro_F1']} |" + \
                      f" {summary_dict['Val_Macro_F1']} {summary_dict['Val_Micro_F1']} |" + \
                      f" {summary_dict['Test_Macro_F1']} {summary_dict['Test_Micro_F1']} |"
        return summary_msg

    project_root = osp.realpath('.')
    exp_dir = osp.join(project_root, 'exp/current')
    for group_tag in group_tag_list:
        workdir = osp.join(exp_dir, group_tag)
        if osp.exists(workdir):
            print(f'Workdir {workdir} exists. Continue?')
            import pdb

            pdb.set_trace()
        else:
            os.makedirs(workdir, exist_ok=False)

    total = len(arg_list)
    time_total = 0
    results = []
    summary_msg_list = []
    for i, arg in enumerate(arg_list):
        tag = tag_list[i]
        print(f'Running: {tag}')

        time_start = time.time()
        group_tag = group_tag_list[id2group[i]]
        workdir = osp.join(exp_dir, group_tag, tag)
        os.makedirs(workdir, exist_ok=False)
        summary_dict = run_exp(arg, workdir)
        time_end = time.time()

        results.append(np.array(list(summary_dict.values())))
        summary_msg = print_result(tag, summary_dict)
        print(summary_msg)
        summary_msg_list.append(summary_msg)

        time_elapsed = time_end - time_start
        done = i + 1
        time_total += time_elapsed
        ETA = (total - done) / done * time_total
        msg = f'{done}/{total} done, ETA {ETA:.2f}, current {time_elapsed:.2f}, total {time_total:.2f}'
        print(msg)
    results = np.array(results)

    print(f'Done. Summary:')
    for msg in summary_msg_list:
        print(msg)

    for group_tag, group in zip(group_tag_list, group2id.values()):
        print(f'Group {group_tag}')
        group_indices = np.array(group)
        result = results[group_indices]
        for i in group:
            print(summary_msg_list[i])
        best_index = np.argmax(result[:, 3])
        best_index = group_indices[best_index]
        print(f'Best is')
        print(summary_msg_list[best_index])
    return


def exp_args():
    dataset = 'IMDB'
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
        group_tag = f'exp2 {dataset}{sample_limit}_{repeat}'
        group2id[group_cnt] = []

        base_tag = f'exp2 {dataset}{sample_limit}_{repeat}'
        arg: dict = dict(
            data=dict(
                dataset='IMDB',
            )
        )
        if flip_rate > 0:
            base_tag += f' noise_u {flip_rate}'
            arg['data']['noise_cfg'] = dict(
                apply=True,
                flip_rate=flip_rate,
            )
            group_tag += f' noise_u {flip_rate}'
        group_tag += ' sft_filter'
        group_tag_list.append(group_tag)

        memory = 1
        for warmup in [2, 4, 6, 8]:
            group2id[group_cnt].append(exp_cnt)
            id2group[exp_cnt] = group_cnt
            exp_cnt += 1

            tag = f'{base_tag} sft_filter {memory} {warmup}'
            arg['train'] = dict(
                sft_cfg=dict(
                    apply_filter=True,
                    filter_cfg=dict(
                        memory=memory,
                        warmup=warmup,
                    ),
                )
            )

            tag_list.append(tag)
            arg['exp'] = dict(
                tag=tag,
                group_tag=group_tag,
            )
            arg_list.append(arg)
        group_cnt += 1
    return tag_list, arg_list, group_tag_list, id2group, group2id


if __name__ == '__main__':
    result = exp_args()
    run_script(*result)
