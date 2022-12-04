import os
import os.path as osp

from typing import Callable


def visit_recursive(root_dir: str, action: Callable[[str], None]):
    entry_list = sorted(os.listdir(root_dir))
    for entry_name in entry_list:
        entry = osp.join(root_dir, entry_name)
        if osp.isdir(entry):
            visit_recursive(entry, action)
        if osp.isfile(entry):
            action(entry)
    return


def remove_ckpt(entry: str):
    if entry.endswith('.pt'):
        os.remove(entry)
    return


if __name__ == '__main__':
    project_dir = osp.realpath(osp.join(osp.realpath(__file__), '..'))
    # exp_dir = osp.join(project_dir, 'exp')
    # visit_recursive(exp_dir, remove_ckpt)
