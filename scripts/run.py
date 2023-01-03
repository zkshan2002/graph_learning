import os
import time


class Timer:
    def __init__(self, total):
        self.cnt = 0
        self.total = total
        self.time_elapsed = 0
        return

    def __call__(self, time_elapsed):
        self.cnt += 1
        self.time_elapsed += time_elapsed
        ETA = (self.total - self.cnt) / self.cnt * self.time_elapsed
        msg = f'{self.cnt}/{self.total} done, ETA {ETA:.2f}, current {time_elapsed:.2f}, total {self.time_elapsed:.2f}'
        return msg


def run_exp(command, timer: Timer):
    print(command)
    time_start = time.time()
    os.system(command)
    time_end = time.time()
    msg = timer(time_end - time_start)
    print(msg)
    return


if __name__ == '__main__':
    timer = Timer(70)

    dataset = 'DBLP'
    sample = 64
    memory = 1
    for noise_u in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = f'"exp3 {dataset}{sample} noise_u {noise_u}"'
        true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                       f' --noise_u {noise_u}'
        command = true_command + f" --command '{true_command}'"
        run_exp(command, timer)
        for warmup in [2, 4, 6, 8]:
            tag = f'"exp3 {dataset}{sample} noise_u {noise_u} sft_filter {memory} {warmup}"'
            true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                           f' --noise_u {noise_u} --sft_filter_memory {memory} --sft_filter_warmup {warmup}'
            command = true_command + f" --command '{true_command}'"
            run_exp(command, timer)

        for v_lr in ['2e-3', '5e-3', '1e-2']:
            for T_lr in ['5e-2', '1e-1', '2e-1']:
                tag = f'"exp4 {dataset}{sample} noise_u {noise_u} mlc {v_lr} {T_lr}"'
                true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                               f' --noise_u {noise_u} --sft_filter_memory {memory} --sft_filter_warmup {warmup}'
                command = true_command + f" --command '{true_command}'"
                run_exp(command, timer)
