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
    timer = Timer(1)

    dataset = 'DBLP'
    sample = 64
    memory = 1
    v_lr = 1e-3

    # tag = f'"exp1 {dataset}{sample}"'
    # true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}'
    # command = true_command + f" --command '{true_command}'"
    # run_exp(command, timer)

    for flip_rate in [0.4]:  # [0.1, 0.2, 0.3, 0.4, 0.5]:
        tag = f'"exp1 {dataset}{sample} noise_u {flip_rate}"'
        true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                       f' --noise_type uniform --flip_rate {flip_rate}'
        command = true_command + f" --command '{true_command}'"
        run_exp(command, timer)
        exit()
        for warmup in [2, 4, 6, 8]:
            tag = f'"exp1 {dataset}{sample} noise_u {flip_rate} sft_filter {memory} {warmup}"'
            true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                           f' --noise_type uniform --flip_rate {flip_rate}' + \
                           f' --sft_filter_memory {memory} --sft_filter_warmup {warmup}'
            command = true_command + f" --command '{true_command}'"
            run_exp(command, timer)
            break
        for T_lr in [0.05, 0.10, 0.20, 0.5, 1, 2, 5]:
            tag = f'"exp1 {dataset}{sample} noise_u {flip_rate} mlc {T_lr}"'
            true_command = f'python3 main.py --tag={tag} --dataset {dataset} --sample_limit {sample}' + \
                           f' --noise_type uniform --flip_rate {flip_rate}' + \
                           f' --mlc_virtual_lr {v_lr} --mlc_T_lr {T_lr}'
            command = true_command + f" --command '{true_command}'"
            run_exp(command, timer)
            break
        break

