#!/bin/bash

#    ap.add_argument('--tag', type=str, default='debug')
#    ap.add_argument('--description', type=str, default='')
#    ap.add_argument('--seed_list', nargs='+', type=int)
#    ap.add_argument('--noise_p', type=float)
#    ap.add_argument('--noise_u', type=float)
#    ap.add_argument('--sft_mb_warmup', type=int)
#    ap.add_argument('--sft_loss_threshold', type=float)
#    ap.add_argument('--sft_loss_weights', nargs='+', type=float)

python3 main.py --tag='exp4p HAN' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000

python3 main.py --tag='exp4p HAN SFT_1_2_05_01' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000 \
--sft_mb_warmup 1 --sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1

python3 main.py --tag='exp4p HAN SFT_2_2_05_01' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000 \
--sft_mb_warmup 2 --sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1

python3 main.py --tag='exp4p HAN SFT_3_2_05_01' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000 \
--sft_mb_warmup 3 --sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1

python3 main.py --tag='exp4p HAN SFT_4_2_05_01' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000 \
--sft_mb_warmup 4 --sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1

python3 main.py --tag='exp4p HAN SFT_5_2_05_01' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000 \
--sft_mb_warmup 5 --sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1