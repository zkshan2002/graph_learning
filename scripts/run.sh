#!/bin/bash

#    ap.add_argument('--tag', type=str)
#    ap.add_argument('--description', type=str)
#    ap.add_argument('--seed_list', nargs='+', type=int)
#    ap.add_argument('--dataset', type=str)
#    ap.add_argument('--batch_size', type=int)
#    ap.add_argument('--sample_limit', type=int)
#
#    ap.add_argument('--noise_p', type=float)
#    ap.add_argument('--noise_u', type=float)
#
#    ap.add_argument('--sft_filtering_memory', type=int)
#    ap.add_argument('--sft_filtering_warmup', type=int)
#    ap.add_argument('--sft_loss_threshold', type=float)
#    ap.add_argument('--sft_loss_weights', nargs='+', type=float)

#python3 main.py --tag='exp2 HAN IMDB SFT_1_4' --dataset 'IMDB' \
#--sft_filtering_memory 1 --sft_filtering_warmup 4
#
#python3 main.py --tag='exp2 HAN IMDB p1 SFT_1_4' --dataset 'IMDB' --noise_p 0.1 \
#--sft_filtering_memory 1 --sft_filtering_warmup 4
#
#python3 main.py --tag='exp2 HAN IMDB p2 SFT_1_4' --dataset 'IMDB' --noise_p 0.2 \
#--sft_filtering_memory 1 --sft_filtering_warmup 4
#
#python3 main.py --tag='exp2 HAN IMDB p3 SFT_1_4' --dataset 'IMDB' --noise_p 0.3 \
#--sft_filtering_memory 1 --sft_filtering_warmup 4
#
#python3 main.py --tag='exp2 HAN IMDB p4 SFT_1_4' --dataset 'IMDB' --noise_p 0.4 \
#--sft_filtering_memory 1 --sft_filtering_warmup 4
#
#python3 main.py --tag='exp2 HAN IMDB p5 SFT_1_4' --dataset 'IMDB' --noise_p 0.5 \
#--sft_filtering_memory 1 --sft_filtering_warmup 4


python3 main.py --tag='exp2 HAN IMDB u1 SFT_1_4' --dataset 'IMDB' --noise_u 0.1 \
--sft_filtering_memory 1 --sft_filtering_warmup 4

python3 main.py --tag='exp2 HAN IMDB u2 SFT_1_4' --dataset 'IMDB' --noise_u 0.2 \
--sft_filtering_memory 1 --sft_filtering_warmup 4

python3 main.py --tag='exp2 HAN IMDB u3 SFT_1_4' --dataset 'IMDB' --noise_u 0.3 \
--sft_filtering_memory 1 --sft_filtering_warmup 4

python3 main.py --tag='exp2 HAN IMDB u4 SFT_1_4' --dataset 'IMDB' --noise_u 0.4 \
--sft_filtering_memory 1 --sft_filtering_warmup 4

python3 main.py --tag='exp2 HAN IMDB u5 SFT_1_4' --dataset 'IMDB' --noise_u 0.5 \
--sft_filtering_memory 1 --sft_filtering_warmup 4