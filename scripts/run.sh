#!/bin/bash

#    ap.add_argument('--tag', type=str, default='debug')
#    ap.add_argument('--description', type=str, default='')
#    ap.add_argument('--seed_list', nargs='+', type=int)
#    ap.add_argument('--batch_size', type=int)
#    ap.add_argument('--sample_limit', type=int)
#    ap.add_argument('--noise_p', type=float)
#    ap.add_argument('--noise_u', type=float)
#    ap.add_argument('--sft_filtering_memory', type=int)
#    ap.add_argument('--sft_filtering_warmup', type=int)
#    ap.add_argument('--sft_loss_threshold', type=float)
#    ap.add_argument('--sft_loss_weights', nargs='+', type=float)

#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2 IMDB' --sft_filtering_memory 1 --sft_filtering_warmup 2
#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_4 IMDB' --sft_filtering_memory 1 --sft_filtering_warmup 4
#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_8 IMDB' --sft_filtering_memory 1 --sft_filtering_warmup 8
#python3 main.py --tag='exp5 HAN_p4 SFT_F_2_2 IMDB' --sft_filtering_memory 2 --sft_filtering_warmup 2
#python3 main.py --tag='exp5 HAN_p4 SFT_F_2_4 IMDB' --sft_filtering_memory 2 --sft_filtering_warmup 4
#python3 main.py --tag='exp5 HAN_p4 SFT_F_2_8 IMDB' --sft_filtering_memory 2 --sft_filtering_warmup 8
#python3 main.py --tag='exp5 HAN_p4 SFT_F_3_2 IMDB' --sft_filtering_memory 3 --sft_filtering_warmup 2
#python3 main.py --tag='exp5 HAN_p4 SFT_F_4_2 IMDB' --sft_filtering_memory 4 --sft_filtering_warmup 2

#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_020501 IMDB' \
#--sft_filtering_memory 1 --sft_filtering_warmup 2 \
#--sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.1
#
#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_050501 IMDB' \
#--sft_filtering_memory 1 --sft_filtering_warmup 2 \
#--sft_loss_threshold 0.5 --sft_loss_weights 0.5 0.1
#
#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_0205005 IMDB' \
#--sft_filtering_memory 1 --sft_filtering_warmup 2 \
#--sft_loss_threshold 0.2 --sft_loss_weights 0.5 0.05
#
#python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_0505005 IMDB' \
#--sft_filtering_memory 1 --sft_filtering_warmup 2 \
#--sft_loss_threshold 0.5 --sft_loss_weights 0.5 0.05

python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_06705005 IMDB' \
--sft_filtering_memory 1 --sft_filtering_warmup 2 \
--sft_loss_threshold 0.67 --sft_loss_weights 0.5 0.05

python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_07505005 IMDB' \
--sft_filtering_memory 1 --sft_filtering_warmup 2 \
--sft_loss_threshold 0.75 --sft_loss_weights 0.5 0.05

python3 main.py --tag='exp5 HAN_p4 SFT_F_1_2_1005005 IMDB' \
--sft_filtering_memory 1 --sft_filtering_warmup 2 \
--sft_loss_threshold 1.0 --sft_loss_weights 0.5 0.05