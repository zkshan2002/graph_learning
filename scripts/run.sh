#!/bin/bash

#    ap.add_argument('--tag', type=str)
#    ap.add_argument('--description', type=str)
#    ap.add_argument('--seed_list', nargs='+', type=int)
#    ap.add_argument('--dataset', type=str)
#    ap.add_argument('--batch_size', type=int)
#    ap.add_argument('--sample_limit', type=int)
#    # LNL
#    ap.add_argument('--noise_p', type=float)
#    ap.add_argument('--noise_u', type=float)
#    # SFT
#    ap.add_argument('--sft_filtering_memory', type=int)
#    ap.add_argument('--sft_filtering_warmup', type=int)
#    ap.add_argument('--sft_loss_threshold', type=float)
#    ap.add_argument('--sft_loss_weights', nargs='+', type=float)
#    # MLC
#    ap.add_argument('--mlc_virtual_lr', type=float)
#    ap.add_argument('--mlc_T_lr', type=float)

dataset='IMDB'

python3 main.py --tag="exp0 $dataset" --dataset $dataset

for (( count=1; count<=5; count++ ))
do
  python3 main.py --tag="exp1 $dataset p$count" --dataset $dataset --noise_p "0.$count"
  python3 main.py --tag="exp1 $dataset u$count" --dataset $dataset --noise_u "0.$count"
done
