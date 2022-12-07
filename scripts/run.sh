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

python3 main.py --tag='exp1 HAN DBLP p1' --dataset 'DBLP' --noise_p 0.1
python3 main.py --tag='exp1 HAN DBLP p2' --dataset 'DBLP' --noise_p 0.2
python3 main.py --tag='exp1 HAN DBLP p3' --dataset 'DBLP' --noise_p 0.3
python3 main.py --tag='exp1 HAN DBLP p4' --dataset 'DBLP' --noise_p 0.4
python3 main.py --tag='exp1 HAN DBLP p5' --dataset 'DBLP' --noise_p 0.5

python3 main.py --tag='exp1 HAN DBLP u1' --dataset 'DBLP' --noise_u 0.1
python3 main.py --tag='exp1 HAN DBLP u2' --dataset 'DBLP' --noise_u 0.2
python3 main.py --tag='exp1 HAN DBLP u3' --dataset 'DBLP' --noise_u 0.3
python3 main.py --tag='exp1 HAN DBLP u4' --dataset 'DBLP' --noise_u 0.4
python3 main.py --tag='exp1 HAN DBLP u5' --dataset 'DBLP' --noise_u 0.5