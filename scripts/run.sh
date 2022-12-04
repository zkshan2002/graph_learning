#!/bin/bash

#    ap.add_argument('--tag', type=str, default='debug')
#    ap.add_argument('--description', type=str, default='')
#    ap.add_argument('--seed_list', nargs='+', type=int)
#    ap.add_argument('--batch_size', type=int)
#    ap.add_argument('--sample_limit', type=int)
#    ap.add_argument('--noise_p', type=float)
#    ap.add_argument('--noise_u', type=float)
#    ap.add_argument('--sft_mb_warmup', type=int)
#    ap.add_argument('--sft_loss_threshold', type=float)
#    ap.add_argument('--sft_loss_weights', nargs='+', type=float)

python3 main.py --tag='exp4 HAN_p1 IMDB' --noise_p 0.1 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_p2 IMDB' --noise_p 0.2 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_p3 IMDB' --noise_p 0.3 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_p4 IMDB' --noise_p 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_p5 IMDB' --noise_p 0.5 \
--seed_list 100 200 300 400 500 600 700 800 900 1000

python3 main.py --tag='exp4 HAN_u1 IMDB' --noise_u 0.1 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_u2 IMDB' --noise_u 0.2 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_u3 IMDB' --noise_u 0.3 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_u4 IMDB' --noise_u 0.4 \
--seed_list 100 200 300 400 500 600 700 800 900 1000
python3 main.py --tag='exp4 HAN_u5 IMDB' --noise_u 0.5 \
--seed_list 100 200 300 400 500 600 700 800 900 1000