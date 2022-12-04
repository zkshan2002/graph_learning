## Questions
- exp
  - verify ok:
    - node split 400/400/3257
      - transductive inference, train set ok?
      - clean validation set too large?
      - fixed at multiple runs?
    - node label skew (1197, 745, 1109, 1006)
    - training
      - almost convergent in < 5 epochs, then jitters about
        - patience matters a lot in the long jitter period
        - tweaking down lr delays convergent time, but still jitters
        - 
      - quite sensitive to random seed, can result in ~1% f1 or ~2% accuracy
        - repeat 10 times with difference seed and report average results


## All Experiments
### Group 1
#### exp0 HAN_batch_sample
|              Tag | Method | Batch_Size |    Macro-F1     |    Micro-F1     |    Accuracy     |
|-----------------:|:------:|-----------:|:---------------:|:---------------:|:---------------:|
|   exp0 HAN_8_100 |  HAN*  |          8 | 0.9301 ~ 0.0036 | 0.9356 ~ 0.0035 | 0.9287 ~ 0.0022 |
|  exp0 HAN_16_100 |  HAN*  |         16 | 0.9323 ~ 0.0043 | 0.9375 ~ 0.0041 | 0.9340 ~ 0.0056 |
|  exp0 HAN_32_100 |  HAN*  |         32 | 0.9337 ~ 0.0040 | 0.9387 ~ 0.0038 | 0.9354 ~ 0.0084 |
|  exp0 HAN_64_100 |  HAN*  |         64 | 0.9345 ~ 0.0057 | 0.9394 ~ 0.0056 | 0.9351 ~ 0.0067 |
| exp0 HAN_128_100 |  HAN*  |        128 | 0.9327 ~ 0.0035 | 0.9378 ~ 0.0034 | 0.9334 ~ 0.0060 |

|              Tag | Method | Sample_Limit |    Macro-F1     |    Micro-F1     |    Accuracy     |
|-----------------:|:------:|-------------:|:---------------:|:---------------:|:---------------:|
|   exp0 HAN_64_64 |  HAN*  |           64 | 0.9291 ~ 0.0034 | 0.9342 ~ 0.0035 | 0.9298 ~ 0.0057 |
|  exp0 HAN_64_128 |  HAN*  |          128 | 0.9319 ~ 0.0041 | 0.9370 ~ 0.0036 | 0.9308 ~ 0.0046 |
|  exp0 HAN_64_256 |  HAN*  |          256 | 0.9359 ~ 0.0040 | 0.9408 ~ 0.0040 | 0.9358 ~ 0.0067 |
|  exp0 HAN_64_512 |  HAN*  |          512 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
| exp0 HAN_64_1024 |  HAN*  |         1024 | 0.9371 ~ 0.0038 | 0.9420 ~ 0.0035 | 0.9380 ~ 0.0045 |
| exp0 HAN_64_2048 |  HAN*  |         2048 | 0.9407 ~ 0.0051 | 0.9452 ~ 0.0051 | 0.9421 ~ 0.0076 |

- **description**
  - vanilla HAN on clean labels
  - exp on effect of batch_size and sample_limit
- **conclusion**
  - batch_size=64, sample=512 works fine

#### exp1 HAN_noise
|             Tag | Method | Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|----------------:|:------:|-----------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |  HAN   |        0.0 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|     exp1 HAN_1p |  HAN   |        0.1 | 0.9295 ~ 0.0036 | 0.9349 ~ 0.0030 | 0.9083 ~ 0.0139 |
|     exp1 HAN_2p |  HAN   |        0.2 | 0.9349 ~ 0.0074 | 0.9400 ~ 0.0065 | 0.9119 ~ 0.0148 |
|     exp1 HAN_3p |  HAN   |        0.3 | 0.9277 ~ 0.0073 | 0.9333 ~ 0.0062 | 0.9054 ~ 0.0136 |
|     exp1 HAN_4p |  HAN   |        0.4 | 0.9224 ~ 0.0108 | 0.9285 ~ 0.0100 | 0.8239 ~ 0.0821 |
|     exp1 HAN_5p |  HAN   |        0.5 |                 |                 |                 |

- **description**
  - vanilla HAN on labels with pair noise
  - exp on its robustness
- **conclusion**
  - something strange is happening
    - as pair noise rate grows, f1 and accuracy drops, but not consistent
    - why 2p > 1p
    - may because the pair noise I choose isn't much confusing

|             Tag | Method | Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|----------------:|:------:|--------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |  HAN   |           0.0 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|     exp1 HAN_u1 |  HAN   |           0.1 |                 |                 |                 |
|     exp1 HAN_u2 |  HAN   |           0.2 |                 |                 |                 |
|     exp1 HAN_u3 |  HAN   |           0.3 |                 |                 |                 |
|     exp1 HAN_u4 |  HAN   |           0.4 |                 |                 |                 |
|     exp1 HAN_u5 |  HAN   |           0.5 |                 |                 |                 |

- **description**
  - vanilla HAN on labels with uniform noise
  - exp on its robustness
- **conclusion**
  - 