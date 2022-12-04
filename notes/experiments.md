## All Experiments
### Group 1
#### exp0 HAN_batch_sample

|              Tag | Batch_Size |    Macro-F1     |    Micro-F1     |    Accuracy     |
|-----------------:|-----------:|:---------------:|:---------------:|:---------------:|
|   exp0 HAN_8_100 |          8 | 0.9301 ~ 0.0036 | 0.9356 ~ 0.0035 | 0.9287 ~ 0.0022 |
|  exp0 HAN_16_100 |         16 | 0.9323 ~ 0.0043 | 0.9375 ~ 0.0041 | 0.9340 ~ 0.0056 |
|  exp0 HAN_32_100 |         32 | 0.9337 ~ 0.0040 | 0.9387 ~ 0.0038 | 0.9354 ~ 0.0084 |
|  exp0 HAN_64_100 |         64 | 0.9345 ~ 0.0057 | 0.9394 ~ 0.0056 | 0.9351 ~ 0.0067 |
| exp0 HAN_128_100 |        128 | 0.9327 ~ 0.0035 | 0.9378 ~ 0.0034 | 0.9334 ~ 0.0060 |

|              Tag | Sample_Limit |    Macro-F1     |    Micro-F1     |    Accuracy     |
|-----------------:|-------------:|:---------------:|:---------------:|:---------------:|
|   exp0 HAN_64_64 |           64 | 0.9291 ~ 0.0034 | 0.9342 ~ 0.0035 | 0.9298 ~ 0.0057 |
|  exp0 HAN_64_128 |          128 | 0.9319 ~ 0.0041 | 0.9370 ~ 0.0036 | 0.9308 ~ 0.0046 |
|  exp0 HAN_64_256 |          256 | 0.9359 ~ 0.0040 | 0.9408 ~ 0.0040 | 0.9358 ~ 0.0067 |
|  exp0 HAN_64_512 |          512 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
| exp0 HAN_64_1024 |         1024 | 0.9371 ~ 0.0038 | 0.9420 ~ 0.0035 | 0.9380 ~ 0.0045 |
| exp0 HAN_64_2048 |         2048 | 0.9407 ~ 0.0051 | 0.9452 ~ 0.0051 | 0.9421 ~ 0.0076 |

- **description**
  - vanilla HAN on clean labels
  - exp on effect of batch_size and sample_limit
- **conclusion**
  - batch_size=64, sample=512 works fine

#### exp1 HAN_noise
|             Tag | Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|----------------:|-----------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |        0.0 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|     exp1 HAN_p1 |        0.1 | 0.9295 ~ 0.0036 | 0.9349 ~ 0.0030 | 0.9083 ~ 0.0139 |
|     exp1 HAN_p2 |        0.2 | 0.9349 ~ 0.0074 | 0.9400 ~ 0.0065 | 0.9119 ~ 0.0148 |
|     exp1 HAN_p3 |        0.3 | 0.9277 ~ 0.0073 | 0.9333 ~ 0.0062 | 0.9054 ~ 0.0136 |
|     exp1 HAN_p4 |        0.4 | 0.9224 ~ 0.0108 | 0.9285 ~ 0.0100 | 0.8239 ~ 0.0821 |
|     exp1 HAN_p5 |        0.5 | 0.9244 ~ 0.0106 | 0.9301 ~ 0.0096 | 0.5027 ~ 0.0873 |

- **description**
  - vanilla HAN on labels with pair noise
  - exp on its robustness
- **conclusion**
  - something strange is happening
    - as pair noise rate grows, f1 and accuracy drops, but not consistent
    - why 2p > 1p
    - may because the pair noise I choose isn't much confusing
  - vanilla HAN is quite robust to pair noise
    - f1 drops by a small margin at large noise rate(~2% at 40%)

|             Tag | Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|----------------:|--------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |           0.0 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|     exp1 HAN_u1 |           0.1 | 0.9367 ~ 0.0053 | 0.9414 ~ 0.0051 | 0.9302 ~ 0.0087 |
|     exp1 HAN_u2 |           0.2 | 0.9315 ~ 0.0053 | 0.9368 ~ 0.0046 | 0.9244 ~ 0.0117 |
|     exp1 HAN_u3 |           0.3 | 0.9327 ~ 0.0120 | 0.9380 ~ 0.0109 | 0.9077 ~ 0.0275 |
|     exp1 HAN_u4 |           0.4 | 0.9292 ~ 0.0115 | 0.9346 ~ 0.0107 | 0.8730 ~ 0.0312 |
|     exp1 HAN_u5 |           0.5 | 0.9164 ~ 0.0131 | 0.9229 ~ 0.0119 | 0.8127 ~ 0.0535 |

- **description**
  - vanilla HAN on labels with uniform noise
  - exp on its robustness
- **conclusion**
  - 

#### exp3 HAN_batch_sample on IMDB

|                  Tag | Batch_Size |    Macro-F1     |    Micro-F1     |    Accuracy     |
|---------------------:|-----------:|:---------------:|:---------------:|:---------------:|
|  exp3 HAN_4_128 IMDB |          4 | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp3 HAN_8_128 IMDB |          8 | 0.6099 ~ 0.0075 | 0.6107 ~ 0.0074 | 0.5949 ~ 0.0030 |
| exp3 HAN_16_128 IMDB |         16 | 0.6084 ~ 0.0080 | 0.6092 ~ 0.0079 | 0.5924 ~ 0.0048 |
| exp3 HAN_32_128 IMDB |         32 | 0.6063 ~ 0.0062 | 0.6071 ~ 0.0061 | 0.5804 ~ 0.0045 |
| exp3 HAN_64_128 IMDB |         64 | 0.5945 ~ 0.0056 | 0.5956 ~ 0.0054 | 0.5302 ~ 0.0048 |