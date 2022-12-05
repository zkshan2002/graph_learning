## All Experiments

### Group 1

#### exp0 HAN_batch_sample

|       Tag        | Batch_Size |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|-----------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN_8_100  |          8 | 0.9301 ~ 0.0036 | 0.9356 ~ 0.0035 | 0.9287 ~ 0.0022 |
| exp0 HAN_16_100  |         16 | 0.9323 ~ 0.0043 | 0.9375 ~ 0.0041 | 0.9340 ~ 0.0056 |
| exp0 HAN_32_100  |         32 | 0.9337 ~ 0.0040 | 0.9387 ~ 0.0038 | 0.9354 ~ 0.0084 |
| exp0 HAN_64_100  |         64 | 0.9345 ~ 0.0057 | 0.9394 ~ 0.0056 | 0.9351 ~ 0.0067 |
| exp0 HAN_128_100 |        128 | 0.9327 ~ 0.0035 | 0.9378 ~ 0.0034 | 0.9334 ~ 0.0060 |

|       Tag        | Sample_Limit |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|-------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN_64_64  |           64 | 0.9291 ~ 0.0034 | 0.9342 ~ 0.0035 | 0.9298 ~ 0.0057 |
| exp0 HAN_64_128  |          128 | 0.9319 ~ 0.0041 | 0.9370 ~ 0.0036 | 0.9308 ~ 0.0046 |
| exp0 HAN_64_256  |          256 | 0.9359 ~ 0.0040 | 0.9408 ~ 0.0040 | 0.9358 ~ 0.0067 |
| exp0 HAN_64_512  |          512 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
| exp0 HAN_64_1024 |         1024 | 0.9371 ~ 0.0038 | 0.9420 ~ 0.0035 | 0.9380 ~ 0.0045 |
| exp0 HAN_64_2048 |         2048 | 0.9407 ~ 0.0051 | 0.9452 ~ 0.0051 | 0.9421 ~ 0.0076 |

- **description**
    - vanilla HAN on clean labels
    - exp on effect of batch_size and sample_limit
- **conclusion**
    - batch_size=64, sample=512 works fine

#### exp1 HAN_noise

|       Tag       | Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------:|:----------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |    0.0     | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|   exp1 HAN_p1   |    0.1     | 0.9295 ~ 0.0036 | 0.9349 ~ 0.0030 | 0.9083 ~ 0.0139 |
|   exp1 HAN_p2   |    0.2     | 0.9349 ~ 0.0074 | 0.9400 ~ 0.0065 | 0.9119 ~ 0.0148 |
|   exp1 HAN_p3   |    0.3     | 0.9277 ~ 0.0073 | 0.9333 ~ 0.0062 | 0.9054 ~ 0.0136 |
|   exp1 HAN_p4   |    0.4     | 0.9224 ~ 0.0108 | 0.9285 ~ 0.0100 | 0.8239 ~ 0.0821 |
|   exp1 HAN_p5   |    0.5     | 0.9244 ~ 0.0106 | 0.9301 ~ 0.0096 | 0.5027 ~ 0.0873 |

- **description**
    - vanilla HAN on labels with pair noise
    - exp on its robustness
- **conclusion**
    - as noise rate grows, f1 drops very slowly, while test accuracy drops faster, but still slow
    - possible explanations are:
        - HAN is already quite robust to pair noise
        - the applied pair noise is not so noisy in semantic
        - node cls task on DBLP is easy

|       Tag       | Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------:|:-------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 |      0.0      | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|   exp1 HAN_u1   |      0.1      | 0.9367 ~ 0.0053 | 0.9414 ~ 0.0051 | 0.9302 ~ 0.0087 |
|   exp1 HAN_u2   |      0.2      | 0.9315 ~ 0.0053 | 0.9368 ~ 0.0046 | 0.9244 ~ 0.0117 |
|   exp1 HAN_u3   |      0.3      | 0.9327 ~ 0.0120 | 0.9380 ~ 0.0109 | 0.9077 ~ 0.0275 |
|   exp1 HAN_u4   |      0.4      | 0.9292 ~ 0.0115 | 0.9346 ~ 0.0107 | 0.8730 ~ 0.0312 |
|   exp1 HAN_u5   |      0.5      | 0.9164 ~ 0.0131 | 0.9229 ~ 0.0119 | 0.8127 ~ 0.0535 |

- **description**
    - vanilla HAN on labels with uniform noise
    - exp on its robustness
- **conclusion**
    - similarly, both f1 and accuracy drops slowly

#### exp3 HAN_batch_128 IMDB

|          Tag          | Batch_Size |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------------:|-----------:|:---------------:|:---------------:|:---------------:|
|  exp3 HAN_4_128 IMDB  |          4 | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp3 HAN_8_128 IMDB  |          8 | 0.6099 ~ 0.0075 | 0.6107 ~ 0.0074 | 0.5949 ~ 0.0030 |
| exp3 HAN_16_128 IMDB  |         16 | 0.6084 ~ 0.0080 | 0.6092 ~ 0.0079 | 0.5924 ~ 0.0048 |
| exp3 HAN_32_128 IMDB  |         32 | 0.6063 ~ 0.0062 | 0.6071 ~ 0.0061 | 0.5804 ~ 0.0045 |
| exp3 HAN_64_128 IMDB  |         64 | 0.5945 ~ 0.0056 | 0.5956 ~ 0.0054 | 0.5302 ~ 0.0048 |

- **description**
    - vanilla HAN on clean labels
    - exp on effect of batch_size
    - sample_limit fixed at 128, since IMDB is a small dataset and maximum metapath per node is less than 128
- **conclusion**
    - batch_size=4 works fine

#### exp4 HAN_noise IMDB

|         Tag         | Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------:|:----------:|:---------------:|:---------------:|:---------------:|
| exp3 HAN_4_128 IMDB |    0.0     | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp4 HAN_p1 IMDB   |    0.1     | 0.5967 ~ 0.0067 | 0.5975 ~ 0.0065 | 0.5676 ~ 0.0070 |
|  exp4 HAN_p2 IMDB   |    0.2     | 0.5910 ~ 0.0087 | 0.5907 ~ 0.0087 | 0.5251 ~ 0.0077 |
|  exp4 HAN_p3 IMDB   |    0.3     | 0.5648 ~ 0.0084 | 0.5642 ~ 0.0083 | 0.4596 ~ 0.0081 |
|  exp4 HAN_p4 IMDB   |    0.4     | 0.5442 ~ 0.0211 | 0.5444 ~ 0.0197 | 0.3953 ~ 0.0091 |
|  exp4 HAN_p5 IMDB   |    0.5     | 0.4940 ~ 0.0174 | 0.4983 ~ 0.0160 | 0.3541 ~ 0.0103 |

|         Tag         | Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------:|:-------------:|:---------------:|:---------------:|:---------------:|
| exp3 HAN_4_128 IMDB |      0.0      | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp4 HAN_u1 IMDB   |      0.1      | 0.5886 ~ 0.0087 | 0.5900 ~ 0.0083 | 0.5712 ~ 0.0048 |
|  exp4 HAN_u2 IMDB   |      0.2      | 0.5584 ~ 0.0101 | 0.5601 ~ 0.0096 | 0.5200 ~ 0.0073 |
|  exp4 HAN_u3 IMDB   |      0.3      | 0.5459 ~ 0.0112 | 0.5481 ~ 0.0107 | 0.4901 ~ 0.0074 |
|  exp4 HAN_u4 IMDB   |      0.4      | 0.5378 ~ 0.0135 | 0.5398 ~ 0.0133 | 0.4936 ~ 0.0060 |
|  exp4 HAN_u5 IMDB   |      0.5      | 0.5031 ~ 0.0136 | 0.5096 ~ 0.0129 | 0.4220 ~ 0.0098 |

- **description**
    - vanilla HAN on labels with pair/uniform noise
    - exp on its robustness
- **conclusion**
    - f1 and acc drops much faster as noise ratio grows, compared to DBLP

#### exp5 HAN_p4 SFT_F_memory_warmup IMDB

|            Tag             |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------:|:---------------:|:---------------:|:---------------:|
|      exp4 HAN_p4 IMDB      | 0.5442 ~ 0.0211 | 0.5444 ~ 0.0197 | 0.3953 ~ 0.0091 |
| exp5 HAN_p4 SFT_F_1_2 IMDB | 0.5586 ~ 0.0099 | 0.5578 ~ 0.0099 | 0.3840 ~ 0.0056 |
| exp5 HAN_p4 SFT_F_1_4 IMDB | 0.5578 ~ 0.0101 | 0.5570 ~ 0.0102 | 0.3833 ~ 0.0055 |
| exp5 HAN_p4 SFT_F_1_8 IMDB | 0.5576 ~ 0.0103 | 0.5567 ~ 0.0103 | 0.3838 ~ 0.0066 |
| exp5 HAN_p4 SFT_F_2_2 IMDB | 0.5601 ~ 0.0097 | 0.5592 ~ 0.0096 | 0.3870 ~ 0.0069 |
| exp5 HAN_p4 SFT_F_2_4 IMDB | 0.5584 ~ 0.0103 | 0.5575 ~ 0.0105 | 0.3829 ~ 0.0037 |
| exp5 HAN_p4 SFT_F_2_8 IMDB | 0.5593 ~ 0.0096 | 0.5586 ~ 0.0097 | 0.3855 ~ 0.0072 |
| exp5 HAN_p4 SFT_F_3_2 IMDB | 0.5587 ~ 0.0124 | 0.5581 ~ 0.0121 | 0.3822 ~ 0.0084 |
| exp5 HAN_p4 SFT_F_4_2 IMDB | 0.5558 ~ 0.0103 | 0.5553 ~ 0.0103 | 0.3842 ~ 0.0057 |

- **description**
    - [SFT] proposes self filtering + adaptive loss + fixmatch
    - apply self filtering only and exp on its effect
- **conclusion**
    - only minor f1 improvement, and difference hyper params do not matter
    - explanation
        - fluctuating ratio is small and do not reflect noisy node very well
            - e.g., pair noise=0.4, yet fluctuating ratio<0.15
        - lack learning from fluctuating nodes

#### exp5 HAN_p4 SFT_F_1_2_threshold_weight0_weight1 IMDB

|                 Tag                 |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-----------------------------------:|:---------------:|:---------------:|:---------------:|
|          exp4 HAN_p4 IMDB           | 0.5442 ~ 0.0211 | 0.5444 ~ 0.0197 | 0.3953 ~ 0.0091 |
|     exp5 HAN_p4 SFT_F_1_2 IMDB      | 0.5586 ~ 0.0099 | 0.5578 ~ 0.0099 | 0.3840 ~ 0.0056 |
|  exp5 HAN_p4 SFT_F_1_2_020501 IMDB  | 0.5585 ~ 0.0089 | 0.5577 ~ 0.0089 | 0.3853 ~ 0.0056 |
|  exp5 HAN_p4 SFT_F_1_2_050501 IMDB  | 0.5571 ~ 0.0103 | 0.5565 ~ 0.0104 | 0.3827 ~ 0.0051 |
| exp5 HAN_p4 SFT_F_1_2_0205005 IMDB  | 0.5586 ~ 0.0100 | 0.5579 ~ 0.0100 | 0.3840 ~ 0.0062 |
| exp5 HAN_p4 SFT_F_1_2_0505005 IMDB  | 0.5601 ~ 0.0107 | 0.5596 ~ 0.0109 | 0.3851 ~ 0.0057 |
| exp5 HAN_p4 SFT_F_1_2_06705005 IMDB | 0.5569 ~ 0.0100 | 0.5562 ~ 0.0101 | 0.3845 ~ 0.0037 |
| exp5 HAN_p4 SFT_F_1_2_07505005 IMDB | 0.5550 ~ 0.0105 | 0.5545 ~ 0.0103 | 0.3860 ~ 0.0035 |
| exp5 HAN_p4 SFT_F_1_2_1005005 IMDB  | 0.5543 ~ 0.0092 | 0.5537 ~ 0.0089 | 0.3881 ~ 0.0044 |

- **description**
    - [SFT] proposes self filtering + adaptive loss + fixmatch
    - apply self filtering and adaptive loss and exp on its effect
- **conclusion**
    - adding adaptive loss brings very limited improvement
    - explanation
        - adaptive loss penalizes overconfident prediction, which doesn't happen much
        - in DBLP, when overconfident prediction happens more often, this loss brings very unstable behavior and damages
          gradients, probably due to its division computation