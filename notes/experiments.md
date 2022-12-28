## All Experiments

### Group 1

#### exp0 HAN_batch_sample

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN_8_100  | 0.9301 ~ 0.0036 | 0.9356 ~ 0.0035 | 0.9287 ~ 0.0022 |
| exp0 HAN_16_100  | 0.9323 ~ 0.0043 | 0.9375 ~ 0.0041 | 0.9340 ~ 0.0056 |
| exp0 HAN_32_100  | 0.9337 ~ 0.0040 | 0.9387 ~ 0.0038 | 0.9354 ~ 0.0084 |
| exp0 HAN_64_100  | 0.9345 ~ 0.0057 | 0.9394 ~ 0.0056 | 0.9351 ~ 0.0067 |
| exp0 HAN_128_100 | 0.9327 ~ 0.0035 | 0.9378 ~ 0.0034 | 0.9334 ~ 0.0060 |

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN_64_64  | 0.9291 ~ 0.0034 | 0.9342 ~ 0.0035 | 0.9298 ~ 0.0057 |
| exp0 HAN_64_128  | 0.9319 ~ 0.0041 | 0.9370 ~ 0.0036 | 0.9308 ~ 0.0046 |
| exp0 HAN_64_256  | 0.9359 ~ 0.0040 | 0.9408 ~ 0.0040 | 0.9358 ~ 0.0067 |
| exp0 HAN_64_512  | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
| exp0 HAN_64_1024 | 0.9371 ~ 0.0038 | 0.9420 ~ 0.0035 | 0.9380 ~ 0.0045 |
| exp0 HAN_64_2048 | 0.9407 ~ 0.0051 | 0.9452 ~ 0.0051 | 0.9421 ~ 0.0076 |

- **description**
    - vanilla HAN on clean labels
    - exp on effect of batch_size and sample_limit
- **conclusion**
    - batch_size=64, sample=512 works fine

#### exp1 HAN_noise

|       Tag       |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|   exp1 HAN_p1   | 0.9295 ~ 0.0036 | 0.9349 ~ 0.0030 | 0.9083 ~ 0.0139 |
|   exp1 HAN_p2   | 0.9349 ~ 0.0074 | 0.9400 ~ 0.0065 | 0.9119 ~ 0.0148 |
|   exp1 HAN_p3   | 0.9277 ~ 0.0073 | 0.9333 ~ 0.0062 | 0.9054 ~ 0.0136 |
|   exp1 HAN_p4   | 0.9224 ~ 0.0108 | 0.9285 ~ 0.0100 | 0.8239 ~ 0.0821 |
|   exp1 HAN_p5   | 0.9244 ~ 0.0106 | 0.9301 ~ 0.0096 | 0.5027 ~ 0.0873 |

- **description**
    - vanilla HAN on labels with pair noise
    - exp on its robustness
- **conclusion**
    - as noise rate grows, f1 drops very slowly, while test accuracy drops faster, but still slow
    - possible explanations are:
        - HAN is already quite robust to pair noise
        - the applied pair noise is not so noisy in semantic
        - node cls task on DBLP is easy

|       Tag       |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN_64_512 | 0.9389 ~ 0.0041 | 0.9434 ~ 0.0035 | 0.9391 ~ 0.0047 |
|   exp1 HAN_u1   | 0.9367 ~ 0.0053 | 0.9414 ~ 0.0051 | 0.9302 ~ 0.0087 |
|   exp1 HAN_u2   | 0.9315 ~ 0.0053 | 0.9368 ~ 0.0046 | 0.9244 ~ 0.0117 |
|   exp1 HAN_u3   | 0.9327 ~ 0.0120 | 0.9380 ~ 0.0109 | 0.9077 ~ 0.0275 |
|   exp1 HAN_u4   | 0.9292 ~ 0.0115 | 0.9346 ~ 0.0107 | 0.8730 ~ 0.0312 |
|   exp1 HAN_u5   | 0.9164 ~ 0.0131 | 0.9229 ~ 0.0119 | 0.8127 ~ 0.0535 |

- **description**
    - vanilla HAN on labels with uniform noise
    - exp on its robustness
- **conclusion**
    - similarly, both f1 and accuracy drops slowly

#### exp3 HAN_batch_128 IMDB

|          Tag          |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------------:|:---------------:|:---------------:|:---------------:|
|  exp3 HAN_4_128 IMDB  | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp3 HAN_8_128 IMDB  | 0.6099 ~ 0.0075 | 0.6107 ~ 0.0074 | 0.5949 ~ 0.0030 |
| exp3 HAN_16_128 IMDB  | 0.6084 ~ 0.0080 | 0.6092 ~ 0.0079 | 0.5924 ~ 0.0048 |
| exp3 HAN_32_128 IMDB  | 0.6063 ~ 0.0062 | 0.6071 ~ 0.0061 | 0.5804 ~ 0.0045 |
| exp3 HAN_64_128 IMDB  | 0.5945 ~ 0.0056 | 0.5956 ~ 0.0054 | 0.5302 ~ 0.0048 |

- **description**
    - vanilla HAN on clean labels
    - exp on effect of batch_size
    - sample_limit fixed at 128, since IMDB is a small dataset and maximum metapath per node is less than 128
- **conclusion**
    - batch_size=4 works fine

#### exp4 HAN_noise IMDB

|         Tag         |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------:|:---------------:|:---------------:|:---------------:|
| exp3 HAN_4_128 IMDB | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp4 HAN_p1 IMDB   | 0.5967 ~ 0.0067 | 0.5975 ~ 0.0065 | 0.5676 ~ 0.0070 |
|  exp4 HAN_p2 IMDB   | 0.5910 ~ 0.0087 | 0.5907 ~ 0.0087 | 0.5251 ~ 0.0077 |
|  exp4 HAN_p3 IMDB   | 0.5648 ~ 0.0084 | 0.5642 ~ 0.0083 | 0.4596 ~ 0.0081 |
|  exp4 HAN_p4 IMDB   | 0.5442 ~ 0.0211 | 0.5444 ~ 0.0197 | 0.3953 ~ 0.0091 |
|  exp4 HAN_p5 IMDB   | 0.4940 ~ 0.0174 | 0.4983 ~ 0.0160 | 0.3541 ~ 0.0103 |

|         Tag         |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------:|:---------------:|:---------------:|:---------------:|
| exp3 HAN_4_128 IMDB | 0.6092 ~ 0.0082 | 0.6102 ~ 0.0077 | 0.5945 ~ 0.0037 |
|  exp4 HAN_u1 IMDB   | 0.5886 ~ 0.0087 | 0.5900 ~ 0.0083 | 0.5712 ~ 0.0048 |
|  exp4 HAN_u2 IMDB   | 0.5584 ~ 0.0101 | 0.5601 ~ 0.0096 | 0.5200 ~ 0.0073 |
|  exp4 HAN_u3 IMDB   | 0.5459 ~ 0.0112 | 0.5481 ~ 0.0107 | 0.4901 ~ 0.0074 |
|  exp4 HAN_u4 IMDB   | 0.5378 ~ 0.0135 | 0.5398 ~ 0.0133 | 0.4936 ~ 0.0060 |
|  exp4 HAN_u5 IMDB   | 0.5031 ~ 0.0136 | 0.5096 ~ 0.0129 | 0.4220 ~ 0.0098 |

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
| exp5 HAN_p4 SFT_F_1_2_05050025 IMDB | 0.5580 ~ 0.0091 | 0.5571 ~ 0.0091 | 0.3850 ~ 0.0057 | 
| exp5 HAN_p4 SFT_F_1_2_05050075 IMDB | 0.5582 ~ 0.0108 | 0.5577 ~ 0.0109 | 0.3843 ~ 0.0028 |

- **description**
    - [SFT] proposes self filtering + adaptive loss + fixmatch
    - apply self filtering and adaptive loss and exp on its effect
- **conclusion**
    - adding adaptive loss brings almost no improvement
    - explanation
        - adaptive loss penalizes overconfident prediction, which doesn't happen much
        - in DBLP, when overconfident prediction happens more often, this loss brings very unstable behavior and damages
          gradients, probably due to its division computation

#### exp5_1 HAN_p2 SFT_threshold_weight0_weight1 IMDB

|               Tag                |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------------:|:---------------:|:---------------:|:---------------:|
|         exp4 HAN_p2 IMDB         | 0.5910 ~ 0.0087 | 0.5907 ~ 0.0087 | 0.5251 ~ 0.0077 |
| exp5_1 HAN_p2 SFT_02_05_01 IMDB  | 0.5924 ~ 0.0060 | 0.5926 ~ 0.0058 | 0.5244 ~ 0.0064 |
| exp5_1 HAN_p2 SFT_02_05_005 IMDB | 0.5929 ~ 0.0058 | 0.5930 ~ 0.0055 | 0.5255 ~ 0.0065 |
| exp5_1 HAN_p2 SFT_05_05_01 IMDB  | 0.5885 ~ 0.0065 | 0.5886 ~ 0.0063 | 0.5197 ~ 0.0059 |
| exp5_1 HAN_p2 SFT_05_05_005 IMDB | 0.5909 ~ 0.0059 | 0.5911 ~ 0.0057 | 0.5241 ~ 0.0066 |

#### exp5_2 HAN_u2 SFT_threshold_weight0_weight1 IMDB

|               Tag                |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------------:|:---------------:|:---------------:|:---------------:|
|         exp4 HAN_u2 IMDB         | 0.5584 ~ 0.0101 | 0.5601 ~ 0.0096 | 0.5200 ~ 0.0073 |
| exp5_2 HAN_u2 SFT_02_05_01 IMDB  | 0.5858 ~ 0.0077 | 0.5889 ~ 0.0070 | 0.5585 ~ 0.0014 |
| exp5_2 HAN_u2 SFT_02_05_005 IMDB | 0.5865 ~ 0.0084 | 0.5894 ~ 0.0077 | 0.5585 ~ 0.0014 |
| exp5_2 HAN_u2 SFT_05_05_01 IMDB  | 0.5828 ~ 0.0067 | 0.5862 ~ 0.0062 | 0.5529 ~ 0.0051 |
| exp5_2 HAN_u2 SFT_05_05_005 IMDB | 0.5855 ~ 0.0082 | 0.5887 ~ 0.0074 | 0.5575 ~ 0.0021 |

#### exp5_3 HAN_u4 SFT_threshold_weight0_weight1 IMDB

|               Tag                |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------------:|:---------------:|:---------------:|:---------------:|
|         exp4 HAN_u4 IMDB         | 0.5378 ~ 0.0135 | 0.5398 ~ 0.0133 | 0.4936 ~ 0.0060 |
| exp5_3 HAN_u4 SFT_02_05_01 IMDB  | 0.5328 ~ 0.0087 | 0.5355 ~ 0.0084 | 0.4806 ~ 0.0041 |
| exp5_3 HAN_u4 SFT_02_05_005 IMDB | 0.5327 ~ 0.0090 | 0.5354 ~ 0.0087 | 0.4808 ~ 0.0042 |
| exp5_3 HAN_u4 SFT_05_05_01 IMDB  | 0.5307 ~ 0.0096 | 0.5335 ~ 0.0094 | 0.4804 ~ 0.0058 |
| exp5_3 HAN_u4 SFT_05_05_005 IMDB | 0.5321 ~ 0.0087 | 0.5348 ~ 0.0083 | 0.4791 ~ 0.0041 |

### Group 2

- description
  - fixes a bug in previous implementation of HAN:
    - activation function after node-level attention is missing
    - also, is the init of batched weights wrong?
- known bugs
  - in HAN, setting type_aware_semantic to True in single control type dataset improves performance

#### exp0 HAN dataset

- description
  - baseline performance on each dataset.
- notes:
  - hyper params tuned.
- conclusion:
  - pass

|      Tag      |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------:|:---------------:|:---------------:|:---------------:|
| exp0 HAN DBLP | 0.9393 ~ 0.0043 | 0.9430 ~ 0.0041 | 0.9382 ~ 0.0039 |
| exp0 HAN IMDB | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |

#### exp1 HAN dataset noise

- description
  - apply noise(pair or uniform) and observe HAN's robustness
- notes:
  - pair noise specified:
    - DBLP: 0 Database, 1 Data Mining, 2 AI, 3 Information Retrieval
      - [1, 0, 3, 2]
    - IMDB: 0 Action, 1 Comedy, 2 Drama
      - [1, 2, 0]
- conclusion:
  - both noises hurt HAN's performance, test time accuracy suffers more than f1 score
    - cls head is susceptible to noisy label, but learnt embeddings are less influenced
  - uniform noise hurts more than pair noise
    - pair noise may be less noisy in semantics
  - performance on DBLP is more robust than IMDB

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN DBLP   | 0.9393 ~ 0.0043 | 0.9430 ~ 0.0041 | 0.9382 ~ 0.0039 |
| exp1 HAN DBLP p1 | 0.9359 ~ 0.0023 | 0.9403 ~ 0.0022 | 0.9320 ~ 0.0026 |
| exp1 HAN DBLP p2 | 0.9355 ~ 0.0041 | 0.9398 ~ 0.0037 | 0.9179 ~ 0.0141 |
| exp1 HAN DBLP p3 | 0.9345 ~ 0.0045 | 0.9388 ~ 0.0039 | 0.9033 ~ 0.0152 |
| exp1 HAN DBLP p4 | 0.9356 ~ 0.0044 | 0.9399 ~ 0.0041 | 0.8868 ~ 0.0145 |
| exp1 HAN DBLP p5 | 0.9314 ~ 0.0069 | 0.9360 ~ 0.0063 | 0.5770 ~ 0.0780 |

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN DBLP   | 0.9393 ~ 0.0043 | 0.9430 ~ 0.0041 | 0.9382 ~ 0.0039 |
| exp1 HAN DBLP u1 | 0.9362 ~ 0.0035 | 0.9403 ~ 0.0034 | 0.9331 ~ 0.0069 |
| exp1 HAN DBLP u2 | 0.9351 ~ 0.0056 | 0.9393 ~ 0.0052 | 0.9188 ~ 0.0139 |
| exp1 HAN DBLP u3 | 0.9361 ~ 0.0076 | 0.9403 ~ 0.0072 | 0.9153 ~ 0.0075 |
| exp1 HAN DBLP u4 | 0.9165 ~ 0.0506 | 0.9217 ~ 0.0481 | 0.8747 ~ 0.0891 |
| exp1 HAN DBLP u5 | 0.8873 ~ 0.0529 | 0.8937 ~ 0.0516 | 0.7942 ~ 0.0894 |

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN IMDB   | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
| exp1 HAN IMDB p1 | 0.5990 ~ 0.0092 | 0.6006 ~ 0.0091 | 0.5779 ~ 0.0053 |
| exp1 HAN IMDB p2 | 0.5874 ~ 0.0102 | 0.5897 ~ 0.0102 | 0.5419 ~ 0.0064 |
| exp1 HAN IMDB p3 | 0.5719 ~ 0.0112 | 0.5756 ~ 0.0112 | 0.4939 ~ 0.0077 |
| exp1 HAN IMDB p4 | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
| exp1 HAN IMDB p5 | 0.5623 ~ 0.0105 | 0.5673 ~ 0.0102 | 0.4186 ~ 0.0072 |

|       Tag        |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------:|:---------------:|:---------------:|:---------------:|
|  exp0 HAN IMDB   | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
| exp1 HAN IMDB u1 | 0.5978 ~ 0.0078 | 0.6001 ~ 0.0077 | 0.5812 ~ 0.0045 |
| exp1 HAN IMDB u2 | 0.5779 ~ 0.0092 | 0.5805 ~ 0.0097 | 0.5473 ~ 0.0035 |
| exp1 HAN IMDB u3 | 0.5539 ~ 0.0100 | 0.5575 ~ 0.0103 | 0.5045 ~ 0.0054 |
| exp1 HAN IMDB u4 | 0.5266 ~ 0.0120 | 0.5285 ~ 0.0121 | 0.4719 ~ 0.0056 |
| exp1 HAN IMDB u5 | 0.5087 ~ 0.0102 | 0.5129 ~ 0.0110 | 0.4282 ~ 0.0070 |

#### exp2 HAN dataset noise SFT_memory_warmup(_threshold_weight0_weight1)

- description
  - apply [SFT] filtering and loss and observe its effect
- notes:
  - [SFT] hyper params fixed
    - F: 1_2
    - L: 05_05_005
- conclusion:
  - almost no improvement.
  - perhaps hyperparam isn't tweaked well

|                Tag                 |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------------------------:|:---------------:|:---------------:|:---------------:|
|          exp1 HAN IMDB p2          | 0.5874 ~ 0.0102 | 0.5897 ~ 0.0102 | 0.5419 ~ 0.0064 |
|      exp2 HAN IMDB p2 SFT_1_2      | 0.5882 ~ 0.0093 | 0.5906 ~ 0.0093 | 0.5438 ~ 0.0061 |
| exp2 HAN IMDB p2 SFT_1_2_05_05_005 | 0.5875 ~ 0.0105 | 0.5902 ~ 0.0106 | 0.5428 ~ 0.0057 |
|          exp1 HAN IMDB p4          | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
|      exp2 HAN IMDB p4 SFT_1_2      | 0.5616 ~ 0.0120 | 0.5646 ~ 0.0113 | 0.4526 ~ 0.0092 |
| exp2 HAN IMDB p4 SFT_1_2_05_05_005 | 0.5614 ~ 0.0123 | 0.5645 ~ 0.0114 | 0.4518 ~ 0.0073 |
|          exp1 HAN IMDB u2          | 0.5779 ~ 0.0092 | 0.5805 ~ 0.0097 | 0.5473 ~ 0.0035 |
|      exp2 HAN IMDB u2 SFT_1_2      | 0.5793 ~ 0.0094 | 0.5821 ~ 0.0097 | 0.5509 ~ 0.0035 |
| exp2 HAN IMDB u2 SFT_1_2_05_05_005 | 0.5790 ~ 0.0112 | 0.5813 ~ 0.0115 | 0.5454 ~ 0.0047 |
|          exp1 HAN IMDB u4          | 0.5266 ~ 0.0120 | 0.5285 ~ 0.0121 | 0.4719 ~ 0.0056 |
|      exp2 HAN IMDB u4 SFT_1_2      | 0.5270 ~ 0.0102 | 0.5290 ~ 0.0101 | 0.4755 ~ 0.0048 |
| exp2 HAN IMDB u4 SFT_1_2_05_05_005 | 0.5271 ~ 0.0099 | 0.5290 ~ 0.0097 | 0.4752 ~ 0.0046 |

- description
  - hyperparam selection
- notes:
  - filtering only
- conclusion:
  - SFT_1_4

|            Tag            |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------------:|:---------------:|:---------------:|:---------------:|
|     exp1 HAN IMDB p4      | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
| exp2 HAN IMDB p4 SFT_1_2  | 0.5616 ~ 0.0120 | 0.5646 ~ 0.0113 | 0.4526 ~ 0.0092 |
| exp2 HAN IMDB p4 SFT_1_4  | 0.5681 ~ 0.0060 | 0.5705 ~ 0.0059 | 0.4555 ~ 0.0082 |
| exp2 HAN IMDB p4 SFT_1_8  | 0.5668 ~ 0.0066 | 0.5694 ~ 0.0065 | 0.4536 ~ 0.0083 |
| exp2 HAN IMDB p4 SFT_1_16 | 0.5650 ~ 0.0080 | 0.5681 ~ 0.0070 | 0.4561 ~ 0.0098 |
| exp2 HAN IMDB p4 SFT_2_2  | 0.5672 ~ 0.0057 | 0.5705 ~ 0.0055 | 0.4454 ~ 0.0145 |
| exp2 HAN IMDB p4 SFT_2_4  | 0.5683 ~ 0.0050 | 0.5705 ~ 0.0051 | 0.4543 ~ 0.0038 |
| exp2 HAN IMDB p4 SFT_2_8  | 0.5674 ~ 0.0065 | 0.5698 ~ 0.0063 | 0.4508 ~ 0.0073 |
| exp2 HAN IMDB p4 SFT_2_16 | 0.5650 ~ 0.0080 | 0.5681 ~ 0.0070 | 0.4561 ~ 0.0098 |

- description
  - apply [SFT] filtering and observe its effect
- notes:
  - hyperparam fixed at 1_2
- conclusion:
  - almost no improvement.
  - perhaps hyper param isn't tweaked

|           Tag            |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:------------------------:|:---------------:|:---------------:|:---------------:|
|      exp0 HAN IMDB       | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
|     exp1 HAN IMDB p1     | 0.5990 ~ 0.0092 | 0.6006 ~ 0.0091 | 0.5779 ~ 0.0053 |
|     exp1 HAN IMDB p2     | 0.5874 ~ 0.0102 | 0.5897 ~ 0.0102 | 0.5419 ~ 0.0064 |
|     exp1 HAN IMDB p3     | 0.5719 ~ 0.0112 | 0.5756 ~ 0.0112 | 0.4939 ~ 0.0077 |
|     exp1 HAN IMDB p4     | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
|     exp1 HAN IMDB p5     | 0.5623 ~ 0.0105 | 0.5673 ~ 0.0102 | 0.4186 ~ 0.0072 |
|  exp2 HAN IMDB SFT_1_4   | 0.6172 ~ 0.0051 | 0.6192 ~ 0.0051 | 0.6015 ~ 0.0035 |
| exp2 HAN IMDB p1 SFT_1_4 | 0.6043 ~ 0.0068 | 0.6064 ~ 0.0070 | 0.5809 ~ 0.0089 |
| exp2 HAN IMDB p2 SFT_1_4 | 0.5909 ~ 0.0103 | 0.5941 ~ 0.0101 | 0.5457 ~ 0.0048 |
| exp2 HAN IMDB p3 SFT_1_4 | 0.5782 ~ 0.0095 | 0.5827 ~ 0.0092 | 0.5015 ~ 0.0083 |
| exp2 HAN IMDB p4 SFT_1_4 | 0.5670 ~ 0.0057 | 0.5711 ~ 0.0054 | 0.4562 ~ 0.0089 |
| exp2 HAN IMDB p5 SFT_1_4 | 0.5680 ~ 0.0090 | 0.5739 ~ 0.0084 | 0.4238 ~ 0.0047 |

|           Tag            |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:------------------------:|:---------------:|:---------------:|:---------------:|
|      exp0 HAN IMDB       | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
|     exp1 HAN IMDB u1     | 0.5978 ~ 0.0078 | 0.6001 ~ 0.0077 | 0.5812 ~ 0.0045 |
|     exp1 HAN IMDB u2     | 0.5779 ~ 0.0092 | 0.5805 ~ 0.0097 | 0.5473 ~ 0.0035 |
|     exp1 HAN IMDB u3     | 0.5539 ~ 0.0100 | 0.5575 ~ 0.0103 | 0.5045 ~ 0.0054 |
|     exp1 HAN IMDB u4     | 0.5266 ~ 0.0120 | 0.5285 ~ 0.0121 | 0.4719 ~ 0.0056 |
|     exp1 HAN IMDB u5     | 0.5087 ~ 0.0102 | 0.5129 ~ 0.0110 | 0.4282 ~ 0.0070 |
| exp2 HAN IMDB u1 SFT_1_4 | 0.6020 ~ 0.0073 | 0.6043 ~ 0.0077 | 0.5870 ~ 0.0071 |
| exp2 HAN IMDB u2 SFT_1_4 | 0.5842 ~ 0.0082 | 0.5874 ~ 0.0088 | 0.5545 ~ 0.0038 |
| exp2 HAN IMDB u3 SFT_1_4 | 0.5580 ~ 0.0101 | 0.5625 ~ 0.0102 | 0.5070 ~ 0.0057 |
| exp2 HAN IMDB u4 SFT_1_4 | 0.5336 ~ 0.0060 | 0.5367 ~ 0.0070 | 0.4770 ~ 0.0060 |
| exp2 HAN IMDB u5 SFT_1_4 | 0.5089 ~ 0.0100 | 0.5147 ~ 0.0101 | 0.4344 ~ 0.0046 |

- description
  - hyperparam selection
- notes:
  - loss on SFT_1_4
- conclusion:
  - SFT_1_4_04_05_01

|                Tag                 |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------------------------------:|:---------------:|:---------------:|:---------------:|
|          exp1 HAN IMDB p4          | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
|      exp2 HAN IMDB p4 SFT_1_4      | 0.5670 ~ 0.0057 | 0.5711 ~ 0.0054 | 0.4562 ~ 0.0089 |
| exp2 HAN IMDB p4 SFT_1_4_02_05_01  | 0.5674 ~ 0.0058 | 0.5712 ~ 0.0060 | 0.4585 ~ 0.0090 |
| exp2 HAN IMDB p4 SFT_1_4_04_05_01  | 0.5676 ~ 0.0057 | 0.5715 ~ 0.0057 | 0.4579 ~ 0.0091 |
| exp2 HAN IMDB p4 SFT_1_4_06_05_01  | 0.5652 ~ 0.0059 | 0.5696 ~ 0.0068 | 0.4542 ~ 0.0082 |
| exp2 HAN IMDB p4 SFT_1_4_02_05_005 | 0.5672 ~ 0.0058 | 0.5713 ~ 0.0055 | 0.4559 ~ 0.0097 |
| exp2 HAN IMDB p4 SFT_1_4_04_05_005 | 0.5671 ~ 0.0054 | 0.5711 ~ 0.0055 | 0.4578 ~ 0.0086 |
| exp2 HAN IMDB p4 SFT_1_4_06_05_005 | 0.5655 ~ 0.0055 | 0.5696 ~ 0.0060 | 0.4590 ~ 0.0072 |

- description
  - apply [SFT] filtering and loss, and observe its effect
- notes:
  - loss on SFT_1_4_04_05_01
- conclusion:
  - pass

|                Tag                |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------------------------:|:---------------:|:---------------:|:---------------:|
|       exp2 HAN IMDB SFT_1_4       | 0.6172 ~ 0.0051 | 0.6192 ~ 0.0051 | 0.6015 ~ 0.0035 |
|     exp2 HAN IMDB p1 SFT_1_4      | 0.6043 ~ 0.0068 | 0.6064 ~ 0.0070 | 0.5809 ~ 0.0089 |
|     exp2 HAN IMDB p2 SFT_1_4      | 0.5909 ~ 0.0103 | 0.5941 ~ 0.0101 | 0.5457 ~ 0.0048 |
|     exp2 HAN IMDB p3 SFT_1_4      | 0.5782 ~ 0.0095 | 0.5827 ~ 0.0092 | 0.5015 ~ 0.0083 |
|     exp2 HAN IMDB p4 SFT_1_4      | 0.5670 ~ 0.0057 | 0.5711 ~ 0.0054 | 0.4562 ~ 0.0089 |
|     exp2 HAN IMDB p5 SFT_1_4      | 0.5680 ~ 0.0090 | 0.5739 ~ 0.0084 | 0.4238 ~ 0.0047 |
|  exp2 HAN IMDB SFT_1_4_04_05_01   | 0.6186 ~ 0.0043 | 0.6208 ~ 0.0046 | 0.6045 ~ 0.0040 |
| exp2 HAN IMDB p1 SFT_1_4_04_05_01 | 0.6037 ~ 0.0072 | 0.6057 ~ 0.0075 | 0.5794 ~ 0.0068 |
| exp2 HAN IMDB p2 SFT_1_4_04_05_01 | 0.5900 ~ 0.0087 | 0.5935 ~ 0.0085 | 0.5435 ~ 0.0050 |
| exp2 HAN IMDB p3 SFT_1_4_04_05_01 | 0.5762 ~ 0.0090 | 0.5809 ~ 0.0087 | 0.5020 ~ 0.0082 |
| exp2 HAN IMDB p4 SFT_1_4_04_05_01 | 0.5676 ~ 0.0057 | 0.5715 ~ 0.0057 | 0.4579 ~ 0.0091 |
| exp2 HAN IMDB p5 SFT_1_4_04_05_01 | 0.5693 ~ 0.0111 | 0.5753 ~ 0.0101 | 0.4231 ~ 0.0049 |

|                Tag                |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------------------------------:|:---------------:|:---------------:|:---------------:|
|       exp2 HAN IMDB SFT_1_4       | 0.6172 ~ 0.0051 | 0.6192 ~ 0.0051 | 0.6015 ~ 0.0035 |
|     exp2 HAN IMDB u1 SFT_1_4      | 0.6020 ~ 0.0073 | 0.6043 ~ 0.0077 | 0.5870 ~ 0.0071 |
|     exp2 HAN IMDB u2 SFT_1_4      | 0.5842 ~ 0.0082 | 0.5874 ~ 0.0088 | 0.5545 ~ 0.0038 |
|     exp2 HAN IMDB u3 SFT_1_4      | 0.5580 ~ 0.0101 | 0.5625 ~ 0.0102 | 0.5070 ~ 0.0057 |
|     exp2 HAN IMDB u4 SFT_1_4      | 0.5336 ~ 0.0060 | 0.5367 ~ 0.0070 | 0.4770 ~ 0.0060 |
|     exp2 HAN IMDB u5 SFT_1_4      | 0.5089 ~ 0.0100 | 0.5147 ~ 0.0101 | 0.4344 ~ 0.0046 |
| exp2 HAN IMDB u1 SFT_1_4_04_05_01 | 0.6006 ~ 0.0058 | 0.6031 ~ 0.0065 | 0.5844 ~ 0.0052 |
| exp2 HAN IMDB u2 SFT_1_4_04_05_01 | 0.5805 ~ 0.0071 | 0.5841 ~ 0.0072 | 0.5516 ~ 0.0058 |
| exp2 HAN IMDB u3 SFT_1_4_04_05_01 | 0.5562 ~ 0.0104 | 0.5609 ~ 0.0104 | 0.5088 ~ 0.0058 |
| exp2 HAN IMDB u4 SFT_1_4_04_05_01 | 0.5335 ~ 0.0062 | 0.5365 ~ 0.0074 | 0.4745 ~ 0.0047 |
| exp2 HAN IMDB u5 SFT_1_4_04_05_01 | 0.5116 ~ 0.0114 | 0.5179 ~ 0.0112 | 0.4338 ~ 0.0064 |

