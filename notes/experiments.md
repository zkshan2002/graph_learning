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

### Group 3

- description
  - now corrupt_ratio is fixed
- known bugs
  - in HAN, setting type_aware_semantic to True in single control type dataset improves performance
- common settings
  - HAN
  - repeat=5

#### backup
##### exp0 IMDB

|    Tag    |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:---------:|:---------------:|:---------------:|:---------------:|
| exp0 IMDB | 0.6181 ~ 0.0042 | 0.6201 ~ 0.0044 | 0.5981 ~ 0.0029 | 

##### exp1 IMDB #noise

|     Tag      |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:------------:|:---------------:|:---------------:|:---------------:|
|  exp0 IMDB   | 0.6181 ~ 0.0042 | 0.6201 ~ 0.0044 | 0.5981 ~ 0.0029 | 
| exp1 IMDB p1 | 0.5953 ~ 0.0078 | 0.5974 ~ 0.0073 | 0.5651 ~ 0.0048 | 
| exp1 IMDB p2 | 0.5846 ~ 0.0054 | 0.5868 ~ 0.0054 | 0.5399 ~ 0.0059 | 
| exp1 IMDB p3 | 0.5764 ~ 0.0046 | 0.5784 ~ 0.0040 | 0.4889 ~ 0.0048 | 
| exp1 IMDB p4 | 0.5665 ~ 0.0085 | 0.5708 ~ 0.0089 | 0.4644 ~ 0.0055 | 
| exp1 IMDB p5 | 0.5632 ~ 0.0063 | 0.5671 ~ 0.0064 | 0.4107 ~ 0.0055 | 

|     Tag      |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:------------:|:---------------:|:---------------:|:---------------:|
|  exp0 IMDB   | 0.6181 ~ 0.0042 | 0.6201 ~ 0.0044 | 0.5981 ~ 0.0029 | 
| exp1 IMDB u1 | 0.5966 ~ 0.0085 | 0.5989 ~ 0.0080 | 0.5770 ~ 0.0040 | 
| exp1 IMDB u2 | 0.5788 ~ 0.0090 | 0.5820 ~ 0.0083 | 0.5395 ~ 0.0041 | 
| exp1 IMDB u3 | 0.5541 ~ 0.0113 | 0.5577 ~ 0.0101 | 0.4988 ~ 0.0063 | 
| exp1 IMDB u4 | 0.5263 ~ 0.0071 | 0.5303 ~ 0.0074 | 0.4704 ~ 0.0088 | 
| exp1 IMDB u5 | 0.5060 ~ 0.0098 | 0.5133 ~ 0.0091 | 0.4249 ~ 0.0080 | 

##### exp2 IMDB u4 mlc_#vlr_#Tlr

|            Tag             |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------:|:---------------:|:---------------:|:---------------:|
|        exp1 IMDB p4        | 0.5665 ~ 0.0085 | 0.5708 ~ 0.0089 | 0.4644 ~ 0.0055 | 
| exp2 IMDB p4 mlc_2e-4_1e-2 | 0.5624 ~ 0.0084 | 0.5672 ~ 0.0087 | 0.4648 ~ 0.0074 | 
| exp2 IMDB p4 mlc_2e-4_5e-2 | 0.5632 ~ 0.0082 | 0.5680 ~ 0.0084 | 0.4648 ~ 0.0074 | 
| exp2 IMDB p4 mlc_2e-4_1e-1 | 0.5629 ~ 0.0080 | 0.5678 ~ 0.0082 | 0.4645 ~ 0.0071 | 
| exp2 IMDB p4 mlc_5e-4_1e-2 | 0.5627 ~ 0.0087 | 0.5675 ~ 0.0088 | 0.4645 ~ 0.0075 | 
| exp2 IMDB p4 mlc_5e-4_5e-2 | 0.5630 ~ 0.0083 | 0.5678 ~ 0.0085 | 0.4644 ~ 0.0078 | 
| exp2 IMDB p4 mlc_5e-4_1e-1 | 0.5633 ~ 0.0079 | 0.5679 ~ 0.0081 | 0.4644 ~ 0.0068 | 
| exp2 IMDB p4 mlc_1e-3_1e-2 | 0.5629 ~ 0.0081 | 0.5676 ~ 0.0083 | 0.4647 ~ 0.0073 | 
| exp2 IMDB p4 mlc_1e-3_5e-2 | 0.5623 ~ 0.0080 | 0.5671 ~ 0.0081 | 0.4644 ~ 0.0072 |
| exp2 IMDB p4 mlc_1e-3_1e-1 | 0.5631 ~ 0.0086 | 0.5679 ~ 0.0088 | 0.4634 ~ 0.0074 | 

|            Tag             |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:--------------------------:|:---------------:|:---------------:|:---------------:|
|        exp1 IMDB u4        | 0.5263 ~ 0.0071 | 0.5303 ~ 0.0074 | 0.4704 ~ 0.0088 | 
| exp2 IMDB u4 mlc_1e-2_1e-1 | 0.5240 ~ 0.0102 | 0.5277 ~ 0.0094 | 0.4631 ~ 0.0099 | 
| exp2 IMDB u4 mlc_1e-2_1e-2 | 0.5248 ~ 0.0131 | 0.5281 ~ 0.0126 | 0.4718 ~ 0.0064 | 
| exp2 IMDB u4 mlc_1e-2_1e-3 | 0.5261 ~ 0.0131 | 0.5293 ~ 0.0127 | 0.4717 ~ 0.0065 | 


#### exp0 noise_u

- reduce sample_limit from 512 to 128, now as noise ratio increases, performance drop more normally
- further reducing 128 to 64 has no dramatic effect

|               Tag               |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------------------:|:---------------:|:---------------:|:---------------:|
|            exp0 DBLP            | 0.9303 ~ 0.0034 | 0.9349 ~ 0.0031 | 0.9351 ~ 0.0042 | 
|      exp0 DBLP noise_u 0.1      | 0.9306 ~ 0.0049 | 0.9349 ~ 0.0046 | 0.9220 ~ 0.0079 | 
|      exp0 DBLP noise_u 0.2      | 0.9275 ~ 0.0037 | 0.9324 ~ 0.0034 | 0.9167 ~ 0.0096 | 
|      exp0 DBLP noise_u 0.3      | 0.9158 ~ 0.0050 | 0.9213 ~ 0.0042 | 0.8762 ~ 0.0181 | 
|      exp0 DBLP noise_u 0.4      | 0.8782 ~ 0.0650 | 0.8842 ~ 0.0641 | 0.8286 ~ 0.0873 | 
|      exp0 DBLP noise_u 0.5      | 0.7733 ~ 0.0643 | 0.7828 ~ 0.0627 | 0.6585 ~ 0.0731 | 
|       exp0 DBLP sample 64       | 0.9296 ~ 0.0015 | 0.9342 ~ 0.0011 | 0.9347 ~ 0.0037 | 
| exp0 DBLP sample 64 noise_u 0.1 | 0.9251 ~ 0.0023 | 0.9296 ~ 0.0021 | 0.9223 ~ 0.0050 | 
| exp0 DBLP sample 64 noise_u 0.2 | 0.9223 ~ 0.0051 | 0.9275 ~ 0.0045 | 0.9066 ~ 0.0102 | 
| exp0 DBLP sample 64 noise_u 0.3 | 0.9161 ~ 0.0078 | 0.9216 ~ 0.0072 | 0.8888 ~ 0.0330 | 
| exp0 DBLP sample 64 noise_u 0.4 | 0.8495 ~ 0.0542 | 0.8590 ~ 0.0507 | 0.7824 ~ 0.0728 | 
| exp0 DBLP sample 64 noise_u 0.5 | 0.7811 ~ 0.0569 | 0.7913 ~ 0.0553 | 0.6606 ~ 0.0533 |

|               Tag               |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------------------------:|:---------------:|:---------------:|:---------------:|
|            exp0 IMDB            | 0.6200 ~ 0.0035 | 0.6215 ~ 0.0043 | 0.6052 ~ 0.0044 | 
|      exp0 IMDB noise_u 0.1      | 0.6039 ~ 0.0067 | 0.6060 ~ 0.0066 | 0.5832 ~ 0.0024 | 
|      exp0 IMDB noise_u 0.2      | 0.5814 ~ 0.0063 | 0.5841 ~ 0.0055 | 0.5515 ~ 0.0037 | 
|      exp0 IMDB noise_u 0.3      | 0.5618 ~ 0.0069 | 0.5653 ~ 0.0066 | 0.5034 ~ 0.0056 | 
|      exp0 IMDB noise_u 0.4      | 0.5359 ~ 0.0072 | 0.5396 ~ 0.0066 | 0.4770 ~ 0.0076 | 
|      exp0 IMDB noise_u 0.5      | 0.5087 ~ 0.0097 | 0.5147 ~ 0.0110 | 0.4249 ~ 0.0068 | 
|       exp0 IMDB sample 64       | 0.6203 ~ 0.0027 | 0.6220 ~ 0.0036 | 0.6030 ~ 0.0050 | 
| exp0 IMDB sample 64 noise_u 0.1 | 0.6011 ~ 0.0059 | 0.6031 ~ 0.0061 | 0.5823 ~ 0.0025 | 
| exp0 IMDB sample 64 noise_u 0.2 | 0.5811 ~ 0.0052 | 0.5836 ~ 0.0053 | 0.5486 ~ 0.0036 | 
| exp0 IMDB sample 64 noise_u 0.3 | 0.5600 ~ 0.0106 | 0.5634 ~ 0.0099 | 0.5017 ~ 0.0040 | 
| exp0 IMDB sample 64 noise_u 0.4 | 0.5360 ~ 0.0080 | 0.5394 ~ 0.0080 | 0.4756 ~ 0.0079 | 
| exp0 IMDB sample 64 noise_u 0.5 | 0.5110 ~ 0.0066 | 0.5175 ~ 0.0062 | 0.4271 ~ 0.0064 | 

#### exp1 noise_u 0.4 sft_filter

- best hyperparameter: memory = 1, warmup = 4
- fluctuate ratio 0 at beginning, then grow to ~0.10

|                          Tag                          | Macro-F1 | Micro-F1 | Accuracy |
|:-----------------------------------------------------:|:--------:|:--------:|:--------:|
|            exp0 DBLP sample 64 noise_u 0.4            |  0.8495  |  0.8590  |  0.7824  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 1 sft_warmup 2 |  0.8327  |  0.8413  |  0.7545  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 1 sft_warmup 4 |  0.9013  |  0.9075  |  0.8356  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 1 sft_warmup 6 |  0.8625  |  0.8703  |  0.7889  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 1 sft_warmup 8 |  0.8495  |  0.8590  |  0.7824  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 2 sft_warmup 2 |  0.8488  |  0.8580  |  0.7490  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 2 sft_warmup 4 |  0.8894  |  0.8967  |  0.8161  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 2 sft_warmup 6 |  0.8655  |  0.8732  |  0.7925  | 
| exp1 DBLP noise_u 0.4sft_filter_memory 2 sft_warmup 8 |  0.8495  |  0.8590  |  0.7824  | 

- sft_filter improves f1 by ~0.02 when noise ratio is large

|               Tag                | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------:|:--------:|:--------:|:--------:|
|            exp0 DBLP             |  0.9303  |  0.9349  |  0.9351  | 
|      exp0 DBLP noise_u 0.1       |  0.9306  |  0.9349  |  0.9220  | 
|      exp0 DBLP noise_u 0.2       |  0.9275  |  0.9324  |  0.9167  | 
|      exp0 DBLP noise_u 0.3       |  0.9158  |  0.9213  |  0.8762  | 
|      exp0 DBLP noise_u 0.4       |  0.8782  |  0.8842  |  0.8286  | 
|      exp0 DBLP noise_u 0.5       |  0.7733  |  0.7828  |  0.6585  | 

|               Tag                | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------:|:--------:|:--------:|:--------:|
|       exp0 DBLP sample 64        |  0.9296  |  0.9342  |  0.9347  | 
| exp0 DBLP sample 64 noise_u 0.1  |  0.9251  |  0.9296  |  0.9223  | 
| exp0 DBLP sample 64 noise_u 0.2  |  0.9223  |  0.9275  |  0.9066  | 
| exp0 DBLP sample 64 noise_u 0.3  |  0.9161  |  0.9216  |  0.8888  | 
| exp0 DBLP sample 64 noise_u 0.4  |  0.8495  |  0.8590  |  0.7824  | 
| exp0 DBLP sample 64 noise_u 0.5  |  0.7811  |  0.7913  |  0.6606  |
|       exp1 DBLP sft_filter       |  0.9293  |  0.9338  |  0.9332  | 
| exp1 DBLP noise_u 0.1 sft_filter |  0.9273  |  0.9320  |  0.9240  |
| exp1 DBLP noise_u 0.2 sft_filter |  0.9204  |  0.9257  |  0.9119  |
| exp1 DBLP noise_u 0.3 sft_filter |  0.9161  |  0.9215  |  0.8814  | 
| exp1 DBLP noise_u 0.4 sft_filter |  0.9013  |  0.9075  |  0.8356  | 
| exp1 DBLP noise_u 0.5 sft_filter |  0.7981  |  0.8083  |  0.6794  | 

|               Tag                | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------:|:--------:|:--------:|:--------:|
| exp0 DBLP sample 64 noise_u 0.4  |  0.8495  |  0.8590  |  0.7824  | 
| exp1 DBLP noise_u 0.4 sft_filter |  0.9013  |  0.9075  |  0.8356  | 


IMDB

|                 Tag                  | Macro-F1 | Micro-F1 | Accuracy |
|:------------------------------------:|:--------:|:--------:|:--------:|
|        exp0 IMDB noise_u 0.4         |  0.5359  |  0.5396  |  0.4770  | 
| exp1 IMDB noise_u 0.4 sft_filter 1 2 |  0.5355  |  0.5389  |  0.4730  | 
| exp1 IMDB noise_u 0.4 sft_filter 1 4 |  0.5356  |  0.5390  |  0.4711  | 
| exp1 IMDB noise_u 0.4 sft_filter 1 6 |  0.5345  |  0.5383  |  0.4722  | 
| exp1 IMDB noise_u 0.4 sft_filter 1 8 |  0.5360  |  0.5396  |  0.4727  | 
| exp1 IMDB noise_u 0.4 sft_filter 2 2 |  0.5355  |  0.5387  |  0.4777  | 
| exp1 IMDB noise_u 0.4 sft_filter 2 4 |  0.5341  |  0.5373  |  0.4713  | 
| exp1 IMDB noise_u 0.4 sft_filter 2 6 |  0.5359  |  0.5396  |  0.4632  | 
| exp1 IMDB noise_u 0.4 sft_filter 2 8 |  0.5339  |  0.5373  |  0.4657  | 


|                 Tag                  | Macro-F1 | Micro-F1 | Accuracy |
|:------------------------------------:|:--------:|:--------:|:--------:|
|              exp0 DBLP               |  0.9303  |  0.9349  |  0.9351  | 
|        exp0 DBLP noise_u 0.1         |  0.9306  |  0.9349  |  0.9220  | 
|        exp0 DBLP noise_u 0.2         |  0.9275  |  0.9324  |  0.9167  | 
|        exp0 DBLP noise_u 0.3         |  0.9158  |  0.9213  |  0.8762  | 
|        exp0 DBLP noise_u 0.4         |  0.8782  |  0.8842  |  0.8286  | 
| exp1 DBLP noise_u 0.4 sft_filter 1 2 |  0.8789  |  0.8851  |  0.8135  | 
| exp1 DBLP noise_u 0.4 sft_filter 1 4 | 0.8826 | 0.8887 | 0.8296 | 
| exp1 DBLP noise_u 0.4 sft_filter 1 6 | 0.8535 | 0.8611 | 0.7754 | 
| exp1 DBLP noise_u 0.4 sft_filter 1 8 | 0.8531 | 0.8609 | 0.7713 | 
| exp1 DBLP noise_u 0.4 sft_filter 2 2 | 0.8835 | 0.8898 | 0.8253 | 
| exp1 DBLP noise_u 0.4 sft_filter 2 4 | 0.8763 | 0.8826 | 0.8092 | 
| exp1 DBLP noise_u 0.4 sft_filter 2 6 | 0.8793 | 0.8852 | 0.8276 | 
| exp1 DBLP noise_u 0.4 sft_filter 2 8 | 0.8662 | 0.8729 | 0.8034 | 



grid search for best sft_filter param conditioned on noise ratio
128

|                 Tag                  | Macro-F1 | Micro-F1 | Accuracy | Comment |
|:------------------------------------:|:--------:|:--------:|:--------:|:-------:|
|             exp1 DBLP128             |  0.9303  |  0.9349  |  0.9351  |   0.0   |
|        exp1 DBLP noise_u 0.1         |  0.9306  |  0.9349  |  0.9220  |   0.1   |
| exp1 DBLP noise_u 0.1 sft_filter 1 2 |  0.9269  |  0.9320  |  0.9191  |         | 
| exp1 DBLP noise_u 0.1 sft_filter 1 4 |  0.9274  |  0.9321  |  0.9201  |         | 
| exp1 DBLP noise_u 0.1 sft_filter 1 6 |  0.9272  |  0.9318  |  0.9184  |         | 
| exp1 DBLP noise_u 0.1 sft_filter 1 8 |  0.9301  |  0.9344  |  0.9221  |  0.1*   | 
|        exp1 DBLP noise_u 0.2         |  0.9275  |  0.9324  |  0.9167  |   0.2   |
| exp1 DBLP noise_u 0.2 sft_filter 1 2 |  0.9217  |  0.9270  |  0.9035  |         | 
| exp1 DBLP noise_u 0.2 sft_filter 1 4 |  0.9269  |  0.9317  |  0.9121  |  0.2*   |
| exp1 DBLP noise_u 0.2 sft_filter 1 6 |  0.9257  |  0.9305  |  0.9066  |         |
| exp1 DBLP noise_u 0.2 sft_filter 1 8 |  0.9238  |  0.9289  |  0.9052  |         |
|        exp1 DBLP noise_u 0.3         |  0.9158  |  0.9213  |  0.8762  |   0.3   |
| exp1 DBLP noise_u 0.3 sft_filter 1 2 |  0.9175  |  0.9229  |  0.8631  |         |
| exp1 DBLP noise_u 0.3 sft_filter 1 4 |  0.9212  |  0.9265  |  0.8793  |         |
| exp1 DBLP noise_u 0.3 sft_filter 1 6 |  0.9239  |  0.9287  |  0.8910  |  0.3*   |
| exp1 DBLP noise_u 0.3 sft_filter 1 8 |  0.9164  |  0.9221  |  0.8764  |         |
|        exp1 DBLP noise_u 0.4         |  0.8782  |  0.8842  |  0.8286  |   0.4   |
| exp1 DBLP noise_u 0.4 sft_filter 1 2 |  0.8789  |  0.8851  |  0.8135  |         |
| exp1 DBLP noise_u 0.4 sft_filter 1 4 |  0.8826  |  0.8887  |  0.8296  |  0.4*   |
| exp1 DBLP noise_u 0.4 sft_filter 1 6 |  0.8535  |  0.8611  |  0.7754  |         |
| exp1 DBLP noise_u 0.4 sft_filter 1 8 |  0.8531  |  0.8609  |  0.7713  |         |
|        exp1 DBLP noise_u 0.5         |  0.7733  |  0.7828  |  0.6585  |   0.5   |
| exp1 DBLP noise_u 0.5 sft_filter 1 2 |  0.7614  |  0.7699  |  0.6289  |         |
| exp1 DBLP noise_u 0.5 sft_filter 1 4 |  0.7741  |  0.7824  |  0.6550  |         |
| exp1 DBLP noise_u 0.5 sft_filter 1 6 |  0.7829  |  0.7923  |  0.6669  |         |
| exp1 DBLP noise_u 0.5 sft_filter 1 8 |  0.7851  |  0.7946  |  0.6681  |  0.5*   |

64

|                  Tag                   | Macro-F1 | Micro-F1 | Accuracy | Comment |
|:--------------------------------------:|:--------:|:--------:|:--------:|:-------:|
|        exp3 DBLP64 noise_u 0.1         |  0.9251  |  0.9296  |  0.9223  |   0.1   |
| exp3 DBLP64 noise_u 0.1 sft_filter 1 2 |  0.9236  |  0.9286  |  0.9221  | 
| exp3 DBLP64 noise_u 0.1 sft_filter 1 4 |  0.9273  |  0.9320  |  0.9240  |  0.1*   |
| exp3 DBLP64 noise_u 0.1 sft_filter 1 6 |  0.9243  |  0.9294  |  0.9180  | 
| exp3 DBLP64 noise_u 0.1 sft_filter 1 8 |  0.9271  |  0.9319  |  0.9239  | 
|        exp3 DBLP64 noise_u 0.2         |  0.9223  |  0.9275  |  0.9066  |   0.2   |
| exp3 DBLP64 noise_u 0.2 sft_filter 1 2 |  0.9224  |  0.9277  |  0.8957  |  0.2*   |
| exp3 DBLP64 noise_u 0.2 sft_filter 1 4 |  0.9204  |  0.9257  |  0.9119  | 
| exp3 DBLP64 noise_u 0.2 sft_filter 1 6 |  0.9175  |  0.9230  |  0.8974  | 
| exp3 DBLP64 noise_u 0.2 sft_filter 1 8 |  0.9175  |  0.9231  |  0.8962  | 
|        exp3 DBLP64 noise_u 0.3         |  0.9161  |  0.9216  |  0.8888  |   0.3   |
| exp3 DBLP64 noise_u 0.3 sft_filter 1 2 |  0.9141  |  0.9195  |  0.8687  | 
| exp3 DBLP64 noise_u 0.3 sft_filter 1 4 |  0.9161  |  0.9215  |  0.8814  | 
| exp3 DBLP64 noise_u 0.3 sft_filter 1 6 |  0.9125  |  0.9180  |  0.8711  | 
| exp3 DBLP64 noise_u 0.3 sft_filter 1 8 |  0.9204  |  0.9256  |  0.9024  |  0.3*   |
|        exp3 DBLP64 noise_u 0.4         |  0.8495  |  0.8590  |  0.7824  |   0.4   |
| exp3 DBLP64 noise_u 0.4 sft_filter 1 2 |  0.8327  |  0.8413  |  0.7545  | 
| exp3 DBLP64 noise_u 0.4 sft_filter 1 4 |  0.9013  |  0.9075  |  0.8356  |  0.4*   |
| exp3 DBLP64 noise_u 0.4 sft_filter 1 6 |  0.8625  |  0.8703  |  0.7889  | 
| exp3 DBLP64 noise_u 0.4 sft_filter 1 8 |  0.8495  |  0.8590  |  0.7824  | 
|        exp3 DBLP64 noise_u 0.5         |  0.7811  |  0.7913  |  0.6606  |   0.5   |
| exp3 DBLP64 noise_u 0.5 sft_filter 1 2 |  0.7624  |  0.7726  |  0.6389  | 
| exp3 DBLP64 noise_u 0.5 sft_filter 1 4 |  0.7981  |  0.8083  |  0.6794  |  0.5*   |
| exp3 DBLP64 noise_u 0.5 sft_filter 1 6 |  0.7556  |  0.7660  |  0.6257  | 
| exp3 DBLP64 noise_u 0.5 sft_filter 1 8 |  0.7773  |  0.7874  |  0.6595  | 

|                  Tag                   | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------------:|:--------:|:--------:|:--------:|
|        exp3 DBLP64 noise_u 0.4         |  0.8495  |  0.8590  |  0.7824  |
| exp4 DBLP64 noise_u 0.4 mlc 0.002 0.05 |  0.8789  |  0.8854  |  0.8114  |
| exp4 DBLP64 noise_u 0.4 mlc 0.002 0.10 |  0.8620  |  0.8692  |  0.7913  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.002 0.20 |  0.8456  |  0.8534  |  0.7654  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.005 0.05 |  0.8655  |  0.8723  |  0.7920  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.005 0.10 |  0.8746  |  0.8815  |  0.8020  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.005 0.20 |  0.8769  |  0.8840  |  0.8190  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.005 0.5  |  0.8630  |  0.8698  |  0.7876  | 
|  exp4 DBLP64 noise_u 0.4 mlc 0.005 1   |  0.8502  |  0.8576  |  0.7666  | 
|  exp4 DBLP64 noise_u 0.4 mlc 0.005 2   |  0.8547  |  0.8619  |  0.7700  | 
|  exp4 DBLP64 noise_u 0.4 mlc 0.005 5   |  0.8608  |  0.8678  |  0.7609  | 
|  exp4 DBLP64 noise_u 0.4 mlc 0.005 10  |  0.8266  |  0.8358  |  0.6990  | 

| exp4 DBLP64 noise_u 0.4 mlc 0.010 0.05 |  0.8584  |  0.8653  |  0.7806  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.010 0.10 |  0.8594  |  0.8664  |  0.7919  | 
| exp4 DBLP64 noise_u 0.4 mlc 0.010 0.20 |  0.8556  |  0.8630  |  0.7767  | 

|               Tag               | Macro-F1 | Micro-F1 | Accuracy |
|:-------------------------------:|:--------:|:--------:|:--------:|
|     exp3 DBLP64 noise_u 0.5     |  0.7811  |  0.7913  |  0.6606  |
| exp5 DBLP64 noise_u 0.5 mlc 0.05 | 0.7497 | 0.7595 | 0.6328 |
| exp5 DBLP64 noise_u 0.5 mlc 0.1 | 0.7820 | 0.7914 | 0.6516 | 
| exp5 DBLP64 noise_u 0.5 mlc 0.2 | 0.7936 | 0.8029 | 0.6636 | 
| exp5 DBLP64 noise_u 0.5 mlc 0.5 | 0.7749 | 0.7838 | 0.6357 | 
| exp5 DBLP64 noise_u 0.5 mlc 1 | 0.7807 | 0.7894 | 0.6314 | 
| exp5 DBLP64 noise_u 0.5 mlc 2 | 0.7451 | 0.7557 | 0.6071 | 
| exp5 DBLP64 noise_u 0.5 mlc 5 | 0.7595 | 0.7690 | 0.6186 | 


|             Tag             | Macro-F1 | Micro-F1 | Accuracy |
|:---------------------------:|:--------:|:--------:|:--------:|
|   exp3 DBLP64 noise_u 0.1   |  0.9251  |  0.9296  |  0.9223  |
| exp5 DBLP64 noise_u 0.1 mlc |  0.9243  |  0.9293  |  0.9232  |
|   exp3 DBLP64 noise_u 0.2   |  0.9223  |  0.9275  |  0.9066  |
| exp5 DBLP64 noise_u 0.2 mlc |  0.9264  |  0.9314  |  0.9154  |
|   exp3 DBLP64 noise_u 0.3   |  0.9161  |  0.9216  |  0.8888  | 
| exp5 DBLP64 noise_u 0.3 mlc |  0.9177  |  0.9230  |  0.8943  |
|   exp3 DBLP64 noise_u 0.4   |  0.8495  |  0.8590  |  0.7824  |
| exp5 DBLP64 noise_u 0.4 mlc |  0.8769  |  0.8840  |  0.8190  |
|   exp3 DBLP64 noise_u 0.5   |  0.7811  |  0.7913  |  0.6606  | 
| exp5 DBLP64 noise_u 0.5 mlc |  0.7936  |  0.8029  |  0.6636  | 


IMDB

|                  Tag                   | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------------:|:--------:|:--------:|:--------:|
|        exp6 IMDB64 noise_u 0.1         |  0.6011  |  0.6031  |  0.5823  | 
| exp6 IMDB64 noise_u 0.1 sft_filter 1 2 |  0.6023  |  0.6042  |  0.5818  | 
| exp6 IMDB64 noise_u 0.1 sft_filter 1 4 |  0.6025  |  0.6043  |  0.5800  | 
| exp6 IMDB64 noise_u 0.1 sft_filter 1 6 |  0.6016  |  0.6038  |  0.5843  | 
| exp6 IMDB64 noise_u 0.1 sft_filter 1 8 |  0.6032  |  0.6050  |  0.5812  | 
|        exp6 IMDB64 noise_u 0.2         |  0.5811  |  0.5836  |  0.5486  | 
| exp6 IMDB64 noise_u 0.2 sft_filter 1 2 |  0.5805  |  0.5830  |  0.5501  | 
| exp6 IMDB64 noise_u 0.2 sft_filter 1 4 |  0.5828  |  0.5852  |  0.5508  | 
| exp6 IMDB64 noise_u 0.2 sft_filter 1 6 |  0.5821  |  0.5846  |  0.5501  | 
| exp6 IMDB64 noise_u 0.2 sft_filter 1 8 |  0.5825  |  0.5851  |  0.5492  | 
|        exp6 IMDB64 noise_u 0.3         |  0.5600  |  0.5634  |  0.5017  | 
| exp6 IMDB64 noise_u 0.3 sft_filter 1 2 |  0.5589  |  0.5623  |  0.5018  | 
| exp6 IMDB64 noise_u 0.3 sft_filter 1 4 |  0.5613  |  0.5648  |  0.5020  | 
| exp6 IMDB64 noise_u 0.3 sft_filter 1 6 |  0.5602  |  0.5636  |  0.5003  |
| exp6 IMDB64 noise_u 0.3 sft_filter 1 8 |  0.5598  |  0.5632  |  0.4984  | 
|        exp6 IMDB64 noise_u 0.4         |  0.5360  |  0.5394  |  0.4756  | 
| exp6 IMDB64 noise_u 0.4 sft_filter 1 2 |  0.5355  |  0.5389  |  0.4730  | 
| exp6 IMDB64 noise_u 0.4 sft_filter 1 4 |  0.5356  |  0.5390  |  0.4711  | 
| exp6 IMDB64 noise_u 0.4 sft_filter 1 6 |  0.5345  |  0.5383  |  0.4722  | 
| exp6 IMDB64 noise_u 0.4 sft_filter 1 8 |  0.5360  |  0.5396  |  0.4727  | 
|        exp6 IMDB64 noise_u 0.5         |  0.5110  |  0.5175  |  0.4271  | 
| exp6 IMDB64 noise_u 0.5 sft_filter 1 2 |  0.5101  |  0.5166  |  0.4235  | 
| exp6 IMDB64 noise_u 0.5 sft_filter 1 4 |  0.5095  |  0.5162  |  0.4252  | 
| exp6 IMDB64 noise_u 0.5 sft_filter 1 6 |  0.5098  |  0.5160  |  0.4262  | 
| exp6 IMDB64 noise_u 0.5 sft_filter 1 8 |  0.5102  |  0.5162  |  0.4265  | 

|               Tag                | Macro-F1 | Micro-F1 | Accuracy |
|:--------------------------------:|:--------:|:--------:|:--------:|
|     exp6 IMDB64 noise_u 0.1      |  0.6011  |  0.6031  |  0.5823  | 
| exp7 IMDB64 noise_u 0.1 mlc 0.05 |  0.6004  |  0.6026  |  0.5808  |
| exp7 IMDB64 noise_u 0.1 mlc 0.1  |  0.6010  |  0.6031  |  0.5808  | 
| exp7 IMDB64 noise_u 0.1 mlc 0.2  |  0.6007  |  0.6028  |  0.5805  | 
| exp7 IMDB64 noise_u 0.1 mlc 0.5  |  0.6005  |  0.6027  |  0.5815  | 
|  exp7 IMDB64 noise_u 0.1 mlc 1   |  0.6004  |  0.6025  |  0.5814  | 
|  exp7 IMDB64 noise_u 0.1 mlc 2   |  0.6009  |  0.6032  |  0.5824  | 
|  exp7 IMDB64 noise_u 0.1 mlc 5   |  0.5991  |  0.6011  |  0.5837  | 
|     exp6 IMDB64 noise_u 0.2      |  0.5811  |  0.5836  |  0.5486  | 
| exp7 IMDB64 noise_u 0.2 mlc 0.05 |  0.5790  |  0.5818  |  0.5506  |
| exp7 IMDB64 noise_u 0.2 mlc 0.1  |  0.5790  |  0.5818  |  0.5504  | 
| exp7 IMDB64 noise_u 0.2 mlc 0.2  |  0.5791  |  0.5819  |  0.5502  | 
| exp7 IMDB64 noise_u 0.2 mlc 0.5  |  0.5787  |  0.5815  |  0.5502  | 
|  exp7 IMDB64 noise_u 0.2 mlc 1   |  0.5789  |  0.5818  |  0.5504  | 
|  exp7 IMDB64 noise_u 0.2 mlc 2   |  0.5793  |  0.5822  |  0.5506  | 
|  exp7 IMDB64 noise_u 0.2 mlc 5   |  0.5797  |  0.5825  |  0.5507  | 
|     exp6 IMDB64 noise_u 0.3      |  0.5600  |  0.5634  |  0.5017  | 
| exp7 IMDB64 noise_u 0.3 mlc 0.05 |  0.5570  |  0.5608  |  0.5011  | 
| exp7 IMDB64 noise_u 0.3 mlc 0.1  |  0.5579  |  0.5617  |  0.5015  | 
| exp7 IMDB64 noise_u 0.3 mlc 0.2  |  0.5575  |  0.5612  |  0.5019  | 
| exp7 IMDB64 noise_u 0.3 mlc 0.5  |  0.5574  |  0.5612  |  0.5025  | 
|  exp7 IMDB64 noise_u 0.3 mlc 1   |  0.5572  |  0.5609  |  0.5041  | 
|  exp7 IMDB64 noise_u 0.3 mlc 2   |  0.5580  |  0.5611  |  0.5109  | 
|  exp7 IMDB64 noise_u 0.3 mlc 5   |  0.5579  |  0.5610  |  0.5121  |
|     exp6 IMDB64 noise_u 0.4      |  0.5360  |  0.5394  |  0.4756  | 
| exp7 IMDB64 noise_u 0.4 mlc 0.05 |  0.5313  |  0.5346  |  0.4779  | 
| exp7 IMDB64 noise_u 0.4 mlc 0.1  |  0.5311  |  0.5344  |  0.4782  | 
| exp7 IMDB64 noise_u 0.4 mlc 0.2  |  0.5303  |  0.5336  |  0.4782  | 
| exp7 IMDB64 noise_u 0.4 mlc 0.5  |  0.5303  |  0.5336  |  0.4780  | 
|  exp7 IMDB64 noise_u 0.4 mlc 1   |  0.5301  |  0.5331  |  0.4782  | 
|  exp7 IMDB64 noise_u 0.4 mlc 2   |  0.5296  |  0.5326  |  0.4791  | 
|  exp7 IMDB64 noise_u 0.4 mlc 5   |  0.5297  |  0.5328  |  0.4773  |
|     exp6 IMDB64 noise_u 0.5      |  0.5110  |  0.5175  |  0.4271  | 
| exp7 IMDB64 noise_u 0.5 mlc 0.05 |  0.5109  |  0.5177  |  0.4273  | 
| exp7 IMDB64 noise_u 0.5 mlc 0.1  |  0.5106  |  0.5173  |  0.4276  | 
| exp7 IMDB64 noise_u 0.5 mlc 0.2  |  0.5112  |  0.5180  |  0.4275  | 
| exp7 IMDB64 noise_u 0.5 mlc 0.5  |  0.5112  |  0.5178  |  0.4275  | 
|  exp7 IMDB64 noise_u 0.5 mlc 1   |  0.5089  |  0.5156  |  0.4266  | 
|  exp7 IMDB64 noise_u 0.5 mlc 2   |  0.5094  |  0.5165  |  0.4270  | 
|  exp7 IMDB64 noise_u 0.5 mlc 5   |  0.5096  |  0.5166  |  0.4260  | 


IMDB: no improvement

|                 Tag                  | Macro-F1 | Micro-F1 | Accuracy |
|:------------------------------------:|:--------:|:--------:|:--------:|
|        exp2 IMDB noise_u 0.1         |  0.6039  |  0.6060  |  0.5832  | 
| exp2 IMDB noise_u 0.1 sft_filter 1 2 |  0.6020  |  0.6039  |  0.5826  | 
| exp2 IMDB noise_u 0.1 sft_filter 1 4 |  0.6017  |  0.6035  |  0.5836  | 
| exp2 IMDB noise_u 0.1 sft_filter 1 6 |  0.6021  |  0.6040  |  0.5850  | 
| exp2 IMDB noise_u 0.1 sft_filter 1 8 |  0.6019  |  0.6039  |  0.5842  | 
|        exp2 IMDB noise_u 0.2         |  0.5814  |  0.5841  |  0.5515  | 
| exp2 IMDB noise_u 0.2 sft_filter 1 2 |  0.5823  |  0.5850  |  0.5515  | 
| exp2 IMDB noise_u 0.2 sft_filter 1 4 |  0.5823  |  0.5852  |  0.5518  | 
| exp2 IMDB noise_u 0.2 sft_filter 1 6 |  0.5823  |  0.5849  |  0.5506  | 
| exp2 IMDB noise_u 0.2 sft_filter 1 8 |  0.5824  |  0.5852  |  0.5511  | 
|        exp2 IMDB noise_u 0.3         |  0.5618  |  0.5653  |  0.5034  | 
| exp2 IMDB noise_u 0.3 sft_filter 1 2 |  0.5618  |  0.5653  |  0.5028  | 
| exp2 IMDB noise_u 0.3 sft_filter 1 4 |  0.5628  |  0.5664  |  0.5005  | 
| exp2 IMDB noise_u 0.3 sft_filter 1 6 |  0.5623  |  0.5660  |  0.5040  | 
| exp2 IMDB noise_u 0.3 sft_filter 1 8 |  0.5625  |  0.5656  |  0.5016  | 
|        exp2 IMDB noise_u 0.4         |  0.5359  |  0.5396  |  0.4770  | 
| exp2 IMDB noise_u 0.4 sft_filter 1 2 |  0.5362  |  0.5401  |  0.4755  | 
| exp2 IMDB noise_u 0.4 sft_filter 1 4 |  0.5340  |  0.5372  |  0.4734  | 
| exp2 IMDB noise_u 0.4 sft_filter 1 6 |  0.5377  |  0.5413  |  0.4749  | 
| exp2 IMDB noise_u 0.4 sft_filter 1 8 |  0.5330  |  0.5368  |  0.4754  | 
|        exp2 IMDB noise_u 0.5         |  0.5087  |  0.5147  |  0.4249  | 
| exp2 IMDB noise_u 0.5 sft_filter 1 2 |  0.5096  |  0.5159  |  0.4304  | 
| exp2 IMDB noise_u 0.5 sft_filter 1 4 |  0.5085  |  0.5150  |  0.4262  | 
| exp2 IMDB noise_u 0.5 sft_filter 1 6 |  0.5093  |  0.5162  |  0.4282  | 
| exp2 IMDB noise_u 0.5 sft_filter 1 8 |  0.5095  |  0.5162  |  0.4308  | 

