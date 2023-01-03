## Setup
### 1. Dataset

#### DBLP

- Node

| Type ID | Semantic  | Count | Feature Dim |            Feature Source            |
|:-------:|:---------:|------:|------------:|:------------------------------------:|
|    0    | author(A) |  4057 |         334 | a bag of words rep of paper keywords |
|    1    | paper(P)  | 14328 |        4231 |      lemma tokenizer transform       |
|    2    |  term(T)  |  7723 |          50 |                glove                 |
|    3    |  conf(C)  |    20 |          20 |                  id                  |

- Edge

| Type ID | Semantic | Total |
|:-------:|:--------:|------:|
|   0-1   |   P-A    | 19645 |
|   1-2   |   P-T    | 85810 |
|   1-3   |   P-C    | 14328 |

- Metapath
  - connects node type 0(A) only

|  Type ID  | Semantic  |    Total |    Mean    |    Max | Min |
|:---------:|:---------:|---------:|:----------:|-------:|----:|
|   0-1-0   |   A-P-A   |    32789 |   8.0821   |    325 |   1 |
| 0-1-2-1-0 | A-P-T-P-A | 41633537 | 10262.1486 | 482033 |   6 |
| 0-1-3-1-0 | A-P-C-P-A | 30803571 | 7592.6968  | 245377 | 284 |

- Labels
  - on node type 0(A) only
  - labeled according to the conferences the authors submitted(?)

| Class ID |       Semantic        | Total |
|:--------:|:---------------------:|------:|
|    0     |       Database        |  1197 |
|    1     |      Data Mining      |   745 |
|    2     |          AI           |  1109 |
|    3     | Information Retrieval |  1006 |

#### IMDB

- Node

| Type ID |  Semantic   | Count | Feature Dim |          Feature Source           |
|:-------:|:-----------:|------:|------------:|:---------------------------------:|
|    0    |  movie(M)   |  4278 |        3066 | bag of words rep of plot keywords |
|    1    | director(D) |  2081 |        3066 |     bag of words rep of name      |
|    2    |  actor(A)   |  5257 |        3066 |     bag of words rep of name      |

- Edge

| Type ID | Semantic | Total |
|:-------:|:--------:|------:|
|   0-1   |   M-D    |  4278 |
|   0-2   |   M-A    | 12828 |

- Metapath
  - only metapaths that connect control type nodes(0-1-0, 0-2-0) are selected

|  Type ID  | Semantic | Total |  Mean   | Max | Min |
|:---------:|:--------:|------:|:-------:|----:|----:|
|   0-1-0   |  M-D-M   | 17446 | 4.0781  |  22 |   1 |
|   0-2-0   |  M-A-M   | 95102 | 22.2305 | 107 |   1 |

- Labels
  - on node type 0(A) only
  - labeled according to genre information

| Class ID |       Semantic        | Total |
|:--------:|:---------------------:|------:|
|    0     |        Action         |  1135 |
|    1     |        Comedy         |  1584 |
|    2     |         Drama         |  1559 |

### 2. Label Noise

- pair noise:
  - each label type has prob of p to be flipped to another specified label type
  - implementation
    - DBLP: 0 Database, 1 Data Mining, 2 AI, 3 Information Retrieval
      - [1, 0, 3, 2]
    - IMDB: 0 Action, 1 Comedy, 2 Drama
      - [1, 2, 0]
- uniform noise:
  - each label type has prob of p to be uniformly flipped to other label types
- only one type of noise is considered at a time

### 3. Training

- sample strategy
  - in each iteration, select <batch_size> of labeled nodes as target nodes
  - for each target node, and for each metapath schemes, sample up to <sample_limit> of metapath instances
  - train on sampled metapaths only
  - this implies that only metapaths that ends with control type(the only node type with label) are utilized
- hyperparam tuning
  - tweaked batch_size, sample_limit, lr, lr_decay
  - tweak hyperparams to reach maximum performance with clean label
  - haven't tweak model dims, but has already reached performance in MAGNN paper
- training with early stopping
  - when validation loss doesn't improve for patience=10 consecutive epochs, stop training
- in each experiment, repeat 10 times and report averaged results
  - in practice, model is quite sensitive to random seed

## Results

- Macro-F1 and Micro-F1 are computed on learnt embeddings, by additional svm classifier(train/test ratio 0.8/0.2)
- Accuracy is test-time classification accuracy, computed with learnt classification head
- reports mean and std on 10 results on different random seeds

### 1. baseline

- both noises hurt HAN's performance, test time accuracy suffers more than f1 score
  - cls head is susceptible to noisy label, but learnt embeddings are less influenced
- uniform noise hurts more than pair noise
  - pair noise may be less noisy in semantics
- performance on DBLP is more robust than IMDB

- DBLP

| Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------:|:---------------:|:---------------:|:---------------:|
|    0.0     | 0.9393 ~ 0.0043 | 0.9430 ~ 0.0041 | 0.9382 ~ 0.0039 |
|    0.1     | 0.9359 ~ 0.0023 | 0.9403 ~ 0.0022 | 0.9320 ~ 0.0026 |
|    0.2     | 0.9355 ~ 0.0041 | 0.9398 ~ 0.0037 | 0.9179 ~ 0.0141 |
|    0.3     | 0.9345 ~ 0.0045 | 0.9388 ~ 0.0039 | 0.9033 ~ 0.0152 |
|    0.4     | 0.9356 ~ 0.0044 | 0.9399 ~ 0.0041 | 0.8868 ~ 0.0145 |
|    0.5     | 0.9314 ~ 0.0069 | 0.9360 ~ 0.0063 | 0.5770 ~ 0.0780 |

| Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------:|:---------------:|:---------------:|:---------------:|
|      0.0      | 0.9393 ~ 0.0043 | 0.9430 ~ 0.0041 | 0.9382 ~ 0.0039 |
|      0.1      | 0.9362 ~ 0.0035 | 0.9403 ~ 0.0034 | 0.9331 ~ 0.0069 |
|      0.2      | 0.9351 ~ 0.0056 | 0.9393 ~ 0.0052 | 0.9188 ~ 0.0139 |
|      0.3      | 0.9361 ~ 0.0076 | 0.9403 ~ 0.0072 | 0.9153 ~ 0.0075 |
|      0.4      | 0.9165 ~ 0.0506 | 0.9217 ~ 0.0481 | 0.8747 ~ 0.0891 |
|      0.5      | 0.8873 ~ 0.0529 | 0.8937 ~ 0.0516 | 0.7942 ~ 0.0894 |

- IMDB

| Pair_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:----------:|:---------------:|:---------------:|:---------------:|
|    0.0     | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
|    0.1     | 0.5990 ~ 0.0092 | 0.6006 ~ 0.0091 | 0.5779 ~ 0.0053 |
|    0.2     | 0.5874 ~ 0.0102 | 0.5897 ~ 0.0102 | 0.5419 ~ 0.0064 |
|    0.3     | 0.5719 ~ 0.0112 | 0.5756 ~ 0.0112 | 0.4939 ~ 0.0077 |
|    0.4     | 0.5588 ~ 0.0129 | 0.5622 ~ 0.0125 | 0.4551 ~ 0.0092 |
|    0.5     | 0.5623 ~ 0.0105 | 0.5673 ~ 0.0102 | 0.4186 ~ 0.0072 |

| Uniform_Noise |    Macro-F1     |    Micro-F1     |    Accuracy     |
|:-------------:|:---------------:|:---------------:|:---------------:|
|      0.0      | 0.6152 ~ 0.0094 | 0.6169 ~ 0.0090 | 0.6000 ~ 0.0058 |
|      0.1      | 0.5978 ~ 0.0078 | 0.6001 ~ 0.0077 | 0.5812 ~ 0.0045 |
|      0.2      | 0.5779 ~ 0.0092 | 0.5805 ~ 0.0097 | 0.5473 ~ 0.0035 |
|      0.3      | 0.5539 ~ 0.0100 | 0.5575 ~ 0.0103 | 0.5045 ~ 0.0054 |
|      0.4      | 0.5266 ~ 0.0120 | 0.5285 ~ 0.0121 | 0.4719 ~ 0.0056 |
|      0.5      | 0.5087 ~ 0.0102 | 0.5129 ~ 0.0110 | 0.4282 ~ 0.0070 |

### 2. SFT paper

- proposes 3 options
  - [F] filter out fluctuating nodes, dropping them before computing loss
  - [L] adaptive loss that penalizes overconfidence
  - [M] strategy from another paper fixmatch to utilize filtered out fluctuating nodes to improve learning
  - implements [F] and [L]

#### 1. effect of [F]

- IMDB
  - filter hyperparams are tweaked for best performance under 0.4 pair noise, then fixed for all experiments
    - memory = 1, warmup = 4
  - F alone can contribute to minor(less than 1%) increase

| Pair_Noise |         Macro-F1         |         Micro-F1         |         Accuracy         |
|:----------:|:------------------------:|:------------------------:|:------------------------:|
|    0.0     | 0.6172(+0.0020) ~ 0.0051 | 0.6192(+0.0023) ~ 0.0051 | 0.6015(+0.0015) ~ 0.0035 |
|    0.1     | 0.6043(+0.0053) ~ 0.0068 | 0.6064(+0.0058) ~ 0.0070 | 0.5809(+0.0030) ~ 0.0089 |
|    0.2     | 0.5909(+0.0035) ~ 0.0103 | 0.5941(+0.0056) ~ 0.0101 | 0.5457(+0.0038) ~ 0.0048 |
|    0.3     | 0.5782(+0.0063) ~ 0.0095 | 0.5827(+0.0071) ~ 0.0092 | 0.5015(+0.0076) ~ 0.0083 |
|    0.4     | 0.5670(+0.0098) ~ 0.0057 | 0.5711(+0.0089) ~ 0.0054 | 0.4562(+0.0013) ~ 0.0089 |
|    0.5     | 0.5680(+0.0057) ~ 0.0090 | 0.5739(+0.0066) ~ 0.0084 | 0.4238(+0.0052) ~ 0.0047 |

| Uniform_Noise |         Macro-F1         |         Micro-F1         |         Accuracy         |
|:-------------:|:------------------------:|:------------------------:|:------------------------:|
|      0.0      | 0.6172(+0.0020) ~ 0.0051 | 0.6192(+0.0023) ~ 0.0051 | 0.6015(+0.0015) ~ 0.0035 |
|      0.1      | 0.6020(+0.0042) ~ 0.0073 | 0.6043(+0.0042) ~ 0.0077 | 0.5870(+0.0058) ~ 0.0071 |
|      0.2      | 0.5842(+0.0063) ~ 0.0082 | 0.5874(+0.0069) ~ 0.0088 | 0.5545(+0.0072) ~ 0.0038 |
|      0.3      | 0.5580(+0.0041) ~ 0.0101 | 0.5625(+0.0050) ~ 0.0102 | 0.5070(+0.0025) ~ 0.0057 |
|      0.4      | 0.5336(+0.0070) ~ 0.0060 | 0.5367(+0.0082) ~ 0.0070 | 0.4770(+0.0051) ~ 0.0060 |
|      0.5      | 0.5089(+0.0002) ~ 0.0100 | 0.5147(+0.0018) ~ 0.0101 | 0.4344(+0.0062) ~ 0.0046 |

#### 2. effect of [F] + [L]

- IMDB
  - filter hyperparams are tweaked for best performance under 0.4 pair noise, then fixed for all experiments
    - memory = 1, warmup = 4, threshold = 0.4, weights = 0.5 0.1
  - still, F + L can contribute to minor(less than 1%) increase

| Pair_Noise |         Macro-F1         |         Micro-F1         |         Accuracy         |
|:----------:|:------------------------:|:------------------------:|:------------------------:|
|    0.0     | 0.6186(+0.0034) ~ 0.0043 | 0.6208(+0.0039) ~ 0.0046 | 0.6045(+0.0045) ~ 0.0040 |
|    0.1     | 0.6037(+0.0047) ~ 0.0072 | 0.6057(+0.0051) ~ 0.0075 | 0.5794(+0.0015) ~ 0.0068 |
|    0.2     | 0.5900(+0.0026) ~ 0.0087 | 0.5935(+0.0038) ~ 0.0085 | 0.5435(+0.0016) ~ 0.0050 |
|    0.3     | 0.5762(+0.0043) ~ 0.0090 | 0.5809(+0.0053) ~ 0.0087 | 0.5020(+0.0081) ~ 0.0082 |
|    0.4     | 0.5676(+0.0088) ~ 0.0057 | 0.5715(+0.0093) ~ 0.0057 | 0.4579(+0.0028) ~ 0.0091 |
|    0.5     | 0.5693(+0.0070) ~ 0.0111 | 0.5753(+0.0080) ~ 0.0101 | 0.4231(+0.0045) ~ 0.0049 |

| Pair_Noise |         Macro-F1         |         Micro-F1         |         Accuracy         |
|:----------:|:------------------------:|:------------------------:|:------------------------:|
|    0.0     | 0.6186(+0.0034) ~ 0.0043 | 0.6208(+0.0039) ~ 0.0046 | 0.6045(+0.0045) ~ 0.0040 |
|    0.1     | 0.6006(+0.0028) ~ 0.0058 | 0.6031(+0.0030) ~ 0.0065 | 0.5844(+0.0032) ~ 0.0052 |
|    0.2     | 0.5805(+0.0026) ~ 0.0071 | 0.5841(+0.0036) ~ 0.0072 | 0.5516(+0.0043) ~ 0.0058 |
|    0.3     | 0.5562(+0.0023) ~ 0.0104 | 0.5609(+0.0034) ~ 0.0104 | 0.5088(+0.0039) ~ 0.0058 |
|    0.4     | 0.5335(+0.0069) ~ 0.0062 | 0.5365(+0.0080) ~ 0.0074 | 0.4745(+0.0026) ~ 0.0047 |
|    0.5     | 0.5116(+0.0029) ~ 0.0114 | 0.5179(+0.0050) ~ 0.0112 | 0.4338(+0.0056) ~ 0.0064 |
