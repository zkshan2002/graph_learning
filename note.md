### baseline results

| method | noise |    macro_f1     |    micro_f1     |       acc       |
|:------:|:-----:|:---------------:|:---------------:|:---------------:|
|  HAN   |   0   | 0.9309 ~ 0.0021 | 0.9361 ~ 0.0020 | 0.9313 ~ 0.0069 |
|  HAN   |  0.2  | 0.9253 ~ 0.0043 | 0.9314 ~ 0.0038 | 0.9181 ~ 0.0112 |
|  HAN   |  0.4  | 0.9140 ~ 0.0101 | 0.9207 ~ 0.0096 | 0.8790 ~ 0.0343 |
|  HAN   |  0.5  | 0.8780 ~ 0.0347 | 0.8870 ~ 0.0327 | 0.7934 ~ 0.0770 |
|  HAN   |  0.6  | 0.8600 ~ 0.0470 | 0.8697 ~ 0.0446 | 0.5950 ~ 0.0429 |
|   -    |   -   |        -        |        -        |        -        |
|  exp0  |  0.6  | 0.8914 ~ 0.0440 | 0.8993 ~ 0.0410 | 0.5789 ~ 0.1315 |
|  exp1  |  0.6  | 0.8965 ~ 0.0411 | 0.9046 ~ 0.0375 | 0.5185 ~ 0.1630 |
|  exp2  |  0.6  | 0.8581 ~ 0.0584 | 0.8680 ~ 0.0559 | 0.5576 ~ 0.1153 |
|  exp3  |  0.4  | 0.9119 ~ 0.0268 | 0.9190 ~ 0.0246 | 0.7858 ~ 0.1078 |
|  exp4  |  0.4  | 0.9248 ~ 0.0049 | 0.9307 ~ 0.0047 | 0.8666 ~ 0.0357 |

exp0: (0.2, 0.5, 0.1)
exp1: (0.2, 0.5, 0.05)
exp2: (0.1, 0.5, 0.1)
exp3: (0.2, 0.5, 0.1)
exp4: (0.2, 0.5, 0.05)

SFT largely increases accuracy variance and decreases accuracy
but both f1 scores are increased. why?
better uses small weight, e.g. exp4 > exp3, exp1 > exp0
but not 0
currently fluctuated are dropped. this is bad when noise ratio is large(fluctuated ratio is large)
consider fixmatch

note
- macro_f1 and micro_f1 reports 0.8/0.2 train/test ratio svm test
- SF uses self filtering and adaptive confidence penalty, no FixMatch yet



important facts
- dataset: DBLP only
  - 4 node types(author 4057, paper 14328, term 7723, conf 20, total 26128)
  - 3 metapath schemes: APA, APTPA, APCPA
  - target on node type 0, 4 classes(1197, 745, 1109, 1006)
  - dataset_split: 400/400/3257(0.1/0.1/0.8)
  - apply uniform noise on training labels
- training
  - sampling: each iteration sample 128 nodes, and up to 100 metapaths connected to each node
  - hyper params tuned at clean label
  - run 5 times and average results
- observations
  - HAN is robust against small noise ratio; performance drops dramatically only after ratio > 0.4
  - as label noise ratio grows, variance on performances between different runs also grow

### DBLP

- a subgraph from DBLP
    - nodes: 0 author 4057, 1 paper 14328, 2 term 7723, 3 conf 20, total 26128
    - edges: PA, PT, PC(how many each?)
    - features
        - 0 author (4057, 334) sparse
            - directly copy from HAN
        - 1 paper (4231, 14328) sparse
            - lemma tokenizer transform
        - 2 term (50, 7723)
            - glove
        - 3 conf (20, 20)
            - id
    - labels
        - only A has labels
          - semantic: 0: Database; 1: Data Mining; 2: AI; 3: Information Retrieval;
          - 0 1197, 1 745, 2 1109, 3 1006, no large skew
          - train/val/test split: fixed 400/400/3257
    - metapath
        - 3 schemes: 0-1-0 APA, 0-1-2-1-0 APTPA, 0-1-3-1-0 APCPA

### MAGNN

MAGNN implementation bugs

- in parse_minibatch, variable "samples" conflicts with function args

todo:

1. dataset split ratio, split with seed rather than fixed
2. validate clean?