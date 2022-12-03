### baseline results

| method | noise |    macro_f1     |    micro_f1     |       acc       |
|:------:|:-----:|:---------------:|:---------------:|:---------------:|
|  HAN   |   0   | 0.9309 ~ 0.0021 | 0.9361 ~ 0.0020 | 0.9313 ~ 0.0069 |
|  HAN   |  0.2  | 0.9253 ~ 0.0043 | 0.9314 ~ 0.0038 | 0.9181 ~ 0.0112 |
|  HAN   |  0.4  | 0.9140 ~ 0.0101 | 0.9207 ~ 0.0096 | 0.8790 ~ 0.0343 |
|  HAN   |  0.5  | 0.8780 ~ 0.0347 | 0.8870 ~ 0.0327 | 0.7934 ~ 0.0770 |
|  HAN   |  0.6  | 0.8600 ~ 0.0470 | 0.8697 ~ 0.0446 | 0.5950 ~ 0.0429 |

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
  - macro_f1 and micro_f1 report 0.8/0.2 svm
- observations
  - HAN is robust against small noise ratio; performance drops dramatically only after ratio > 0.4
  - as label noise ratio grows, variance on performances between different runs also grow

### DBLP

- a subgraph from DBLP
    - nodes
        - 0 author 4057
        - 1 paper 14328
        - 2 term 7723
        - 3 conf 20
        - total 26128
    - edges
        - paper-author
        - paper-term
        - paper-conf
    - features
        - 0 author (4057, 334) sparse
            - directly copy from HAN
        - 1 paper (4231, 14328) sparse
            - lemma tokenizer transfrom
        - 2 term (50, 7723)
            - glove
        - 3 conf (20, 20)
            - id
    - labels
        - only author labels are used
            - 4 labels, 0: Database; 1: Data Mining; 2: AI; 3: Information Retrieval;
            - full 4057 label, 0: 1197; 1: 745; 2: 1109; 3: 1006
            - 400/400/3257 for train/val/test
    - metapath
        - consider 3 metapath schemes
            - author-paper-author 0-1-0
            - author-paper-term-paper-author 0-1-2-1-0
            - author-paper-conf-paper-author 0-1-3-1-0
        - instance collection
            - naively collects paths
            - repetitive and self-connecting
            - adjlist stores (start node, end node) * len(metapath) for each collected metapath
            - idx stores node indices for each collected metapath

### MAGNN

MAGNN implementation bugs

- in parse_minibatch, variable "samples" conflicts with function args

todo:

1. dataset split ratio, split with seed rather than fixed
2. validate clean?