### baseline results(pair noise)
| tag    |  method  | noise |    macro_f1     |    micro_f1     |       acc       |
|--------|:--------:|:-----:|:---------------:|:---------------:|:---------------:|
| exp0   |   HAN    |   0   | 0.9345 ~ 0.0025 | 0.9394 ~ 0.0023 | 0.9386 ~ 0.0039 |
| exp1   |   HAN    | 0.1p  | 0.9194 ~ 0.0057 | 0.9254 ~ 0.0051 | 0.9106 ~ 0.0068 |
| exp2   |   HAN    | 0.2p  | 0.9225 ~ 0.0084 | 0.9283 ~ 0.0083 | 0.8915 ~ 0.0284 |
| exp3   |   HAN    | 0.3p  | 0.9222 ~ 0.0078 | 0.9280 ~ 0.0074 | 0.8831 ~ 0.0184 |
| exp4   |   HAN    | 0.4p  | 0.9123 ~ 0.0125 | 0.9188 ~ 0.0113 | 0.7012 ~ 0.0504 |
|        |          |       |                 |                 |                 |

important facts
- note
  - macro_f1 and micro_f1 reports 0.8/0.2 train/test ratio svm test
  - SF uses self filtering and adaptive confidence penalty, no FixMatch yet
- dataset: DBLP only
  - 4 node types(author 4057, paper 14328, term 7723, conf 20, total 26128)
  - 3 metapath schemes: APA, APTPA, APCPA
  - target on node type 0, 4 classes(1197, 745, 1109, 1006)
  - dataset_split: 400/400/3257(0.1/0.1/0.8)
    - is this ok?
  - apply noise on training labels only
- training
  - sampling: each iteration sample 32 nodes, and up to 100 metapaths connected to each node
    - sample number matters. is this ok?
  - hyper params tuned at clean label
  - run 5 times and average results
    - results not stable. is 5 repeat enough?
- observations
  - 

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