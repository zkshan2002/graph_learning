## 0.DBLP

- Node

| Type ID | Semantic  | Count | Feature Dim |      Feature Source       |
|:-------:|:---------:|------:|------------:|:-------------------------:|
|    0    | author(A) |  4057 |         334 |       HAN paper(?)        |
|    1    | paper(P)  | 14328 |        4231 | lemma tokenizer transform |
|    2    |  term(T)  |  7723 |          50 |           glove           |
|    3    |  conf(C)  |    20 |          20 |            id             |

- Edge
    - no feature

| Type ID | Semantic | Total |
|:-------:|:--------:|:-----:|
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

| Class ID |       Semantic        | Total |
|:--------:|:---------------------:|------:|
|    0     |       Database        |  1197 |
|    1     |      Data Mining      |   745 |
|    2     |          AI           |  1109 |
|    3     | Information Retrieval |  1006 |
