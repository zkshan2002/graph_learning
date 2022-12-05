## Questions
- exp
  - verify ok:
    - node split 0.1/0.1/0.8
      - transductive inference, train set ok?
      - clean validation set too large?
      - fixed at multiple runs?
    - training
      - almost convergent in < 10 epochs, then jitters about
        - patience matters a lot in the long jitter period
        - tweaking down lr delays convergent time, but still jitters
      - quite sensitive to random seed, can result in ~1% f1 or ~2% accuracy
        - repeat 10 times with difference seed and report average results
  - DBLP too easy, exp on IMDB
  - assert: control type cnt == label type cnt

## Todo
- SFT fixmatch to utilize fluctuated
- MLC
  - label transition matrix?
- implement MAGNN
  - rnn
- multiple control type?