## Questions
- exp
  - verify ok:
    - node split 400/400/3257
      - transductive inference, train set ok?
      - clean validation set too large?
      - fixed at multiple runs?
    - node label skew (1197, 745, 1109, 1006)
    - training
      - almost convergent in < 5 epochs, then jitters about
        - patience matters a lot in the long jitter period
        - tweaking down lr delays convergent time, but still jitters
        - 
      - quite sensitive to random seed, can result in ~1% f1 or ~2% accuracy
        - repeat 10 times with difference seed and report average results
  - DBLP too easy?
  - assert: control type cnt == label type cnt

## Todo
- run SFT
- SFT fixmatch to utilize fluctuated
  - converges too fast, fluctuate?
  - ?
- MLC
  - label transition matrix?
- implement MAGNN
  - rnn
- run on IMDB
  - multiple control type