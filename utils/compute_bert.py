# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import bert_score
import numpy as np
from transformers import logging


logging.set_verbosity_error()
bertscore = bert_score.BERTScorer(lang='en', rescale_with_baseline=True)

scores = []
with open(sys.argv[1],'r') as fin:
    hyps = []
    for line in fin.readlines():
        hyps.append(line.strip())
with open(sys.argv[2],'r') as fin:
    refs = []
    for line in fin.readlines():
        refs.append(line.strip())

for l1,l2 in zip(hyps, refs):
    p, r, f = bertscore.score([l1], [l2])
    scores.append(round(f[0].tolist(),4))

print('The average bert score is {}'.format(np.mean(scores)))
