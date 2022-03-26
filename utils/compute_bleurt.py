# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from bleurt import score
import numpy as np


checkpoint = 'checkpoints/bleurt-large-512'
scorer = score.BleurtScorer(checkpoint)

scores = []
with open(sys.argv[1],'r') as fin:
    hyps = []
    for line in fin.readlines():
        hyps.append(line.strip())
with open(sys.argv[2],'r') as fin:
    refs = []
    for line in fin.readlines():
        refs.append(line.strip())

scores = scorer.score(refs, hyps)

print('The average bleurt score is {}'.format(np.mean(scores)))
