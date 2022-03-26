# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np

from comet.models import download_model

model = download_model("wmt-large-da-estimator-1719")


src = []
out = []
ref = []
with open(sys.argv[1],'r') as fin:
    for line in fin.readlines():
        src.append("")
        #src.append(line.strip())

with open(sys.argv[2],'r') as fin:
    for line in fin.readlines():
        out.append(line.strip())

with open(sys.argv[3],'r') as fin:
    for line in fin.readlines():
        ref.append(line.strip())

data = {"src": src, "mt": out, "ref": ref}
data = [dict(zip(data, t)) for t in zip(*data.values())]

scores=model.predict(data, cuda=True, show_progress=False)[-1]

print('The average comet score is {}'.format(np.mean(scores)))
