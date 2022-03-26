# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
from torch import cuda
import torch.nn.functional as F
from transformers import logging
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append("")

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Score pairs of Paraphrasing')
    parser.add_argument('-seed', default=42, type=int, help='random seed')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-fig', default=0, type=str, help='the figurative form')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    config = BertConfig.from_pretrained(
        'bert-base-cased',
        problem_type='single_label_classification',
        num_labels=2)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        config=config)
    model_dir = 'checkpoints/bert_{}.chkpt'.format(opt.fig)
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    fi = open('data/para/parabank2', 'r').readlines()
    fo = open('data/para/para_{}'.format(opt.fig), 'w')
    with torch.no_grad():
        for idx in range(0, len(fi), opt.batch_size):
            sentences = [line.strip() for line in fi[idx: idx + opt.batch_size]]
            inp = tokenizer.batch_encode_plus(
                sentences,
                padding=True,
                return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model(src, mask)
            logits = outs.logits
            logits = F.softmax(logits, dim=-1)
            for line, item in zip(sentences, logits.cpu().tolist()):
                line = line.strip() + '\t' + str(round(item[1], 4))
                fo.write(line + '\n')


if __name__ == '__main__':
    main()
