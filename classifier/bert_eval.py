# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

import torch
from torch import cuda
from transformers import logging
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append("")
from classifier.bert_train import FigIterator

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Evaluating Figurative Strength')
    parser.add_argument('-seed', default=42, type=int, help='random seed')
    parser.add_argument('-src_form', default=0, type=str, help='source form')
    parser.add_argument('-tgt_form', default=0, type=str, help='target form')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    config = BertConfig.from_pretrained(
        'bert-base-cased',
        problem_type='single_label_classification',
        num_labels=2)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        config=config)
    model_dir = 'checkpoints/bert_{}.chkpt'.format(
        opt.tgt_form if opt.tgt_form != 'literal' else opt.src_form)
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    src, tgt = [], []
    with open('data/outputs/bart_{}_{}'.format(
            opt.src_form, opt.tgt_form), 'r') as f:
        for line in f.readlines():
            src.append(tokenizer.encode(line.strip()))
            if opt.tgt_form == 'literal':
                tgt.append(0)
            else:
                tgt.append(1)
    print('[Info] {} instances from test set'.format(len(src)))
    test_loader = FigIterator(src, tgt, opt)

    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

    print('[Info] Test: {}'.format('acc {:.2f}% | loss {:.4f}').format(
        corre_num / total_num * 100, np.mean(loss_list)))


if __name__ == '__main__':
    main()
