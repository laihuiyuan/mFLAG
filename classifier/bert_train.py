# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np

import torch
from torch import cuda
from transformers import logging
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append("")
from utils.optim import ScheduledOptim

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def read_insts(mode, opt):
    src, tgt = [], []
    for i in range(2):
        dir = 'data/multi-fig/{}_{}.{}'.format(mode, opt.fig, str(i))
        with open(dir, 'r') as f:
            for line in f.readlines():
                s = tokenizer.encode(line.strip())
                src.append(s)
                tgt.append(i)

    return src, tgt


def collate_fn(insts):
    """
    Pad the instance to the max sequence length in batch.
    """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [tokenizer.pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


class FigDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def FigIterator(insts, labels, opt, shuffle=True):
    """
    Data iterator for classifier
    Args:
        insts_neg (list): negative instances
        insts_pos (list): positive instances
    Returns:
        Pytorch DataLoader
    """

    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts)
        labels = torch.LongTensor(labels)

        return (seq, labels)

    loader = torch.utils.data.DataLoader(
        FigDataset(
            insts=insts,
            label=labels),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def evaluate(model, valid_loader, epoch, tokenizer):
    '''Evaluation function for classifier'''
    model.eval()
    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
        epoch, 'acc {:.4f} | loss {:.4f}').format(
        corre_num / total_num, np.mean(loss_list)))

    return corre_num / total_num, np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('Figurative Classifier')
    parser.add_argument('-seed', default=42, type=int, help='random seed')
    parser.add_argument('-fig', default='0', type=str, help='figurative form')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-lr', default=1e-5, type=float, help='ini. learning rate')
    parser.add_argument('-log_step', default=100, type=int, help='log every x step')
    parser.add_argument('-epoch', default=80, type=int, help='force stop at x epoch')
    parser.add_argument('-eval_step', default=1000, type=int, help='eval every x step')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    config = BertConfig.from_pretrained(
        'bert-base-cased',
        problem_type='single_label_classification',
        num_labels=2)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        config=config)
    model.to(device).train()

    print('[Info] Built a model with {} parameters'.format(
        sum(p.numel() for p in model.parameters())))

    # read instances from input file
    train_src, train_tgt = read_insts('train', opt)
    valid_src, valid_tgt = read_insts('valid', opt)
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_src)))

    train_loader = FigIterator(train_src, train_tgt, opt)
    valid_loader = FigIterator(valid_src, valid_tgt, opt)

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad,
                                model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09),
        lr=opt.lr, decay_step=1000)

    tab = 0
    eval_acc = 0
    corre_num = 0.
    total_num = 0.
    loss_list = []
    start = time.time()
    for epoch in range(opt.epoch):
        for batch in train_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

            if optimizer.steps % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | acc {:.4f} | '
                      'loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    epoch, optimizer.steps, corre_num / total_num,
                    np.mean(loss_list), lr, time.time() - start))
                corre_num = 0.
                total_num = 0.
                loss_list = []
                start = time.time()

            if ((len(train_loader) >= opt.eval_step
                 and optimizer.steps % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and optimizer.steps % len(train_loader) == 0)):
                valid_acc, valid_loss = evaluate(model, valid_loader, epoch, tokenizer)
                if eval_acc < valid_acc:
                    eval_acc = valid_acc
                    save_path = 'checkpoints/bert_{}.chkpt'.format(opt.fig)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 5:
                        exit()


if __name__ == '__main__':
    main()
