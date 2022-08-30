# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
from torch import cuda

from transformers import logging
from transformers import BartTokenizerFast

from utils.dataset import BartIterator
from utils.optim import ScheduledOptim

from model import MultiFigurativeGeneration
from tokenization_mflag import MFlagTokenizerFast

logging.set_verbosity_error()
device = 'cuda' if cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate(model, valid_loader, tokenizer, step):
    """
    Evaluation function for model

    Args:
        model: the BART model.
        valid_loader: pytorch valid DataLoader.
        tokenizer: BART tokenizer
        step: the current training step.

    Returns:
        the average cross-entropy loss
    """

    loss_list = []
    with torch.no_grad():
        model.eval()
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            loss = model(
                input_ids=src,
                attention_mask=mask,
                fig_ids=tgt[:, :1],
                labels=tgt)[0]
            loss_list.append(loss.item())
        model.train()
    avg_loss = np.mean(loss_list)

    print('[Info] valid {:05d} | loss {:.4f}'.format(step, avg_loss))

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-seed', default=42, type=int, help='random seed')
    parser.add_argument(
        '-figs', nargs='+', help='figure tags', required=True)
    parser.add_argument(
        '-batch_size', default=32, type=int, help='batch size')
    parser.add_argument(
        '-patience', default=5, type=int, help='early stopping')
    parser.add_argument(
        '-dataset', default='ParapFG', type=str, help='dataset name')
    parser.add_argument(
        '-lr', default=1e-5, type=float, help='ini. learning rate')
    parser.add_argument(
        '-log_step', default=100, type=int, help='log every x step')
    parser.add_argument(
        '-acc_steps', default=8, type=int, help='accumulation_steps')
    parser.add_argument(
        '-epoch', default=30, type=int, help='force stop at x epoch')
    parser.add_argument(
        '-eval_step', default=1000, type=int, help='eval every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    tokenizer = MFlagTokenizerFast.from_pretrained('checkpoints/mFLAG')
    model = MultiFigurativeGeneration.from_pretrained('checkpoints/mFLAG')
    model = model.to(device).train()

    # load data for training
    train_loader, valid_loader = BartIterator('ft', tokenizer, opt).loader

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09),
        lr=opt.lr, decay_step=1000)

    tab = 0
    step = 0
    avg_loss = 1e9
    loss_list = []
    start = time.time()

    for epoch in range(opt.epoch):
        for batch in train_loader:
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            mask = src.ne(tokenizer.pad_token_id).long()
            loss = model(
                input_ids=src,
                attention_mask=mask,
                fig_ids=tgt[:, :1],
                labels=tgt)[0]
            loss_list.append(loss.item())

            loss = loss / opt.acc_steps
            loss.backward()

            if step % opt.acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] steps {:05d} | loss {:.4f} | lr {:.6f} '
                      '| second {:.2f}'.format(step, np.mean(loss_list),
                                               lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) > opt.eval_step
                 and step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and step % len(train_loader) == 0)):
                eval_loss = evaluate(model, valid_loader, tokenizer, step)
                if avg_loss >= eval_loss:
                    model.save_pretrained('checkpoints/mFLAG')
                    print('[Info] The checkpoint file has been updated.')
                    avg_loss = eval_loss
                    tab = 0
                else:
                    tab += 1
                if tab == opt.patience:
                    exit()


if __name__ == "__main__":
    main()
