# -*- coding: utf-8 -*-

import torch
import argparse
from torch import cuda
from transformers import BartTokenizerFast

from model import MultiFigurativeGeneration
from tokenization_mflag import MFlagTokenizerFast

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-bs', default=64, type=int, help='the batch size')
    parser.add_argument(
        '-src_form', default=0, type=str, help='source form')
    parser.add_argument(
        '-tgt_form', default=0, type=str, help='target form')
    parser.add_argument(
        '-nb', default=5, type=int, help='beam search number')
    parser.add_argument(
        '-seed', default=42, type=int, help='the random seed')
    parser.add_argument(
        '-length', default=60, type=int, help='the max length')
    parser.add_argument(
        '-dataset', default='0', type=str, help='dataset name')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = MFlagTokenizerFast.from_pretrained('checkpoints/mFLAG')
    fig_id = torch.tensor(tokenizer.encode('<{}>'.format(opt.tgt_form),
                         add_special_tokens=False)).to(device)
    model = MultiFigurativeGeneration.from_pretrained('checkpoints/mFLAG')
    model.to(device).eval()

    src_seq = []
    if opt.src_form != 'literal':
        inp_dir = 'data/{}/test_{}.1'.format(opt.dataset, opt.src_form)
    else:
        inp_dir = 'data/{}/test_{}.0'.format(opt.dataset, opt.tgt_form)
    with open(inp_dir, 'r') as fin:
        for line in fin.readlines():
            src_seq.append('<{}>'.format(opt.src_form) + line.strip())

    with open('./data/outputs/bart_{}_{}'.format(
            opt.src_form, opt.tgt_form), 'w') as fout:
        for idx in range(0, len(src_seq), opt.bs):
            inp = tokenizer.batch_encode_plus(src_seq[idx: idx + opt.bs],
                                              padding=True, return_tensors='pt')
            src = inp['input_ids'].to(device)[:, 1:]
            mask = inp['attention_mask'].to(device)[:, 1:]
            decoder_input_ids = fig_id.expand((src.size(0), len(fig_id)))
            outs = model.generate(
                input_ids=src,
                attention_mask=mask,
                num_beams=opt.nb,
                fig_ids=decoder_input_ids,
                max_length=opt.length,
                forced_bos_token_id=fig_id.item())
            for x, y in zip(outs, src_seq[idx:idx + opt.bs]):
                text = tokenizer.decode(
                    x.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                try:
                    text = text.split('<{}>'.format(opt.tgt_form))[-1]
                except:
                    text = text

                if len(text.strip()) == 0:
                    text = y

                fout.write(text.strip() + '\n')


if __name__ == "__main__":
    main()
