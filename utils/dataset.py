# -*- coding: utf-8 -*-

import random
random.seed(42)

import torch
import torch.utils.data


def word_mask(seq, seq_len, mask_prob=0.3, mask_id=0):
    if mask_prob == 0:
        return seq

    noise = torch.rand(seq.size(), dtype=torch.float).to(seq.device)
    pos_idx = torch.arange(seq.size(1)).expand_as(seq).to(seq.device)
    token_mask = (0<pos_idx) & (pos_idx < seq_len.unsqueeze(1)-1)
    drop_mask = (noise < mask_prob) & token_mask
    
    x = seq.clone()
    x.masked_fill_(drop_mask, mask_id)

    return x


class BartDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst):

        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


class BartIterator(object):
    """ Data iterator for fine-tuning BART """

    def __init__(self, task, tokenizer, opt):

        self.task = task
        self.tokenizer = tokenizer
        self.opt = opt

        self.train_src, self.train_tgt = self.read_insts('train')
        self.valid_src, self.valid_tgt = self.read_insts('valid')
        print('[Info] {} instances from train set'.format(len(self.train_src)))
        print('[Info] {} instances from valid set'.format(len(self.valid_src)))

        self.loader = self.gen_loader(self.train_src, self.train_tgt, 
                                      self.valid_src, self.valid_tgt)

    def read_insts(self, mode):
        """
        Read instances from input file
        Args:
            mode (str): 'train' or 'valid'.
        Returns:
            src_seq: list of the lists of token ids for each source sentence.
            tgt_seq: list of the lists of token ids for each tgrget sentence.
        """
        
        src, tgt = [], []
        for f in self.opt.figs:
            src_seq, tgt_seq = [], []

            src_dir = 'data/{}/{}_{}.0'.format(self.opt.dataset, mode, f)
            tgt_dir = 'data/{}/{}_{}.1'.format(self.opt.dataset, mode, f)

            with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2:
                f1 = f1.readlines()
                f2 = f2.readlines()
                prefix_0 = '<literal>'
                prefix_1 = '<{}>'.format(f)
                for i in range(min(len(f1), len(f2))):
                    s = self.tokenizer.encode(prefix_0+f1[i].strip())
                    t = self.tokenizer.encode(prefix_1+f2[i].strip())
                    src_seq.append(s[1:min(len(s)-1, 150)]+s[-1:])
                    tgt_seq.append(t[1:min(len(t)-1, 150)]+t[-1:])

                if len(tgt_seq)<20000 and mode!='valid':
                    ups = int(20000/len(tgt_seq))
                else:
                    ups = 1

                if self.task=='pt':
                    random.shuffle(src_seq)
                    tgt.extend(tgt_seq*ups)
                    tgt.extend(src_seq[:50000]*ups)
                
                else:
                    src.extend(src_seq*ups)
                    tgt.extend(tgt_seq*ups)
                    src.extend(tgt_seq[:50000]*ups)
                    tgt.extend(src_seq[:50000]*ups)

        if self.task=='pt':
            return tgt, tgt.copy()
        else:
            return src, tgt


    def gen_loader(self, train_src, train_tgt, valid_src, valid_tgt):
        """Generate pytorch DataLoader."""

        train_loader = torch.utils.data.DataLoader(
            BartDataset(
                src_inst=train_src,
                tgt_inst=train_tgt),
            num_workers=2,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn,
            shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            BartDataset(
                src_inst=valid_src,
                tgt_inst=valid_tgt),
            num_workers=2,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn)

        return train_loader, valid_loader


    def collate_fn(self, insts):
        """Pad the instance to the max seq length in batch"""

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(inst) for inst in insts)

        batch_seq = [inst + [pad_id]*(max_len - len(inst))
                     for inst in insts]
        batch_seq = torch.LongTensor(batch_seq)

        return batch_seq


    def paired_collate_fn(self, insts):
        src_inst, tgt_inst = list(zip(*insts))
        src_inst = self.collate_fn(src_inst)
        tgt_inst = self.collate_fn(tgt_inst)

        return src_inst, tgt_inst


