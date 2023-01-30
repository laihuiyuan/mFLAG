# [Multi-Figurative Language Generation (COLING 2022)](https://arxiv.org/abs/2209.01835)

## Overview

![](./figs/overview.png)

## Quick Start
### How to use
```bash
git clone git@github.com:laihuiyuan/mFLAG.git
cd mFLAG
```

```python
from model import MultiFigurativeGeneration
from tokenization_mflag import MFlagTokenizerFast
tokenizer = MFlagTokenizerFast.from_pretrained('laihuiyuan/mFLAG')
model = MultiFigurativeGeneration.from_pretrained('laihuiyuan/mFLAG')


# an example for hyperbole-to-sarcasm generation
# a token (<hyperbole>) is added at the beginning of the source sentence to indicate its figure of speech
inp_ids = tokenizer.encode("<hyperbole> I am not happy that he urged me to finish all the hardest tasks in the world", return_tensors="pt")
# the target figurative form (<sarcasm>)
fig_ids = tokenizer.encode("<sarcasm>", add_special_tokens=False, return_tensors="pt")
outs = model.generate(input_ids=inp_ids[:, 1:], fig_ids=fig_ids, forced_bos_token_id=fig_ids.item(), num_beams=5, max_length=60,)
text = tokenizer.decode(outs[0, 2:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
# special tokens: <literal>, <hyperbole>, <idiom>, <sarcasm>, <metaphor>, or <simile>
```

### Training

#### Step 1: Pre-training
```bash
python train_pt.py -dataset ParapFG -figs hyperbole idiom metaphor sarcasm simile
```

#### Step 2: Fine-tuning
```bash
# parallel paraphrase pretraining data
python train_ft.py -dataset ParapFG -figs hyperbole idiom metaphor sarcasm simile

# literal-figurative parallel data
python train_ft.py -dataset MultiFG -figs hyperbole idiom metaphor sarcasm simile
```

#### Step 3: Figurative Generation
```bash
# Generating idioms form hyperbolic text
python inference.py -src_form hyperbole -tgt_form idiom
```

#### Model and Outputs
- Our model **mFLAG** can be found in [Hugging Face](https://huggingface.co/laihuiyuan/mFLAG), the corresponding outputs are in the `/data/outputs/` directory

## Citation
```
@inproceedings{lai-etal-2022-multi,
    title = "Multi-Figurative Language Generation",
    author = "Lai, Huiyuan and Nissim, Malvina",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = October,
    year = "2022",
    address = "Gyeongju, Republic of korea",
}
```
