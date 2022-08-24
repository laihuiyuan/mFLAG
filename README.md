# Multi-Figurative Language Generation (COLING 2022) 

## Overview

![](./figs/overview.png)

## Quick Start
### Step 1: Pre-training
```bash
python train_pt.py -dataset parap-fig -figs hyperbole idiom irony metaphor simile
```

### Step 2: Fine-tuning
```bash
# parallel paraphrase pretraining data
python train_ft.py -dataset parap-fig -figs hyperbole idiom irony metaphor simile

# literal-figurative parallel data
python train_ft.py -dataset multi-fig -figs hyperbole idiom irony metaphor simile
```

### Step 3: Figurative Generation
```bash
# Generating idioms form hyperbolic text
python inference.py -src_form hyperbole -tgt_form idiom
```

### Models and Outputs
- All model can found [here](https://drive.google.com/drive/folders/1s8Q_IBzmvcVlDp_Zaln3YX3npq_Vrsia?usp=sharing), and their corresponding outputs can be found in the `/data/outputs/` directory

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
