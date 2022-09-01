# -*- coding:utf-8 _*-

"""
Most code for this implementation is borrowed from transformers
"""

from transformers import logging, BartTokenizer
from transformers import BartTokenizerFast


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# See all BART models at https://huggingface.co/models?filter=bart
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "laihuiyuan/mFLAG": "https://huggingface.co/laihuiyuan/mFLAG/resolve/main/vocab.json",
    },
    "merges_file": {
        "laihuiyuan/mFLAG": "https://huggingface.co/laihuiyuan/mFLAG/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "laihuiyuan/mFLAG": "https://huggingface.co/laihuiyuan/mFLAG/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "laihuiyuan/mFLAG": 1024,
}



class MFlagTokenizerFast(BartTokenizerFast):
    r"""
    Construct an mFLAG tokenizer based on BartTokenizerFast. [`MFlagTokenizerFast`] is the
    same as [`BartTokenizerFast`] with the addition of six figurative codes to the vocabulary.
    {50265:<literal>, 50266:<hyperbole>, 50267:<idiom>, 50268:<sarcasm>, 50269:<metaphor>, 50270:<simile>}
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BartTokenizer