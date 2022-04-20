#%% md
# Script for BERT training for multilabel classification
#tutorial source: https://towardsdatascience.com/building-a-multi-label-text-classifier-using-bert-and-tensorflow-f188e0ecdc5d
#github: https://github.com/javaidnabi31/Multi-Label-Text-classification-Using-BERT/blob/master/multi-label-classification-bert.ipynb

import os
import collections
import pandas as pd
import tensorflow as tf

# compatibility module
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow_hub as hub
from datetime import datetime
#%%
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

BERT_DIR = "{}/BERT_uncased".format(os.curdir)
BERT_VOCAB = "{}/vocab.txt".format(BERT_DIR)
BERT_INIT_CHKPNT = "{}/bert_model.ckpt.index".format(BERT_DIR)
BERT_CONFIG = "{}/bert_config.json".format(BERT_DIR)

tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)



tokenizer.tokenize("This here's an example of using the BERT tokenizer")

