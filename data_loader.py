
import json
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from functools import partial

from tokenization import tokenizer
from config import config

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103




def pad(list, n, pad=0):
    '''
    Pad the list to have size n
    '''
    pad_width = (0, max(0, n-len(list)))
    return np.pad(list, pad_width, mode='constant', constant_values=pad)

def encode(sent_1, sent_2, tokenizer, input_seq_len, out_seq_len):
    '''
    Encode the text to the BERT expected format

    input_seq_len is used to truncate the article length
    out_seq_len is used to truncate the summary length

    BERT has the following special tokens

    [CLS]: the first token for every sentence. A classification token is
    normally used together with a softmax layer for classification tasks.

    [SEP]: a sequence delimited token, was used at the pre-training for
    sequence-pair tasks (i.e: next sentence prediction). MUST BE USED WHEN SEQUENCE
    PAIR TASKS ARE REQUIRED.

    [MASK]: used for masked words. Only used for pre-training

    Additionally, BERT requires additional inputs to work correctly:
        - Mark IDs
        - Segment IDs

    The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended.
    Sentence Embedding is just a numeric class to distingush between pairs of sentences
    '''

    tokens_1 = tokenizer.tokenize(sent_1.numpy())
    tokens_2 = tokenizer.tokenize(sent_2.numpy())

    # account for [CLS] and [SEP] with '-2'
    if len(tokens_1) > input_seq_len - 2:
        tokens_1 =  tokens_1[0:(input_seq_len - 2)]
    if len(tokens_2) > (out_seq_len +1) - 2:
        tokens_2 = tokens_2[0:(out_seq_len + 1) - 2]

    tokens_1 = ['CLS'] + tokens_1 + ['SEP']
    tokens_2 = ['CLS'] + tokens_2 + ['SEP']

    input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
    input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)

    input_mask_1 = [1] * len(input_ids_1)
    input_mask_2 = [1] * len(input_ids_2)

    input_ids_1 = pad(input_ids_1, input_seq_len, 0)
    input_ids_2 = pad(input_ids_2, out_seq_len + 1, 0)

    input_mask_1 = pad(input_mask_1, input_seq_len, 0)
    input_mask_2 = pad(input_mask_2, out_seq_len + 1, 0)

    input_type_ids_1 = [0] * len(input_ids_1)
    input_type_ids_2 = [0] * len(input_ids_2)

    return input_ids_1, input_mask_1, input_type_ids_1, input_ids_2, input_mask_2, input_type_ids_2

def tf_encode(tokenizer, input_seq_len, output_seq_len):
    def f(s1, s2):
        encode_ = partial(encode, tokenizer= tokenizer, input_seq_len=input_seq_len, output_seq_len=output_seq_len)
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])

    return f

def filter_max_length(x, x1, x2, y, y1, y2, max_length=config['MAX_SAMPLE']):
    predicate = tf.logical_and(
        tf.size(x[0]) <= max_length,
        tf.size(y[0]) <= max_length
    )
    return predicate

def pipeline(samples, tokenizer, cache=False):
    '''
    prepare a dataset to return the following elements
    x_ids, x_mask, x_segments, y_ids, y_mask, y_segments
    '''
    ds = samples.map(tf_encode(tokenizer, config['INPUT_SEQ_LEN'], config['OUTPUT_SEQ_LEN']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if config['MAX_SAMPLE'] != None:
        ds = ds.filter(filter_max_length)

    if cache:
        ds = ds.cache()

    ds = ds.shuffle(config['BUFFER_SIZE']).padded_batch(config['BATCH_SIZE'], padded_shapes=([-1], [-1], [-1], [-1], [-1], [-1]))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

def load_cnn_dailymail_dataset(tokenizer = tokenizer):
    '''
    Load the CNN/Dailymail dataset from Tensorflow Datasets
    '''
    samples, metadata = tfds.load('cnn_dailymail', with_info=True, as_supervised=True)
    train, val, test = samples['train'], samples['validation'], samples['test']
    train_ds = pipeline(train, tokenizer)
    val_ds = pipeline(val, tokenizer)
    test_ds = pipeline(test, tokenizer)

    #grab infor about the number of samples
    metadata = json.loads(metadata.as_json)

    n_test_samples = sum(map(int, metadata['splits'][2]['shardLengths']))
    n_train_samples = sum(map(int,metadata['splits'][0]['shardLengths']))
    n_val_samples = sum(map(int,metadata['splits'][1]['shardLengths']))
    return train_ds, val_ds, test_ds, n_train_samples, n_val_samples, n_test_samples


if __name__ == "__main__":
    train_ds, val_ds, test_ds, n_train_samples, n_val_samples, n_test_samples = load_cnn_dailymail_dataset()
    print('There are %i train samples, %i test samples and %i validation samples in the CNN/Dailymaill dataset' %(n_train_samples, n_test_samples, n_val_samples))



 