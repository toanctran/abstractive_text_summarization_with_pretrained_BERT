# code adapted from a) https://github.com/ShenakhtPajouh/GPT-language-model-tf.keras/blob/master/utils.py
#                  b)https://github.com/raufer/bert-summarization/tree/master/models
from layers.transformer import create_masks
from ops.beam_search import beam_search
from ops.local_tf_ops import *
from ops.create_tokenizer_inference import tokenizer, model
from ops.creates import log, train_summary_writer, valid_summary_writer
from ops.input_path import file_path
from config import config, h_parms
import tensorflow_addons as tfa
import time
import numpy as np
import tensorflow as tf
tf.random.set_seed(100)

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


def with_column(x, i, column):
    """
    Given a tensor `x`, change its i-th column with `column`
    x :: (N, T)
    return :: (N, T)
    """
    left = x[:, :i]
    right = x[:, i+1:]

    return tf.concat([left, column, right], axis=1)


def mask_timestamp(x, i, mask_with):
    """
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.

    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]

    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with

    masked = tf.concat([left, mask, right], axis=1)

    return masked


def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present:
    it outputs a 1 at those locations, and a 0 otherwise.    
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def sampling(logits):
    sample = tf.random.categorical(
        logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample


def top_k_sampling(logits, k=25):
    'k must be greater than 0'
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)
    logits = tf.reshape(logits, (h_parms.batch_size, -1))
    sample = tf.random.categorical(
        logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample


def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(
        tf.expand_dims(indices, 1),
        t_sorted_indices_to_remove[:-1],
        logits.shape
    )
    logits = tf.where(
        sorted_indices_to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )
    logits = tf.reshape(logits, (h_parms.batch_size, -1))
    sample = tf.random.categorical(
        logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample


def topp_topk(logits, p, k):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(
        tf.expand_dims(indices, 1),
        t_sorted_indices_to_remove[:-1],
        logits.shape
    )
    logits = tf.where(
        sorted_indices_to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)
    logits = tf.reshape(logits, (h_parms.batch_size, -1))
    sample = tf.random.categorical(
        logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample
# TODO stop decoding when the SEP_ID is predicted instead of looping till the end of summary len.


def draft_summary_sampling(
    inp,
    enc_output,
    look_ahead_mask,
    padding_mask,
    sampling_type='greedy',
    temperature=0.9,
    p=0.9,
    k=25,
    training=False
):
    """
    Inference call, builds a draft summary auto-regressively
    """
    log.info(f"Building: 'Draft {sampling_type} decoder'")
    N = tf.shape(enc_output)[0]
    T = tf.shape(enc_output)[1]

    # (batch_size, 1)
    dec_input = tf.ones([N, 1], dtype=tf.int32) * CLS_ID
    summary, dec_outputs, dec_logits, attention_dists = [], [], [], []
    summary += [dec_input]
    for i in (range(0, config['summ_length'])):
        _, _, dec_padding_mask = create_masks(inp, dec_input)
        # (batch_size, i+1, d_bert)
        embeddings = model.embedding(dec_input)

        # (batch_size, i+1, vocab), (_)
        dec_output, dec_logits_i, attention_dist = model.decoder(
            embeddings,
            enc_output,
            training,
            look_ahead_mask,
            padding_mask
        )

        if config['COPY_GEN']:
            dec_output = model.decoder.pointer_generator(
                dec_logits_i,
                dec_output,
                attention_dist,
                inp,
                tf.shape(inp)[1],
                tf.shape(dec_output)[1],
                training=False,
            )

        # (batch_size, 1, vocab)
        dec_output_i = dec_output[:, -1:, :]
        if sampling_type == 'nucleus':
            preds = tf.cast(nucleus_sampling(
                ((dec_output_i) / temperature), p=p), tf.int32)
        elif sampling_type == 'topk':
            preds = tf.cast(top_k_sampling(
                ((dec_output_i) / temperature), k=k), tf.int32)
        elif sampling_type == 'random_sampling':
            preds = tf.cast(sampling((dec_output_i) / temperature), tf.int32)
        elif sampling_type == 'topktopp':
            preds = tf.cast(
                topp_topk(((dec_output_i) / temperature), p=p, k=k), tf.int32)
        else:
            preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
        dec_outputs += [dec_output_i]
        dec_logits_i = dec_logits_i[:, -1:, :]
        dec_logits += [dec_logits_i]
        summary += [preds]
        dec_input = with_column(dec_input, i+1, preds)
    summary = tf.concat(summary, axis=1)
    # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
    return summary, attention_dist


def draft_summary_beam_search(input_ids, enc_output, dec_padding_mask, beam_size):

    log.info(f"Building: 'Draft beam search decoder'")
    input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
    enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
    dec_padding_mask = tfa.seq2seq.tile_batch(
        dec_padding_mask, multiplier=beam_size)

    def beam_search_decoder(output):
        # (batch_size, seq_len, d_bert)
        embeddings = model.embedding(output)
        predictions, dec_op, attention_weights = model.decoder(
            embeddings,
            enc_output,
            False,
            None,
            dec_padding_mask
        )
        if config['COPY_GEN']:
            predictions = model.decoder.pointer_generator(
                dec_op[:, -1:, :],
                predictions[:, -1:, :],
                attention_weights[:, :, -1:, :],
                input_ids,
                tf.shape(input_ids)[1],
                tf.shape(predictions[:, -1:, :])[1],
                training=False,
            )
        # (batch_size, 1, target_vocab_size)
        return (predictions[:, -1:, :])
    return beam_search(
        beam_search_decoder,
        [CLS_ID] * h_parms.batch_size,
        beam_size,
        config['summ_length'],
        config['input_vocab_size'],
        h_parms['length_penalty'],
        stop_early=False,
        eos_id=[[SEP_ID]]
    )


def refined_summary_sampling(inp,
                             enc_output,
                             draft_summary,
                             padding_mask,
                             sampling_type='greedy',
                             temperature=0.9,
                             p=0.9,
                             k=25,
                             training=False):
    """
    Inference call, builds a refined summary

    It first masks each word in the summary draft one by one,
    then feeds the draft to BERT to generate context vectors.
    """

    log.info(f"Building: 'Refined {sampling_type} decoder'")
    N = tf.shape(enc_output)[0]
    refined_summary = draft_summary
    batch = tf.shape(draft_summary)[0]
    dec_outputs = []
    dec_logits = []
    attention_dists = []
    for i in (range(1, config['summ_length'])):

        # (batch_size, seq_len)
        refined_summary_ = mask_timestamp(refined_summary, i, MASK_ID)

        # (batch_size, seq_len, d_bert)
        context_vectors = model.bert_model(refined_summary_)[0]

        # (batch_size, seq_len, d_bert), (_)
        dec_output, dec_logits_i, attention_dist = model.decoder(
            context_vectors,
            enc_output,
            training=training,
            look_ahead_mask=None,
            padding_mask=padding_mask
        )

        # (batch_size, 1, vocab_len)
        dec_output_i = dec_output[:, i:i+1, :]
        if sampling_type == 'nucleus':
            preds = tf.cast(nucleus_sampling(
                (dec_output_i / temperature), p=p), tf.int32)
        elif sampling_type == 'topk':
            preds = tf.cast(top_k_sampling(
                ((dec_output_i) / temperature), k=k), tf.int32)
        elif sampling_type == 'topktopp':
            preds = tf.cast(
                topp_topk(((dec_output_i) / temperature), p=p, k=k), tf.int32)
        elif sampling_type == 'random_sampling':
            preds = tf.cast(sampling((dec_output_i) / temperature), tf.int32)
        else:
            preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
        refined_summary = with_column(refined_summary, i, preds)
    # (batch_size, seq_len, vocab_len), (_)
    return refined_summary, attention_dist


def predict_using_sampling(
        inp,
        draft_decoder_sampling_type='topk',
        refine_decoder_sampling_type='topk',
        temperature=0.9,
        p=0.9,
        k=25):

    dec_padding_mask = create_padding_mask(inp)

    # (batch_size, seq_len, d_bert)
    enc_output = model.bert_model(inp)[0]
    # (batch_size, seq_len, vocab_len), (_)
    preds_draft_summary, draft_attention_dist = draft_summary_sampling(
        inp,
        enc_output=enc_output,
        look_ahead_mask=None,
        padding_mask=dec_padding_mask,
        sampling_type=draft_decoder_sampling_type,
        temperature=temperature,
        p=p,
        k=k,
    )
    # (batch_size, seq_len, vocab_len), ()
    preds_refined_summary, refined_attention_dist = refined_summary_sampling(
        inp,
        enc_output=enc_output,
        padding_mask=dec_padding_mask,
        draft_summary=preds_draft_summary,
        sampling_type=refine_decoder_sampling_type,
        temperature=temperature,
        p=p,
        k=k
    )

    return preds_draft_summary, draft_attention_dist, preds_refined_summary, refined_attention_dist


def predict_using_beam_search(
        inp,
        beam_size=3,
        refine_decoder_sampling_type='nucleus',
        temperature=0.9,
        p=0.9,
        k=25):

    dec_padding_mask = create_padding_mask(inp)
    # (batch_size, seq_len, d_bert)
    enc_output = model.bert_model(inp)[0]

    #[batch_size*beam_size, input_Seq_len, d_bert]
    translated_output_temp = draft_summary_beam_search(
        inp, enc_output, dec_padding_mask, beam_size)
    # Take the sequence with high score (the last one)
    preds_draft_summary = translated_output_temp[0][:, 0, :]

    preds_refined_summary, refined_attention_dist = refined_summary_sampling(
        inp,
        enc_output=enc_output,
        padding_mask=dec_padding_mask,
        draft_summary=preds_draft_summary,
        sampling_type=refine_decoder_sampling_type,
        temperature=temperature,
        p=p,
        k=k
    )
    return preds_draft_summary, preds_refined_summary, refined_attention_dist
