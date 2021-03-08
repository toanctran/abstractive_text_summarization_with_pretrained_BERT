import tensorflow as tf

def scaled_dot_product_attention(q, k,v, mask=None):
    '''
    Calculate the attention weight

    q, k, v must have matchig leadng dimensions.
    k, v mush have matching penultimate dimension, ie.: seq_len_k = seq_len_v

    The mask has different shapes depending on its type (padding or look ahead)
    but it must be broadcastable for addition.

    The mask is multiplied with -1e-9 (close to negative inf)
    This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax.
    The goal is to zero out these cells and large negative inputs to softmax are near zero in the output

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    matmul_qk = tf.cast(matmul_qk, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask != None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) #(..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v) # (..., seq_len_v, depth_v)

    return output, attention_weights

    