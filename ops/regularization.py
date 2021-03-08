import tensorflow as tf
from torch._C import dtype

def label_smoothing(inputs, epsilon=0.1):
    '''
    Apply label smoothing as https://arxiv.org/abs/1512.00567
    iputs: 3d tensor [N,T,V] where V is the number of vocabulary
    epsilon: smoothing rate
    '''
    V = inputs.get_shape().as_list()[-1]
    V = tf.cast(V, dtype=inputs.dtype)
    return ((1-epsilon)*inputs)+(epsilon/V)