import tensorflow as tf

from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

loss_object = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
def convert_idx_to_token_tensor(inputs, tokenizer = tokenizer):
    '''
    Convert int32 tensor to string tensor
    inputs: 1d int32 tensor.indices
    tokenizers :: [int] -> str
    return 1d str tensor
    '''
    def f(inputs):
        return ' '.join(tokenizer.convert_ids_to_tokens(inputs))

    return tf.py_function(f, [inputs], tf.string)

def convert_wordpiece_to_words(w_piece):
    new = []
    for i in w_piece:
        if '##' in i:
            m = i.replace('##', '')
        else:
            if w_piece.index(i) == 0:
                m = i
            else:
                m = ' '+i
        new.append(m)
    return(''.join(new))