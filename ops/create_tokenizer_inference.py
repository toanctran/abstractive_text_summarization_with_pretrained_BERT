import pandas as pd
from transformers import BertTokenizer
from model.abstractive_summarizer_inference import AbstractiveSummarization

from config import config, h_parms


model = AbstractiveSummarization(
                                num_layers=config['NUM_LAYERS'], 
                                d_model=config['D_MODEL'], 
                                num_heads=config['NUM_HEADS'], 
                                dff=config['D_FF'], 
                                vocab_size=config['input_vocab_size'],
                                output_seq_len=config['summ_length'], 
                                rate=h_parms['dropout_rate']
                                )

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def create_dataframe(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    assert len(df.columns) == 2, 'column names should be document and summary'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["Document"].values, df["Summary"].values)