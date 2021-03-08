

config = {
    'NUM_EPOCHS': 4,    
    "BATCH_SIZE": 2,
    "GRADIENT_ACCUMULATION_N_STEPS": 18,
    "BUFFER_SIZE": 1,
    "INITIAL_LR": 0.0003,
    "WARMUP_STEPS": 4000,
    "INPUT_SEQ_LEN": 512,
    "OUTPUT_SEQ_LEN": 72,
    "MAX_SAMPLE": 0,
    "VOCAB_SIZE": 30522,    
    "NUM_LAYERS": 8,
    "D_MODEL": 768,
    "D_FF": 2048,    
    "NUM_HEADS": 8,
    "DROPOUT_RATE": 0.1,
    "COPY_GEN":True,
    'copy_gen':True,
    'create_hist':False,             # create histogram of summary length and # of tokens per batch
    'doc_length': 512,
    'd_model': 768,                  # the projected word vector dimension
    'dff': 2048,                      # feed forward network hidden parameters
    'early_stop' : False,
    'eval_after' : 5000,              #Run evaluation after this many samples are trained 
    'init_tolerance':0,
    'input_vocab_size': 30522,        # total vocab size + start and end token
    'last_recorded_value': None,
    'monitor_metric' : 'combined_metric',
    'monitor_only_after': 1,        # monitor the monitor_metric only after this epoch                                           
    'max_tokens_per_line' : 1763,   # 1763 = 90 percentile  
    'num_examples_to_train': None,   #If None then all the examples in the dataset will be used to train
    'num_examples_to_infer': None,
    'num_heads': 8,                  # the number of heads in the multi-headed attention unit
    'num_layers': 8,                 # number of transformer blocks
    'print_chks': 50,                # print training progress per number of batches specified
    'pretrained_bert_model': 'bert-base-uncased',
    'run_tensorboard': True,
    'show_detokenized_samples' : False,
    'summ_length': 72,
    'start_from_batch':353556,
    'target_vocab_size': 30522,       # total vocab size + start and end token
    'test_size': 0.05,               # test set split size
    'tfds_name' : 'cnn_dailymail',   # tfds dataset to be used
    'tolerance_threshold': 5,        # tolerance counter used for early stopping
    'use_tfds' : True,               # use tfds datasets as input to the model 
    'valid_samples_to_eval' : 100,
    'write_per_step': 5000,            # write summary for every specified epoch
    'write_summary_op': True         # write validation summary to hardisk
}


h_parms = {
	 'accumulation_steps': 36,                                                                                   
	 'batch_size': 1,
	 'beam_sizes': [2, 3, 4],        	     # Used only during inference                                                 
	 'combined_metric_weights': [0.4, 0.3, 0.3], #(bert_score, rouge, validation accuracy)
	 'dropout_rate': 0.0,
	 'epochs': 4,
	 'epsilon_ls': 0.0,              	     # label_smoothing hyper parameter
	 'grad_clipnorm':None,
	 'l2_norm':0,
	 'learning_rate': None,          	     # change to None to set learning rate decay
	 'length_penalty' : 1,                       # Beam search hyps . Used only during inference                                                 
	 'mean_attention_heads':True,                # if False then the attention parameters of the last head will be used
	 'mean_parameters_of_layers':True,           # if False then the attention parameters of the last layer will be used
	 'validation_batch_size' : 8
	 }  

