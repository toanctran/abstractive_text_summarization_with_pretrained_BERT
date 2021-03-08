import os

core_path = os.getcwd()
dataset_name = 'cnn_dailymail'
vocab_path = os.path.join(core_path, 'input_files', 'vocab_files')
file_path = {
    'best_ckpt_path': os.path.join(core_path, 'created_files', 'training_summarization_model_ckpt', dataset_name, 'best_checkpoints'),
    'ckpt_path': os.path.join(core_path, 'created_files', 'training_summarization_model_ckpt', dataset_name, 'checkpoints'),
    'vocab_path': os.path.join(core_path, 'input_files', 'vocab_files'),
    'infer_csv_path': os.path.join(core_path, "input_files/Azure_dataset/Test.csv"),
    'infer_ckpt_path': os.path.join(core_path, 'created_files', 'training_summarization_model_ckpt', dataset_name, 'infer_checkpoints'),
    'log_path': os.path.join(core_path, "created_files/tensorflow.log"),
    'tensorboard_log': os.path.join(core_path, "created_files/tensorboard_logs/"+dataset_name+"/"),
    'summary_write_path': os.path.join(core_path, 'created_files', 'summaries', dataset_name),
    'subword_vocab_path': os.path.join(core_path, "input_files/vocab_file_summarization_"+dataset_name),
    'train_csv_path': os.path.join(core_path, "input_files/Azure_dataset/Train.csv"),
    'dataset': os.path.join(core_path, 'input_files', dataset_name)
}

