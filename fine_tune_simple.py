import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import os


# Due to a bug within simpletransformers, please disable parallel processing for Linux systems
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

cls_args = ClassificationArgs(num_train_epochs=10,
                              save_model_every_epoch=False,
                              save_eval_checkpoints=False,
                              save_steps=-1,
                              use_multiprocessing=False,
                              use_multiprocessing_for_evaluation=False,
                              sliding_window=True,
                              stride=0.8,
                              max_seq_length=512,
                              loss_type='focal',
                              loss_args={
                                  'alpha': 0.25,  # We might need to tune this param
                                  'gamma': 2,
                                  'reduction': 'mean',
                                  'ignore_index': -100,
                              })

model = ClassificationModel(model_type='bert', model_name='google-bert/bert-base-multilingual-cased', num_labels=2, use_cuda=True, args=cls_args)

train = pd.read_csv('train_balanced.csv')
test = pd.read_csv('test.csv')

train = train.rename(columns={'flag': 'labels'})
test = test.rename(columns={'flag': 'labels'})

model.train_model(train)
model.eval_model(test)

test_entries = test['text'].tolist()
predictions, raw = model.predict(test_entries)

np.save('predictions_mbert_base.npy', np.array(predictions))
np.save('raw_mbert_base.npy', np.array(raw))
