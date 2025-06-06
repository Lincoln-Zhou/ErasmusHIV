import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
import time


# RoBERTa supports up to 512 token length, lower the max_seq_length value if we encounter VRAM overflow issues during training
def default_model_args():
    return ClassificationArgs(num_train_epochs=10,
                              save_model_every_epoch=False,
                              save_eval_checkpoints=False,
                              save_steps=-1,
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


model = ClassificationModel(model_type='roberta', model_name='./medroberta', args=default_model_args)

dataset = pd.read_csv('dataset.csv')


# Wrapper for training
def run(args: ClassificationArgs = default_model_args()) -> float:
    model.args = args

    train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['labels'], random_state=24)

    model.train_model(train_df, output_dir='./output')

    train_result, _, _ = model.eval_model(train_df)
    test_result, _, _ = model.eval_model(test_df)

    print(train_result)
    print(test_result)

    # with open(f'./output/results_{int(time.time())}.pkl', 'wb') as file:
    #    pickle.dump(results, file)


run()
