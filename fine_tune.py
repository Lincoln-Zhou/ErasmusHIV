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

os.path.isdir('./output') or os.makedirs('./output')  # Check if output dir exists, if not create it
os.path.isdir('./final') or os.makedirs('./final')


# Wrapper for training
def run(args: ClassificationArgs) -> float:
    model.args = args

    seed_list = list(range(24, 35))
    results = []

    for seed in seed_list:
        train_df, test_df = train_test_split(dataset, test_size=0.2, stratify=dataset['labels'], random_state=seed)

        model.train_model(train_df, output_dir='./output')

        train_result, _, _ = model.eval_model(train_df)
        test_result, _, _ = model.eval_model(test_df)

        results.append((train_result, test_result))

    return np.mean([x[1]['mcc'] for x in results])

    # with open(f'./output/results_{int(time.time())}.pkl', 'wb') as file:
    #    pickle.dump(results, file)


# Create args for param search
candidate_alphas = [0.15, 0.25, 0.5, 0.75, 0.9, None, 'cel']
candidate_args = []

for candidate_alpha in candidate_alphas:
    cfg = default_model_args()

    if isinstance(candidate_alpha, str):    # Default to cross-entropy loss
        cfg.loss_args = dict()
        cfg.loss_type = None
    else:
        cfg.loss_args['alpha'] = candidate_alpha

    candidate_args.append(cfg)

# Execute the runs, find the arg that yields highest Matthews Correlation Coefficient
best_arg = max(candidate_args, key=run)

# Retrain with the same arg using full data
model.args = best_arg

model.train_model(dataset, output_dir='./final')

print(model.eval_model(dataset, output_dir='./final')[0])
