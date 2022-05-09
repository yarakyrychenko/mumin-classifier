import argparse
import tg_data
import data_utils
import finetuning_utils_mt5
import pandas as pd
import torch
import transformers
from transformers import MT5Tokenizer, Trainer, TrainingArguments
from transformers import MT5Model
import sklearn
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

train_df = pd.read_csv("data/train_m.csv")
val_df = pd.read_csv("data/val_m.csv")
test_df = pd.read_csv("data/test_m.csv")

tokenizer = MT5Tokenizer.from_pretrained("mt5-base")
train_data = tg_data.TGDataset(train_df, tokenizer)
val_data = tg_data.TGDataset(val_df, tokenizer)
test_data = tg_data.TGDataset(test_df, tokenizer)

model_path = "out_mt5"
trainingargs = TrainingArguments(
    output_dir=model_path,
    do_train=True,
    do_eval=True,
    disable_tqdm=False,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch"
    )


trainer = Trainer(
    args = trainingargs,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    model_init = finetuning_utils_mt5.model_init,
    compute_metrics = finetuning_utils_mt5.compute_metrics,
    )


print("STARTED TRAINING")
trainer.train(resume_from_checkpoint=True)
print("TRAINING DONE")

#trainer.evaluate()

trainer.save_model()
print("MODEL SAVED")    

#Metrics
model = MT5Model.from_pretrained(model_path, num_labels=2)

predictions = trainer.predict(test_data)
preds = np.argmax(predictions.predictions, axis=-1)


accuracy = metrics.accuracy_score(y_true=predictions.label_ids, y_pred=preds)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true=predictions.label_ids, y_pred=preds, average="macro")

print({'accuracy': accuracy,
       'f1': f1,
       'precision': precision,
       'recall': recall})

test_scores = f1_score(y_test, test_preds, average=None)
print(f'macro-average F1: {100 * test_scores.mean():.4f}%')


report = sklearn.metrics.classification_report(y_pred=preds, y_true=predictions.label_ids)

print(report)
