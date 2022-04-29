import argparse
import tg_data
import data_utils
import finetuning_utils_t5
import pandas as pd
import torch
import transformers
from transformers import T5Tokenizer, Trainer, TrainingArguments
from transformers import T5Model
import sklearn
import numpy as np
from sklearn import metrics

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#from ray.tune.suggest.bayesopt import BayesOptSearch

parser = argparse.ArgumentParser(
    description="Finetune an T5 model on the TG dataset."
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory containing the TG dataset. Can be downloaded from https://github.com/yarakyrychenko/tg-misinfo-data.",
)

args = parser.parse_args()

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")


tokenizer = T5Tokenizer.from_pretrained("t5-base")
train_data = tg_data.TGDataset(train_df, tokenizer)
val_data = tg_data.TGDataset(val_df, tokenizer)
test_data = tg_data.TGDataset(test_df, tokenizer)


trainingargs = TrainingArguments(
    output_dir='out',
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


def hp_space(trial):
    from ray import tune
    space = {
        "learning_rate": tune.loguniform(1e-5, 5e-5),
    }
    return space

trainer = Trainer(
    args = trainingargs,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    model_init = finetuning_utils.model_init,
    compute_metrics = finetuning_utils.compute_metrics,
    )

print("START")

#best = trainer.hyperparameter_search(
#    hp_space = hp_space,
#    direction= "minimize",
#    backend="ray",
#    n_trials= 5,
#    search_alg = BayesOptSearch(metric="objective", mode="min"),
#    compute_objective=lambda obj : obj['eval_loss']
#)
#print(best)


print("STARTED TRAINING")
trainer.train(resume_from_checkpoint=True)
print("TRAINING DONE")

trainer.evaluate()

trainer.save_model()
print("done done")    

#Metrics
model_path = "out"
model = T5Model.from_pretrained(model_path, num_labels=3)

predictions = trainer.predict(val_data)
preds = np.argmax(predictions.predictions, axis=-1)


accuracy = metrics.accuracy_score(y_true=predictions.label_ids, y_pred=preds)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true=predictions.label_ids, y_pred=preds, average="macro")

print({'Accuracy': accuracy,
       'f1': f1,
       'precision': precision,
       'recall': recall})

target_names=['Misinformation', 'Russian propaganda', 'Non-propaganda']
report = sklearn.metrics.classification_report(y_pred=preds, y_true=predictions.label_ids,
                                               target_names=target_names)
print(report)