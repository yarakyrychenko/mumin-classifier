import argparse
import tg_data
import data_utils
import finetuning_utils
import pandas as pd
import torch


from transformers import RobertaTokenizer, Trainer, TrainingArguments
#from ray.tune.suggest.bayesopt import BayesOptSearch

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    description="Finetune an RoBERTa model on the TG dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the TG dataset. Can be downloaded from https://github.com/yarakyrychenko/tg-misinfo-data.",
)

args = parser.parse_args()

train_df = pd.read_csv(f"{args.data_dir}/train_en.csv")
val_df = pd.read_csv(f"{args.data_dir}/val_en.csv")
test_df = pd.read_csv(f"{args.data_dir}/test_en.csv")


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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
    model_init = finetuning_utils_roberta.model_init,
    compute_metrics = finetuning_utils_roberta.compute_metrics,
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
trainer.train()
print("TRAINING DONE")

trainer.evaluate()

trainer.save_model()
print("done done")  
