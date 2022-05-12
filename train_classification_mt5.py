import torch

def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.
  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.
  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """

    message = dataset['text'].apply(lambda x: 'true or false: ' + x).astype(str).values.tolist()

    inputs = tokenizer(
      text= message,
      padding = 'max_length',
      truncation = True,
      max_length = max_seq_length,
      is_split_into_words = False,
      return_tensors='pt'
      )
    
    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    return input_ids, attention_mask

def extract_labels(dataset, tokenizer):
    """Converts labels into numerical labels.
  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.
  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 1 is Misinformation, 0 is factual. 
  """
    CLASS_TOKENS = ['true','false']

    target = list(dataset.label.apply(lambda x: CLASS_TOKENS[x]).astype(str).values)

    target_encodings = tokenizer(
      text = target,
      padding = 'longest',
      truncation = False,
      is_split_into_words = False,
      return_tensors='pt')

    labels = torch.tensor(target_encodings['input_ids'])
    decoder_attention_mask = torch.tensor(target_encodings['attention_mask'])

    return labels, decoder_attention_mask



    from torch.utils.data import Dataset


class TGDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the BoolQ dataset.
    """

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        """
        Args:
          dataframe: A Pandas dataframe containing the data.
          tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
            tokenize the data.
          max_seq_length: Maximum sequence length to either pad or truncate every
            input example to.
        """
        self.encoded_data = encode_data(dataframe, tokenizer, max_seq_length)

        self.label_list = extract_labels(dataframe, tokenizer)

    def __len__(self):
        label, decoder_attention_mask = self.label_list
        return len(label)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
    
        input_ids, attention_mask = self.encoded_data
        label, decoder_attention_mask = self.label_list
        example = {
          'input_ids': input_ids[i],
          'attention_mask': attention_mask[i],
          'labels': label[i],
          'decoder_attention_mask': decoder_attention_mask[i]
        }

        return example


def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    from sklearn import metrics

    labels = eval_pred.label_ids[:,0]
    preds = np.argmax(eval_pred.predictions[0], axis=2)[:,0]
    

    accuracy = metrics.accuracy_score(y_true=labels, y_pred=preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true=labels, y_pred=preds, average='macro')

    result = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    print("result ", result)
    return result

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    from transformers import MT5Config, MT5ForConditionalGeneration

    configuration = MT5Config()
    model = MT5ForConditionalGeneration(configuration).from_pretrained("google/mt5-base")

    return model


import pandas
from transformers import MT5Tokenizer, Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration
import sklearn
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

train_df = pandas.read_csv("mumin-classifier/data/train_m.csv")
val_df = pandas.read_csv("mumin-classifier/data/val_m.csv")
test_df = pandas.read_csv("mumin-classifier/data/test_m.csv")

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
train_data = TGDataset(train_df, tokenizer)
val_data = TGDataset(val_df, tokenizer)
test_data = TGDataset(test_df, tokenizer)


model_path = "out_mt5"
trainingargs = TrainingArguments(
    output_dir=model_path,
    do_train=True,
    do_eval=True,
    disable_tqdm=False,
    learning_rate=1e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    #logging_steps=500,
    logging_first_step=True,
    #save_steps=1000,
    evaluation_strategy = "epoch"
    )

trainer = Trainer(
    args = trainingargs,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    model_init = model_init,
    compute_metrics = compute_metrics
    ) 


print("STARTED TRAINING")
trainer.train()
print("TRAINING DONE")

trainer.save_model()
print("MODEL SAVED")  

print("\nRESULTS (TEST DATA)\n")

predictions = trainer.predict(test_data)
preds = np.argmax(predictions.predictions[0],axis=2)[:,0]
labels = predictions.label_ids[:,0]

test_scores = f1_score(y_true=labels, y_pred=preds, average=None)
print(f'\nMisinformation F1: {100 * test_scores[1]:.2f}%')
print(f'Factual F1: {100 * test_scores[0]:.2f}%')
print(f'OVERALL macro-average F1: {100 * test_scores.mean():.4f}%\n')

report = sklearn.metrics.classification_report(y_pred=preds,y_true=labels)
print(report)

print("\nLANGUAGES")
langs = pd.concat([val_df, test_df])
langs = langs.query("lang=='en' | lang=='pt' | lang=='es'| lang=='fr'| lang=='ar'")
languages = ['en','pt','es','fr','ar']
f1 = []
for lang in languages:
  print("\nLANG ", lang)
  lang_data = langs.query(f"lang=='{lang}'")
  test_data = tg_data.TGDataset(lang_data, tokenizer)
  predictions = trainer.predict(test_data)
  preds = np.argmax(predictions.predictions, axis=-1)

  test_scores = f1_score(test_data.label_list, preds, average=None)

  print(f'\nMisinformation F1: {100 * test_scores[1]:.2f}%')
  print(f'Factual F1: {100 * test_scores[0]:.2f}%')
  print(f'macro-average F1: {100 * test_scores.mean():.4f}%\n')
  f1.append(100 * test_scores.mean())

print("F1: ", f1)