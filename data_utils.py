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

    message = dataset['message'].astype(str).values.tolist()

    inputs = tokenizer(
      text= message,
      padding = 'max_length',
      truncation = True,
      max_length = max_seq_length,
      is_split_into_words = False
      )

    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    return input_ids, attention_mask


def extract_labels(dataset):
    """Converts labels into numerical labels.
  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.
  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is Misinformation, 1 is Russian Propaganda and 2 is Real News.
  """
    labels = dataset.label.astype(int).to_list()

    return labels
