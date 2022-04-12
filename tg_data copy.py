import data_utils
import torch

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
        self.encoded_data = data_utils.encode_data(dataframe, tokenizer, max_seq_length)

        self.label_list = data_utils.extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
    
        input_ids, attention_mask = self.encoded_data
        example = {
          'input_ids': input_ids[i],
          'attention_mask': attention_mask[i],
          'labels': self.label_list[i]
        }
        return example
