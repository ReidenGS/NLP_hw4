import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        self.encoder_input_ids = []
        self.decoder_target_ids = []
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        self.encoder_input_ids = [
            torch.tensor(tokenizer.encode(f"translate to SQL: {x}", add_special_tokens=True), dtype=torch.long)
            for x in nl_lines
        ]

        if split != "test":
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            self.decoder_target_ids = [
                torch.tensor(tokenizer.encode(y, add_special_tokens=True), dtype=torch.long)
                for y in sql_lines
            ]
    
    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, idx):
        encoder_ids = self.encoder_input_ids[idx]
        initial_decoder_input = torch.tensor([self.decoder_start_token_id], dtype=torch.long)

        if self.split == "test":
            return encoder_ids, initial_decoder_input

        decoder_targets = self.decoder_target_ids[idx]
        return encoder_ids, decoder_targets, initial_decoder_input

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [x[0] for x in batch]
    decoder_targets_list = [x[1] for x in batch]
    initial_decoder_inputs_list = [x[2] for x in batch]

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs_list = []
    for initial_decoder_input, decoder_targets in zip(initial_decoder_inputs_list, decoder_targets_list):
        decoder_inputs_list.append(torch.cat([initial_decoder_input, decoder_targets[:-1]], dim=0))

    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [x[0] for x in batch]
    initial_decoder_inputs_list = [x[1] for x in batch]

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
