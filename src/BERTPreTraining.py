import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import trange
import random
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import AdamW
import contractions
import os
import pyarrow.parquet as pq
import re
import time
import gc
from tqdm.notebook import tqdm
from itertools import filterfalse
from tqdm import trange
from transformers import AutoTokenizer, BertForPreTraining,BertForMaskedLM, AutoModel
from transformers import AutoTokenizer, RobertaForMaskedLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ENTER data_path TO DATA FILE WITH CURRICULAR DATA
data_path = "PATH TO DATA"
# hf_token = "ENTER YOUR HUGGINGFACE TOKEN"

def sliding_window(row, chunk_size=509, overlap=50):
    words = re.findall(r'\b\w+\b', row['post'])
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        start = i
        end = min(i + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
    return pd.DataFrame({'post': chunks})

def expand_contractions(sentence):
    contractions_expanded = [contractions.fix(word) for word in sentence.split()]
    return ' '.join(contractions_expanded)

def lower_case(sentence):
    return ' '.join([word.lower() for word in sentence.split()])
def remove_punctuation(sentence):
    return ' '.join([re.sub(r'[^\w\s]', '', word) for word in sentence.split()])

def preprocess(lst, process=True, min_words=20):
    lst[:] = filterfalse(lambda x: len(x.split()) <= min_words, lst)
    if process == True:
        for i, sent in enumerate(lst):
            lst[i] = lower_case(remove_punctuation(expand_contractions(sent)))
    return lst

class SelfSupDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=512):

        self.source_column = "post"
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask}

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            input_ = self.data.loc[idx, self.source_column]
            tokenized_inputs = self.tokenizer(
                input_, max_length=self.max_len, truncation=True, return_tensors="pt",padding="max_length"
            )

            self.inputs.append(tokenized_inputs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_ss_dataset(tokenizer, data):
    return SelfSupDataset(tokenizer = tokenizer, data = data)

parquet_files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
dataframes = []

for file in parquet_files:
    file_path = os.path.join(data_path, file)
    df = pq.read_table(file_path).to_pandas()
    dataframes.append(df)

df_self_sup = pd.concat(dataframes, ignore_index=True)
df_self_sup.rename(columns={"description":"post"}, inplace=True)
df_self_sup = pd.concat([sliding_window(row) for _, row in df_self_sup.iterrows()], ignore_index=True)

set_seed(42)
start_time = time.time()
train_sentences = preprocess(list(df_self_sup['post']), min_words = 50)
train_df = pd.DataFrame([])
train_df['post'] = train_sentences

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

batch_size = 16
target_batch_size = 128
accumulation_size = target_batch_size//batch_size

train_dataset = get_ss_dataset(tokenizer, data = train_df)
dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
optimizer = AdamW(model.parameters(), lr = 5e-5)

n_epochs = 50

for epoch in range(n_epochs):
    loop = tqdm(dataloader, leave=True)
    for i, batch in enumerate(loop):
        batch['labels'] = batch['source_ids'].clone()
        batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
        batch['source_ids'], batch['source_mask'] = batch['source_ids'].to(device), batch['source_mask'].to(device)

        masked_indices = torch.where(batch['source_ids'] != tokenizer.pad_token_id)
        masked_indices = masked_indices[0].to('cpu')
        mask = torch.rand(batch['source_ids'].shape) < 0.15
        mask[batch['source_ids'] == tokenizer.pad_token_id] = False
        mask.to(device)
        a = torch.full(masked_indices.shape, 0.8)
        mask[masked_indices] &= torch.bernoulli(a).unsqueeze(-1).bool()
        batch['source_ids'][mask] = tokenizer.mask_token_id

        outputs = model(input_ids=batch['source_ids'].to(device),attention_mask=batch['source_mask'].to(device),labels=batch['labels'].to(device))
        loss = outputs.loss
        loss.backward()
        
        if (i + 1)%accumulation_size == 0 or i == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            
            loop.set_description(f'Epoch {100+epoch+1}')
            loop.set_postfix(loss=loss.sum().item())
            
    # if (epoch + 1) % 5 == 0:
    #     model.push_to_hub(f"bert-base-{100+epoch+1}-ep-pretrain-on-textbooks")
model.save_pretrained('CASE_model')




