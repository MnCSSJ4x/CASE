import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, AutoModel, AutoTokenizer
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import trange
import random
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import AdamW

import contractions

import re

import time
import gc

from itertools import filterfalse
from tqdm import trange
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# SET data_path TO FINETUNING DATASET
data_path = "PATH TO DATA"
# SET base_dir TO PRE TRAINED MODEL PATH
base_dir = 'PATH TO PRE TRAINED MODEL'
# SET you HF TOKEN TO PUSH MODEL TO HUB
hf_token = "ENTER YOUR HUGGINGFACE TOKEN"

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



class ClassificationDataset(Dataset):
    def __init__(self, 
                target_column, 
                tokenizer, 
                data, 
                max_len=512):

        self.source_column = 'post'
        self.target_column = target_column
        self.data = data

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target = self.targets[index]
        return {"source_ids": source_ids, "source_mask": src_mask, "target": target}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx, self.source_column], self.data.loc[idx, self.target_column]

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            if target == True:
                self.targets.append(1)
            else:
                self.targets.append(0)

df_class = pd.read_csv(data_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_classification_dataset(target_col, tokenizer, data):
    return ClassificationDataset(target_col, tokenizer = tokenizer, data = data)

metric_dict = {}

def process_and_evaluate(
    model_loc: str,
    tokenizer_loc: str,
    column: str,
    sanity_check: bool = False
):
    # preprocessing
    train_sentences = preprocess(list(df_class['questionFull']))
    labels = list(df_class[column])
    
    # creating a new dataframe
    train_df = pd.DataFrame([])
    train_df['post'] = train_sentences
    train_df['Disorder'] = labels 
    del train_sentences
    gc.collect()
    
    # getting the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_loc, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_loc, token=hf_token).to(device)
    
    # Putting dummy y's for y_train y_test
    y = np.ones(train_df.shape[0])
    
    X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.20, random_state=42)
    
    X_train = X_train.reset_index().drop(columns=["index"])
    X_test = X_test.reset_index().drop(columns=["index"])
    
    train_dataset = get_classification_dataset('Disorder', tokenizer, data=X_train)
    test_dataset = get_classification_dataset('Disorder', tokenizer, data=X_test)
    train_dataloader = DataLoader(train_dataset, batch_size = 32)
    test_dataloader = DataLoader(test_dataset, batch_size = 32)

    del X_train, X_test, y_train, y_test
    gc.collect()
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch['source_ids'],attention_mask=batch['source_mask'],labels=batch['target'])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if sanity_check == True:
                break
        if sanity_check == True:
            break
            
    
    model.eval()
    y_true = []
    y_pred = []
    debug_mode = False
    with torch.no_grad():
        for batch in test_dataloader:
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(source_ids, attention_mask=source_mask)
            _, predicted = torch.max(outputs.logits, 1) 
            
            y_true.extend(targets.cpu().numpy())
            if debug_mode: 
                y_pred.extend([1] * len(predicted)) 
            else:
                y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    print(f'For class: {column} f1 score: {f1}   Accuracy: {accuracy}')
    arr.append({
        "model_name": model_loc,
        "class": column,
        "f1 score": f1,
        "accuracy": accuracy,
    })
    
# COLUMNS TO PROCESS AND EVALUATE
good_cols = ['Anxiety', 'Depression']

arr = []
tokenizer_dir = 'bert-base-uncased'
print('-'*100)
for col in good_cols:
    print(f'{col} Started')
    torch.cuda.empty_cache()
    gc.collect()
    try:
        process_and_evaluate(base_dir,tokenizer_dir, col)
    except Exception as e:
        print(f'EXCEPTION OCCURED AT CLASS {col}, for MODEL {base_dir}')
        print(e)
        continue
    print('-'*100)