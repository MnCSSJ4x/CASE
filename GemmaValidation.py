import numpy as np
import torch
from evaluate import load
import pandas as pd
from BARTScore.SUM.bart_score import BARTScorer

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
data_path = "PATH TO DATA"

model_name = "roberta-large"
bart_model_name = 'facebook/bart-large-cnn'


def compute_bert_score(df):
    predictions = df["Predicted Diagnosis"].tolist()
    references = df["Generated Diagnosis Summary"].tolist()
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type=model_name)
    return results

def compute_bart_score(df):
    predictions = df["Predicted Diagnosis"].tolist()
    references = df["Generated Diagnosis Summary"].tolist()
    bartscore = BARTScorer(device, max_length=2048)
    bartscore.load()
    results = bartscore.score(predictions, references, batch_size=2)
    return results

def compute_rouge_score(df):
    rouge = load("rouge")
    predictions = df["Predicted Diagnosis"].tolist()
    references = df["Generated Diagnosis Summary"].tolist()
    results = rouge.compute(predictions=predictions, references=references, rouge_types=["rouge1", "rouge2", "rougeL"])
    return results

def compute_meteor_score(df):
    meteor = load("meteor")
    predictions = df["Predicted Diagnosis"].tolist()
    references = df["Generated Diagnosis Summary"].tolist()
    results = meteor.compute(predictions=predictions, references=references)
    return results

df = pd.read_csv(data_path, sep='|')

eval_bert_score = compute_bert_score(df.copy())
eval_bart_score = compute_bart_score(df.copy())
eval_rouge_score = compute_rouge_score(df.copy())
eval_meteor_score = compute_meteor_score(df.copy())
print(eval_bert_score)
print(eval_bart_score)
print(eval_rouge_score)
print(eval_meteor_score)

print(np.array(eval_bart_score).mean())

print(np.array(eval_bert_score['precision']).mean())

print(np.array(eval_bert_score['recall']).mean())

print(np.array(eval_bert_score['f1']).mean())




