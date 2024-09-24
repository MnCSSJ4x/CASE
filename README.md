# CASE: Efficient Curricular Data Pre-training for Building Assistive Psychology Expert Models

This repository contains the code and data for the paper "CASE: Efficient Curricular Data Pre-training for Building Assistive Psychology Expert Models" presented at EMNLP2024.

## Authors

- Sarthak Harne
- Monjoy Narayan Choudhury
- Madhav Rao
- TK Srikanth
- Seema Mehrotra
- Apoorva Vashisht
- Aarushi Basu
- Manjit Sodhi

## Abstract

The limited availability of psychologists necessitates efficient identification of individuals requiring urgent mental healthcare. This study explores the use of Natural Language Processing (NLP) pipelines to analyze text data from online mental health forums used for consultations. By analyzing forum posts, these pipelines can flag users who may require immediate professional attention. A crucial challenge in this domain is data privacy and scarcity. To address this, we propose utilizing readily available curricular texts used in institutes specializing in mental health for pre-training the NLP pipelines. This helps us mimic the training process of a psychologist. Our work presents CASE-BERT that flags potential mental health disorders based on forum text. CASE-BERT demonstrates superior performance compared to existing methods, achieving an f1 score of 0.91 for Depression and 0.88 for Anxiety, two of the most commonly reported mental health disorders.

## Repository Structure

- `src/`: Contains all training and evaluation scripts.
- `data/`: Contains the datasets used for training and evaluation.

## Model Weights

The model weights can be found on [Hugging Face 🤗](https://huggingface.co/sarthakharne/bert-base-60-ep-pretrain-on-textbooks).
