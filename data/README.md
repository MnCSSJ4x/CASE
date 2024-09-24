# Dataset for "CASE: Efficient Curricular Data Pre-training for Building Assistive Psychology Expert Models"

This folder contains the dataset used for the paper "CASE: Efficient Curricular Data Pre-training for Building Assistive Psychology Expert Models".

## Pretrain Dataset Description

This dataset includes 110 curricular text materials for training graduate-level Psychology students. It covers a wide range of topics, including general psychology, interviewing and counseling skills, cognition, clinical psychology, psychotherapy, telepsychotherapy, sociology, and geriatric counseling.

The texts are sourced globally, particularly from North America, Asia, and Europe, following the American Psychology Association’s (APA, 2019) guidelines. The dataset aims to be data-efficient, containing 7,567,108 words or 365,937 sentences, making it suitable for domains with data scarcity.

## Finetuning Dataset(s) Description

The finetuning datasets include:

### CounselChat Dataset

CounselChat is an expert community platform where therapists answer queries from customers. The dataset consists of 1374 rows with an average passage size of 140 words. Each sentence is tagged with multiple labels such as depression, anxiety, addiction, marriage, and relationships. For our model, we use data points tagged with depression and anxiety, resulting in 198 and 231 rows labeled true for Depression and Anxiety, respectively.

### Other Datasets on Depression and Stress

We also use Depression_Reddit and Dreaddit datasets for finetuning and testing. Depression_Reddit focuses on identifying signs of depression from Reddit posts, while Dreaddit focuses on identifying signs of stress from Reddit posts across five different forums. These datasets help ensure consistency with prior work.

## Usage Restrictions

The dataset is provided for academic purposes and experimentation only. It cannot be used for any commercial purposes.
