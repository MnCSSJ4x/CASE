import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def get_depr_anx(gen_text):
    dep_start_id = gen_text.rfind("DEPRESSION")
    if gen_text[dep_start_id : dep_start_id + 17].rfind("True") != -1:
        dep = 1
    elif gen_text[dep_start_id : dep_start_id + 17].rfind("False") != -1:
        dep = 0
    else:
        dep = 2

    anx_start_id = gen_text.rfind("ANXIETY")
    if gen_text[anx_start_id : anx_start_id + 14].rfind("True") != -1:
        anx = 1
    elif gen_text[anx_start_id : anx_start_id + 14].rfind("False") != -1:
        anx = 0
    else:
        anx = 2

    return dep, anx


df = pd.read_csv("oneHotData.csv")  # ENTER THE PATH TO THE CSV FILE
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
test_pred = pd.read_csv(
    "GPT3.5_Outputs_1Shot_CounselChat.csv"
)  # ENTER THE PATH TO GPT3.5 OUTPUTS
preds = test_pred["generated_outputs"].apply(get_depr_anx)
pred_dep, pred_anx = zip(*preds)
label_dep = list(test_data["Depression"].apply(lambda x: 1 if x == True else 0))
label_anx = list(test_data["Anxiety"].apply(lambda x: 1 if x == True else 0))

# For Depression
print(accuracy_score(label_dep, pred_dep))
print(f1_score(label_dep, pred_dep))
print(precision_score(label_dep, pred_dep))
print(recall_score(label_dep, pred_dep))

# For Anxiety
print(accuracy_score(label_anx, pred_anx))
print(f1_score(label_anx, pred_anx))
print(precision_score(label_anx, pred_anx))
print(recall_score(label_anx, pred_anx))
