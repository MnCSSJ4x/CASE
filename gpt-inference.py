import pandas as pd
from tqdm.auto import tqdm
from retry import retry
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


def make_prompt(top_n_rows, test_row):
    prompt = """Act as a professional psychiatrist and study the passage for signs of mental health disorders. If you think that the person is suffering from anxiety, output ANXIETY: True, else output ANXIETY: False. Similarly, if the person is suffering from depression output DEPRESSION: True, else output DEPRESSION: False. Do not print anything else. There are some examples on how to do this provided in the backticks.
```"""
    for i, row in top_n_rows.iterrows():
        prompt += f"""
Passage: {row['questionFull']}
ANXIETY: {row['Anxiety']}
DEPRESSION: {row['Depression']}
"""

    prompt += f"""
```
Using the understanding of the task gained from the above examples do the same for the following passage
Passage: {test_row['questionFull']}
"""
    return prompt


def make_message(top_n_rows, test_row):
    message = [{"role": "user", "content": f"{make_prompt(top_n_rows, test_row)}"}]

    return message


@retry(exceptions=Exception, tries=2, delay=30)
def make_call(top_n_rows, test_row):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=make_message(top_n_rows, test_row),
            # request_timeout=60
        )
        return response
    except KeyboardInterrupt:
        print("Closing...")
        return "Error"
    except Exception as e:
        print("Timeout occurred, retrying")
        print(e)
        raise e


def make_few_shot_prompt_call(top_n_rows, test_row):
    # prompt = make_prompt(top_n_rows, test_row)
    resp = make_call(top_n_rows, test_row)
    # return resp
    pbar.update(1)

    return resp


def get_text(api_response):
    try:
        ret = api_response.choices[0].message.content
        return ret
    except:
        print("Error occurred with the text retrieval")
        return "Error"


client = OpenAI(api_key="<YOUR-API-TOKEN>")
n = 1

df = pd.read_csv("oneHotData.csv")  # Enter Link to your Dataset
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
few_shot_args = [
    (train_data.sample(n), test_row) for i, test_row in test_data.iterrows()
]

MAX_THREADS = 10
pbar = tqdm(total=len(few_shot_args))
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    res = list(executor.map(make_few_shot_prompt_call, *zip(*few_shot_args)))

outputs = [op.choices[0].message.content.replace("\n", " <NEWLINE> ") for op in res]
res_df = pd.DataFrame({"generated_outputs": outputs})
res_df.to_csv(
    f"GPT3.5_Outputs_{n}Shot_CounselChat.csv", index=False
)  # Enter Output Path
