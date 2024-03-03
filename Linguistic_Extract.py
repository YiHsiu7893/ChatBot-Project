### Incomplete!!
import requests
import torch

from gensim.models import KeyedVectors
from PreProcessor import w2v_preprocess


# To-Do: These hyperparameters are undecided.
api_endpoint = "https://api.openai.com/v1/completions"
api_key = "OPENAI_API_KEY"
gpt_model = "text-davinci-003"
max_tokens = 100
hidden_dim = 200


w2v = KeyedVectors.load_word2vec_format('../bio_embedding_extrinsic', binary=True)


# To-Do: Define a more appropriate prompt template.
prompt_template = "Now, imagine yourself as a doctor.\
\nThe patient is describing their symptoms.\
Please extract the keywords of these symptoms and analyze the properties of time and the impact level.\
\nThen, produce a description similar to \"Analysis\".\
\n\nThe definitions are as follows:\
\nKeyword Extraction: Extract keywords from the patient's description that may be related to specific diseases.\
\nTime Description: Extract information related to the duration of symptom presentation, frequency of occurrence, and other time-related details.\
\nImpact Assessment: Analyze the extent to which the symptoms described by the patient affect daily life, such as work ability, sleep quality, appetite, etc.\
\nAdditionally, omit subject pronouns (e.g., I, he, the patient, etc.), and use an affirmative tone.\
\n\
\n\"Analysis\"\
\n[keyword 1] persists for [time 1], and [keyword 2] persists for [time 2]... . These symptoms have [impact] on life.\
\n\
\n{patient description}"
"""
# template 修改方向:
*1. 關鍵詞提取：從病患的敘述中提取關鍵詞，這些詞可能與特定的疾病相關。
2. 情感分析：分析病患的語言中隱含的情感信息，如焦慮、憂鬱、痛苦等，這些情感可能與某些疾病有關。
3. 疼痛描述：從病患的描述中提取與疼痛相關的信息，如疼痛的位置、程度、性質等。
*4. 時間描述：提取描述症狀持續時間、發作頻率等時間相關信息。
*5. 影響程度：分析病患敘述中症狀對日常生活的影響程度，如工作能力、睡眠品質、食慾等。
"""


headers = {
    "Content-Type": "text/plain",
    "Authorization": f"Bearer {api_key}"
}


def gpt_call(input):
    input_text = prompt_template.format(**{"patient description": input})
    data = {
    "model": gpt_model,   
    "prompt": input_text,          
    "max_tokens": max_tokens            
    }

    #response = requests.post(api_endpoint, headers=headers, json=data)
    #out = Preprocessing(response.text)
    out = w2v_preprocess(input)

    embedded_out = torch.empty((len(out), hidden_dim), dtype=torch.float32)
    for i, word in enumerate(out):
        embedded_out[i] = torch.tensor(w2v[word])


    print("\n--- just a test for successfully running gpt_call ---\nreceiving sentence is:")
    print(input)

    return embedded_out


"""
# for 測試用
sent = "I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. It is red, itchy, and covered in dry, scaly patches."
result = gpt_call(sent)
print(result)
print(result.shape)
"""
