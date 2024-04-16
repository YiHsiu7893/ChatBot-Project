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
prompt_template = "You are a professional doctor and you have many experiences about patients. The following is patients’ descriptions and corresponded output. Please do the same thing as it. Please be careful of the description like duration in (). I just need the output other description should be discarded.\
\
“I've been feeling really exhausted lately, like I can barely get out of bed in the mornings. It's been going on for a few weeks now, and no matter how much sleep I get, I still feel drained. I've also noticed that I'm bruising more easily than usual, even from minor bumps or scratches. My appetite has decreased too, and I've lost a bit of weight without trying. Sometimes I feel nauseous, and I've had some stomach pain as well. Oh, and I've been running a low-grade fever off and on. I'm not sure what's going on, but it's starting to worry me.” \
[symptom] Exhaustion (a few weeks, despite sleep) \
[symptom] Easy bruising \
[symptom] Decreased appetite \
[symptom] Unintentional weight loss \
[symptom] Nausea (occasional) \
[symptom] Stomach pain (occasional) \
[symptom] Low-grade fever (occasional) \
\
“I've been experiencing a persistent cough for the past month or so. It started out as just a tickle in my throat, but now it's become quite frequent, especially at night. Along with the cough, I've noticed that I'm bringing up yellowish-green mucus. I also feel a tightness in my chest when I cough, and sometimes it even hurts. On top of that, I've been feeling really fatigued lately, like I can't keep up with my usual activities. I haven't had much of an appetite either, and I've lost a bit of weight unintentionally. Occasionally, I've had a low-grade fever, and I've been feeling generally achy all over. I'm concerned about what might be causing all of this.” \
[symptom] Persistent cough (1 month, worsening at night) \
[symptom] Yellowish-green mucus \
[symptom] Chest tightness \
[symptom] Chest pain (occasional) \
[symptom] Fatigue \
[symptom] Decreased appetite \
[symptom] Unintentional weight loss \
[symptom] Low-grade fever (occasional) \
[symptom] Generalized body aches \
\
“I have a rash on my legs that is causing a lot of discomforts. It seems there is a cramp and I can see prominent veins on the calf. Also, I have been feeling very tired and fatigued in the past couple of days.” \
"


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
