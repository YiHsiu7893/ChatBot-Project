import pandas as pd
from sentence_transformers import SentenceTransformer, util

def otc_recmd(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    #nlp = spacy.load("en_core_web_md")

    # Load data (length 362)
    sample = pd.read_csv("OTC.csv")
    otcs = sample["中文品名"]
    idcs = sample["Indications"]

    idc_vecs = model.encode(idcs)

    simi = list()

    inp = model.encode(text)
    idx = 0
    for idc_vec in idc_vecs:
        simi.append((idx, util.pytorch_cos_sim(inp, idc_vec)[0][0]))
        idx += 1

    # def a(x):
    #     return x[1]
    # similarity.sort(key=a, reverse=True)
    simi.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 OTC recommendations:")
    for i in range(5):
        print(otcs[simi[i][0]], simi[i][1].item())

# otc_recmd("I have a sore throat and keep coughing. I feel my throat is very dry and I have a fever.")
