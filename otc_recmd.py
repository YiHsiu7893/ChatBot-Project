import pandas as pd
import spacy
import numpy as np

def otc_recmd(text):
    nlp = spacy.load("en_core_web_md")

    # Load data (length 362)
    sample = pd.read_csv("OTC.csv")
    otcs = sample["中文品名"]
    idcs = sample["Indications"]

    simi = list()

    inp = nlp(text)
    idx = 0
    for idc in idcs:
        idc_vec = nlp(idc)
        if(idc_vec and idc_vec.vector_norm):
            simi.append((idx, inp.similarity(idc_vec)))
        else:
            simi.append((idx, 0))
        idx += 1

    # def a(x):
    #     return x[1]
    # similarity.sort(key=a, reverse=True)
    simi.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 OTC recommendations:")
    for i in range(5):
        print(otcs[simi[i][0]], simi[i][1])
