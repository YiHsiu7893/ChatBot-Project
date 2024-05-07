import pandas as pd
from sentence_transformers import SentenceTransformer, util


def otc_recmd(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # Load data (length 362)
    sample = pd.read_csv("OTC.csv")

    otcs = sample["中文品名"]

    sample.loc[sample["Indications"].str.endswith("."), "Indications"] = sample["Indications"].str.rstrip(".")
    idcs_en = sample["Indications"].str.split(', ').apply(lambda x: ["I have a " + item for item in x])

    idcs_zh = sample["適應症"]


    # Indications embedding
    idc_vecs = [model.encode(idc) for idc in idcs_en]

    # Description embedding
    text = text.rstrip(".").split('. ')
    text_vecs = [model.encode(t) for t in text]


    # Compute similarity
    simi = list()
   
    idx = 0
    for idc_vec in idc_vecs:
        # Get the highest score for each medicine
        highest = 0.0

        # for each indication of the medicine
        for idc in idc_vec:
            # for each sentence in the description
            for text in text_vecs:
                score = util.pytorch_cos_sim(text, idc)[0][0]
                if score > highest:
                    highest = score

        simi.append((idx, highest))
        idx += 1

    # def a(x):
    #     return x[1]
    # similarity.sort(key=a, reverse=True)
    simi.sort(key=lambda x: x[1], reverse=True)


    # Show the result
    print("Top 5 OTC recommendations:")
    results = ""
    for i in range(5):
        if simi[i][1].item()>0.825:
            results += otcs[simi[i][0]] + ", " + str(simi[i][1].item()) + "\n" + idcs_zh[simi[i][0]] + "\n"
            # print(otcs[simi[i][0]], simi[i][1].item(), "\n", idcs_zh[simi[i][0]])
    return results
    

#otc_recmd("I have a sore throat and keep coughing. I feel my throat is very dry and I have a fever.")
