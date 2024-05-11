import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests, os
from googletrans import Translator

def update_otc():
    req = requests.get("https://data.fda.gov.tw/opendata/exportDataList.do?method=ExportData&InfoId=36&logType=2")

    cont = req.content
    csv = open("OTC.zip", 'wb')
    csv.write(cont)
    csv.close()
    os.system("unzip OTC.zip > /dev/null && rm OTC.zip")
    os.system("head -n 1 36_2.csv > tmp.csv && grep -v '已註銷' 36_2.csv >> tmp.csv && rm 36_2.csv")
    os.system("head -n 1 tmp.csv > OTC.csv && grep '成藥' tmp.csv >> OTC.csv && rm tmp.csv")
    df_tmp = pd.read_csv("OTC.csv")
    for idx, row in df_tmp.iterrows():
        if "成藥" not in row["藥品類別"]:
            df_tmp.drop(idx, inplace=True)
    
    # do translation
    translator = Translator(
        service_urls=['translate.google.com', 
                      'translate.google.com.tw'])
    # use dictionary to store the translation
    d = dict()
    trans = []
    for row in df_tmp.iterrows():
        i = row[1]["適應症"]
        if i not in d:
            d[i] = translator.translate(i, src='zh-tw', dest='en').text
        trans.append(d[i])
    df_tmp.insert(12, "Indications", trans)
    df_tmp.to_csv("OTC.csv", index=False)
    # df_tmp.to_excel("OTC.xlsx", index=False)


def otc_recmd(text):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    # Load data
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
            results += otcs[simi[i][0]] + ", " + str(simi[i][1].item()) + ", " + idcs_zh[simi[i][0]] + "\n"
            # print(otcs[simi[i][0]], simi[i][1].item(), "\n", idcs_zh[simi[i][0]])
    return results
    
# update_otc()
# otc_recmd("I have a sore throat and keep coughing. I feel my throat is very dry and I have a fever.")
