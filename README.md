# ChatBot-Project

## Usage
Please use `pip install -r requirements.txt` to get modules you need for this chatbot.

Then, run `python -W ignore Chatbot.py` to start.

You can also run `python api.py` to check web service run in Flask.

## Architecture
<img src="https://github.com/YiHsiu7893/ChatBot-Project/blob/main/images/architecture.jpg" width=70% height=70%>


## Datasets
Use following datasets for training or drug recommendations.
* https://www.kaggle.com/datasets/niyarrbarman/symptom2disease/
* https://data.fda.gov.tw/frontsite/data/DataAction.do?method=doDetail&infoId=36

## Feature_Ext.py
Use pre-trained model and vector, biomedical-ner-all and BioWordVec.
* https://huggingface.co/d4data/biomedical-ner-all
* https://github.com/ncbi-nlp/BioWordVec
