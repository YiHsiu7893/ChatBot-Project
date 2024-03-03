# for NER
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# for word2vec
from gensim.models import KeyedVectors
# import gensim.downloader as api
import numpy as np

# Problem: every time call extraction, need to load these two pre-trained model or vector
# which might be slow
def feat_extraction(text):
    # input is free text (string)

    # load NER pre-trained model
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # 83 labels (actually 41 entity)
    # https://huggingface.co/d4data/biomedical-ner-all/blob/main/config.json
    pipe = pipeline("ner", model=model, tokenizer=tokenizer,  aggregation_strategy="simple") # pass device=0 if using gpu

    ent2id = {'Age': 0, 'Personal_background': 1, 'Sex': 2, 'Sign_symptom': 3, 'Duration': 4, 'Clinical_event': 5, 'Nonbiological_location': 6, 'Diagnostic_procedure': 7, 'Biological_structure': 8, 'History': 9, 'Medication': 10, 'Family_history': 11, 'Lab_value': 12, 'Detailed_description': 13, 'Coreference': 14, 'Volume': 15, 'Disease_disorder': 16, 'Therapeutic_procedure': 17, 'AnnotatorNotes': 18, 'Dosage': 19, 'Date': 20, 'Color': 21, 'Texture': 22, 'Administration': 23, 'Time': 24, 'Severity': 25, 'Distance': 26, 'Shape': 27, 'Area': 28, 'Frequency': 29, 'Other_entity': 30, 'Other_event': 31, 'Subject': 32, 'Occupation': 33, 'Quantitative_concept': 34, 'Outcome': 35, 'Mass': 36, 'Height': 37, 'Weight': 38, 'Biological_attribute': 39, 'Activity': 40, 'Qualitative_concept': 41}

    # get pre-downloaded BioWordVec
    w2v = KeyedVectors.load_word2vec_format('../bio_embedding_extrinsic', binary=True)

    def medical_NER(text):
        return pipe(text)

    # dealing with subword tokenization
    def subword_refactor(entity_list):
        # for ent_list in NER_rslt:
        idx = 0
        while idx < len(entity_list):
            while idx < len(entity_list) and entity_list[idx]['word'][0] == '#':
                print(entity_list[idx])
                if entity_list[idx-1]['end'] == entity_list[idx]['start']:
                    if len(entity_list[idx]['word'].split('##')) > 1:
                        entity_list[idx-1]['word'] = entity_list[idx-1]['word'] + entity_list[idx]['word'].split('##')[1]
                    entity_list[idx-1]['end'] = entity_list[idx]['end']
                entity_list.pop(idx)
            idx += 1
        return entity_list

    def get_vec(ner_list):
        vecs = list()
        # add entity info
        for ner in ner_list:
            if ner['word'] in w2v:
                vec = w2v[ner['word']]
                vec = np.append(ent2id[ner['entity_group']], vec)
                vecs.append(vec)
                #vecs = np.vstack((vecs, vec))
            else:
                for tk in ner['word'].split():
                    if tk in w2v:
                        vec = w2v[tk]
                        vec = np.append(ent2id[ner['entity_group']], vec)
                        vecs.append(vec)
                    else: # exception
                        print(ner)
                        pass

        vecs = np.array(vecs)
        # print(vecs.shape)
        return vecs

    return get_vec(subword_refactor(medical_NER(text)))
# Output dimension will be #(tokens)*201
