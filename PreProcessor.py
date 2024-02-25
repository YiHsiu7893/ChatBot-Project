import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer


# Deal with contractions
def en_contractions(text):
    return " ".join([contractions.fix(word) if word in contractions.contractions_dict else word
                     for word in text.split()])

# Set of English stopwords 
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()


# 1: Normal Pre-Processing Module
def normal_preprocess(sent):
    sent = en_contractions(sent)

    # Remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # Tokenize and Remove stopwords
    words = word_tokenize(sent)
    words = [word.lower() for word in words if word not in stop_words or word=="yes" or word=="no"]

    # Lemmatization
    lemmatized_word = [lemmatizer.lemmatize(word, pos="v") for word in words]

    #return " ".join(lemmatized_word)
    return lemmatized_word


# 2: Word2Vec Pre-Processing Module
def w2v_preprocess(sent):
    sent = en_contractions(sent)

    # Remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    tokenized_sent = [tokenizer.decode(word) for word in tokenizer(sent)["input_ids"]][1:-1]

    idx = 0
    while idx < len(tokenized_sent):
        while idx < len(tokenized_sent) and tokenized_sent[idx][0] == '#':
            if len(tokenized_sent[idx].split('##')) > 1:
                    tokenized_sent[idx-1] = tokenized_sent[idx-1] + tokenized_sent[idx].split('##')[1]
            tokenized_sent.pop(idx)
        idx += 1

    return tokenized_sent