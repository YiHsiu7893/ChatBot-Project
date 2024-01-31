import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.tokenize import word_tokenize


# Read data
df = pd.read_csv("Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)

print("\n===== before =====")
print(df[:5])


# Deal with contractions
def en_contractions(text):
    return " ".join([contractions.fix(word) if word in contractions.contractions_dict else word
                     for word in text.split()])

# Set of English stopwords 
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()


# Pre-Processing Module
def Preprocessing(sent):
    sent = en_contractions(sent)

    # Remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # Remove stopwords
    words = word_tokenize(sent)
    words = [word.lower() for word in words if word not in stop_words]

    # Lemmatization
    lemmatized_word = [lemmatizer.lemmatize(word, pos="v") for word in words]

    #return " ".join(lemmatized_word)
    return lemmatized_word


df["text"] = df["text"].apply(Preprocessing)
print("\n===== after =====")
print(df[:5])