import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.tokenize import word_tokenize


# Deal with contractions
def en_contractions(text):
    return " ".join([contractions.fix(word) if word in contractions.contractions_dict else word
                     for word in text.split()])

# Set of English stopwords 
stop_words = set(stopwords.words('english'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()


# 1: Normal Pre-Processing Module
def tokenizer(sent):
    sent = en_contractions(sent)

    # Remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # Tokenize and Remove stopwords
    words = word_tokenize(sent)
    words = [word.lower() for word in words if word not in stop_words or word=="yes" or word=="no"]

    # Lemmatization
    lemmatized_word = [lemmatizer.lemmatize(word, pos="v") for word in words]

    return lemmatized_word
