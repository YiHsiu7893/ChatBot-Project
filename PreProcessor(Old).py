import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.tokenize import word_tokenize
import nltk

# pre download corpus
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

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
    words = [word.lower() for word in words if word not in stop_words or word=="yes" or word=="no"]

    # Lemmatization
    lemmatized_word = [lemmatizer.lemmatize(word, pos="v") for word in words]

    #return " ".join(lemmatized_word)
    return lemmatized_word