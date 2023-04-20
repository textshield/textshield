from flask import Flask, jsonify, request, json 
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore") 

app = Flask(__name__)

CORS(app)

def load_model():
    with open('best_model_RoBERTa.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

model = load_model()

@app.route('/test', methods=['POST'])
def test():
    text = request.get_json()['text']
    result = ""
    pred = ""
    t = modelPredict(text)
    if t[0][0] == 1:
       pred = "Offensive"
    else:
       pred = "Not Offensive"
    
    result = jsonify({"result":pred,"text":text})
    
    return result 



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

def stemmer(text):
  tokenized = nltk.word_tokenize(text)
  ps = PorterStemmer()
  return ' '.join([ps.stem(words) for words in tokenized])
def lemmatize(text):
  tokenized = nltk.word_tokenize(text)
  lm = WordNetLemmatizer()
  return ' '.join([lm.lemmatize(words) for words in tokenized])
def preprocess(text):
  text = stemmer(text)
  text = lemmatize(text)
  return text

# Function to convert text to lowercase, remove  Fucntion to Convert text to lowercase, remove line breaks, URLs, non-utf characters, Numbers, punctuations
def clean_data(text):
    
    # Convert text to lower case
    text = text.lower()

    # replace contarctions

    # contraction dictionary
    contractions = {
        "ain't": "am not / are not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is",
        "i'd": "I had / I would",
        "i'd've": "I would have",
        "i'll": "I shall / I will",
        "i'll've": "I shall have / I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }

    for word in text.split():
        if word in contractions:
            text = text.replace(word, contractions[word.lower()])

    # replace line breaks
    text = text.replace('\n', '').replace('\r', ' ').lower()

    # remove user names
    text = text.replace('@user ', '')

    # remove urls
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)

    # remove numbers
    text = re.sub('[0-9]+', '', text)

    # remove non-utf charactwrs
    text = re.sub(r'[^\x00-\x7f]',r'', text)

    # remove multiple spaces
    text = re.sub("\s\s+" , " ", text)

    # r'(.)1+' matches any character followed by one or more instances of the same character
    text = re.sub(r'(.)1+', r'1', text)

    # remove hashtags
    temp = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
    text = " ".join(word.strip() for word in re.split('#|_', temp))
    
    
    # remove punctuation [keep at the end, to avoid including hashtag terms and usernames]
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove stop words
    # text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

def modelPredict(text):
    df_trial = pd.DataFrame({"text": text }, index = [0])
    
    df_trial['text'] = df_trial['text'].apply(clean_data)
    print(df_trial['text'])
    
    df_trial['text'] = df_trial['text'].apply(preprocess)
    print(df_trial['text'])
    
    predictions = model.predict(df_trial['text'])
    
    return predictions



# def do_something():
#     # Load the pickle file
#     with open('best_model_RoBERTa.pkl', 'rb') as f:
#         model = pickle.load(f)

#     return model

if __name__ == '__main__':
    app.run(debug=True)