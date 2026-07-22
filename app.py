import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    tokens = [i for i in tokens if i.isalnum()]

    # Remove stopwords and punctuation
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit UI
st.markdown("## 📧 Email/SMS Spam Classifier")
st.write("This app uses **Machine Learning** to detect whether a message is spam or not.")

# Sidebar with examples
st.sidebar.title("📌 Example Messages")
examples = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
    "Hey, are we still meeting tomorrow?",
    "URGENT! Your account has been compromised. Reset your password immediately."
]
choice = st.sidebar.radio("Select an example:", examples)
input_sms = st.text_area("Enter the message", value=choice if choice else "")

# Prediction
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
