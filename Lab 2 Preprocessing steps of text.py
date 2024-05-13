import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag


# Sample article
article = """
Machine learning is a subfield of artificial intelligence that gives computers the ability to learn without being explicitly programmed. 
Machine learning algorithms use historical data as input to predict new output values. 
The goal of machine learning is to create algorithms that can generalize from the training data to new, unseen data. 
Machine learning is used in a wide variety of applications, including:
- Natural language processing
- Image recognition
- Speech recognition
- Medical diagnosis
- Financial forecasting
- Fraud detection
- Robot control
"""

# Tokenization
tokens = word_tokenize(article)
print("\nTokenization:")
print(tokens)

# Lemmatization on tokenized words
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
print("\nLemmatization:")
print(lemmatized_words)

# Stop word removal on lemmatized words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
print("\nStop Word Removal:")
print(filtered_words)

# POS tagging on filtered words
pos_tags = pos_tag(filtered_words)
print("\nPOS Tagging:")
print(pos_tags)

# Porter stemming on POS tagged words
porter_stemmer = PorterStemmer()
stemmed_words = [porter_stemmer.stem(word) for word, tag in pos_tags]
print("\nPorter Stemming:")
print(stemmed_words)
