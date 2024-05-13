import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from collections import Counter
import re

# Sample corpus
corpus = """
King Krishnadevaraya loved horses and had the best collection of horse breeds in the Kingdom. 
Well, one day, a trader came to the King and told him that he had brought with him a horse of the best breed in Arabia. 
He invited the King to inspect the horse. King Krishnadevaraya loved the horse; so the trader said that the King could buy this one and that he had two more like this one, back in Arabia that he would go back to get. 
The King loved the horse so much that he had to have the other two as well. 
He paid the trader 5000 gold coins in advance. The trader promised that he would return within two days with the other horses.
Two days turned into two weeks, and still, there was no sign of the trader and the two horses. 
One evening, to ease his mind, the King went on a stroll in his garden. 
There he spotted Tenali Raman writing down something on a piece of paper. 
Curious, the King asked Tenali what he was jotting down. 
Tenali Raman was hesitant, but after further questioning, he showed the King the paper. 
On the paper was a list of names, the King’s being at the top of the list. 
Tenali said these were the names of the biggest fools in the Vijayanagara Kingdom! 
As expected, the King was furious that his name was at the top and asked Tenali Raman for an explanation. 
Tenali referred to the horse story, saying the King was a fool to believe that the trader, a stranger, would return after receiving 5000 gold coins. 
Countering his argument, the King then asked, what happens if/when the trader does come back? 
In true Tenali humour, he replied saying, in that case, the trader would be a bigger fool, and his name would replace the King’s on the list!
"""

# Preprocess the corpus
corpus = re.sub(r'\n', ' ', corpus)  # Remove newline characters
corpus = re.sub(r'\s+', ' ', corpus)  # Replace multiple spaces with single space
corpus = corpus.lower()  # Convert to lowercase

# Tokenize the corpus
tokens = nltk.word_tokenize(corpus)

# Compute TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([corpus])

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create unigrams, bigrams, and trigrams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Compute probabilities for unigrams, bigrams, and trigrams
unigram_prob = Counter(unigrams)
bigram_prob = Counter(bigrams)
trigram_prob = Counter(trigrams)

# Normalize probabilities
total_unigrams = sum(unigram_prob.values())
total_bigrams = sum(bigram_prob.values())
total_trigrams = sum(trigram_prob.values())

unigram_prob = {gram: count / total_unigrams for gram, count in unigram_prob.items()}
bigram_prob = {gram: count / total_bigrams for gram, count in bigram_prob.items()}
trigram_prob = {gram: count / total_trigrams for gram, count in trigram_prob.items()}

# Print TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Print feature names
print("\nFeature Names (Words):")
print(feature_names)

# Print N-gram probabilities
print("\nUnigram Probabilities:")
print(unigram_prob)
print("\nBigram Probabilities:")
print(bigram_prob)
print("\nTrigram Probabilities:")
print(trigram_prob)
