import nltk
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

# Part-of-speech tagging
pos_tags = nltk.pos_tag(tokens)

# Define a chunk grammar to identify noun phrases
chunk_grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(chunk_grammar)

# Apply chunking
chunked_tokens = chunk_parser.parse(pos_tags)

# Extract noun phrases
noun_phrases = []
for subtree in chunked_tokens.subtrees(filter=lambda t: t.label() == 'NP'):
    noun_phrase = ' '.join(word for word, tag in subtree.leaves())
    noun_phrases.append(noun_phrase)

# Print extracted noun phrases
print("Extracted Noun Phrases:")
for phrase in noun_phrases:
    print(phrase)
