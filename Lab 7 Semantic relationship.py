import nltk
from nltk.corpus import wordnet

def find_semantic_relationship(word1, word2):
    # Get synsets for the input words
    synsets_word1 = wordnet.synsets(word1)
    synsets_word2 = wordnet.synsets(word2)

    if not synsets_word1:
        print(f"No synsets found for '{word1}'")
        return
    if not synsets_word2:
        print(f"No synsets found for '{word2}'")
        return

    # Find semantic relationships between synsets of the two words
    relationships = []
    for synset1 in synsets_word1:
        for synset2 in synsets_word2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None:
                relationships.append((synset1, synset2, similarity))

    if not relationships:
        print(f"No semantic relationships found between '{word1}' and '{word2}'")
        return

    # Sort relationships by similarity score in descending order
    relationships.sort(key=lambda x: x[2], reverse=True)

    # Print top 3 semantic relationships
    print(f"Top semantic relationships between '{word1}' and '{word2}':")
    for i, (synset1, synset2, similarity) in enumerate(relationships[:3]):
        print(f"{i+1}. Similarity: {similarity:.2f}")
        print(f"   {synset1.definition()}")
        print(f"   {synset2.definition()}")
        print()

# Input prompt
word1 = input("Enter the first word: ").lower()
word2 = input("Enter the second word: ").lower()

# Call the function with user-provided input
find_semantic_relationship(word1, word2)
