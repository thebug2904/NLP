import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import defaultdict

def highlight_text(text, summary_length=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Calculate word frequency
    freq = FreqDist(filtered_words)

    # Score sentences based on word frequency
    ranking = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                ranking[i] += freq[word]

    # Get top 'summary_length' sentences with highest scores
    top_sentences_idx = sorted(ranking, key=ranking.get, reverse=True)[:summary_length]
    top_sentences = [sentences[i] for i in top_sentences_idx]

    # Highlight important words in the summary
    highlighted_summary = []
    for sentence in top_sentences:
        highlighted_sentence = ""
        for word in word_tokenize(sentence):
            if word.lower() in freq and freq[word.lower()] > 1:
                highlighted_sentence += f"\033[1m{word}\033[0m "  # Highlight important words
            else:
                highlighted_sentence += f"{word} "
        highlighted_summary.append(highlighted_sentence)

    return " ".join(highlighted_summary)

# Example text on equal rights for women
text = """Artificial Intelligence (AI) has revolutionized various industries, offering innovative solutions through Machine Learning (ML) and Natural Language Processing (NLP). NLP, a subset of AI, focuses on enabling machines to understand, interpret, and generate human language, opening doors to numerous applications across different sectors.

Importance in Different Industries:
- Healthcare: AI-powered NLP systems analyze medical records, assisting in diagnosis, treatment planning, and patient care, leading to more accurate and efficient healthcare delivery.
- Finance: ML algorithms process vast financial data for fraud detection, risk assessment, and algorithmic trading, enhancing security and decision-making in the financial sector.
- E-commerce: NLP-based recommendation systems personalize product suggestions, improving user experience and driving sales in online retail platforms.
- Customer Service: AI chatbots equipped with NLP capabilities handle customer inquiries, providing instant responses and improving customer satisfaction in various industries.

Use Cases:
- Sentiment Analysis: NLP algorithms analyze text data from social media, customer reviews, and surveys to gauge public opinion, helping businesses understand market trends and customer preferences.
- Language Translation: NLP models translate text between different languages, facilitating global communication and cross-cultural interactions.
- Text Summarization: NLP techniques condense large volumes of text into concise summaries, aiding information retrieval and knowledge extraction from documents.

Methods:
- Deep Learning: Deep neural networks power advanced NLP models, such as recurrent neural networks (RNNs) and transformers, enabling tasks like language translation, sentiment analysis, and text generation.
- Rule-based Systems: Traditional NLP approaches use predefined rules and linguistic patterns to process text, offering interpretable solutions for tasks like named entity recognition and syntactic parsing.

Advantages:
- Efficiency: AI-driven NLP systems automate tasks that would be time-consuming or impractical for humans, leading to increased productivity and cost savings.
- Scalability: ML algorithms can handle large datasets and adapt to evolving data distributions, making them suitable for scalable applications in diverse domains.
- Personalization: NLP-powered recommendation systems tailor content and services to individual preferences, enhancing user engagement and satisfaction.

Disadvantages:
- Data Bias: NLP models trained on biased datasets may perpetuate societal biases, leading to unfair or discriminatory outcomes in decision-making processes.
- Interpretability: Deep learning models often lack interpretability, making it challenging to understand their decision-making mechanisms and ensure accountability in critical applications.
- Privacy Concerns: NLP applications that process sensitive textual data raise privacy concerns regarding data collection, storage, and misuse, requiring robust privacy protection measures.

In conclusion, AI, ML, and NLP play crucial roles in transforming industries, offering innovative solutions, and driving advancements in various domains. While their applications bring numerous benefits, addressing challenges such as data bias, interpretability, and privacy is essential to ensure responsible and ethical deployment of AI technologies.
"""

# Get the summarized and highlighted text
highlighted_summary = highlight_text(text)
print(highlighted_summary)
