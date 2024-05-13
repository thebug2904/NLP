import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import re

# Given document
document = """Title: Transforming Healthcare with Machine Learning: A Journey Towards Precision Medicine

In recent years, the intersection of healthcare and machine learning has been nothing short of revolutionary. The integration of advanced technologies into the healthcare sector has opened up endless possibilities, promising to enhance patient care, streamline processes, and ultimately save lives. At the forefront of this revolution is the burgeoning field of precision medicine, which leverages machine learning algorithms to tailor treatments to individual patients based on their unique genetic makeup, lifestyle, and environmental factors.

Healthcare, traditionally driven by reactive measures, is now transitioning towards a proactive and personalized approach, thanks to the power of machine learning. Here's how it's reshaping the landscape of healthcare as we know it:

### Early Disease Detection and Diagnosis

One of the most significant applications of machine learning in healthcare is early disease detection and diagnosis. Machine learning algorithms can analyze vast amounts of patient data, including medical records, imaging scans, and genetic information, to identify patterns and markers indicative of diseases such as cancer, diabetes, and heart disease. By detecting these conditions at their nascent stages, physicians can intervene early, significantly improving patient outcomes and reducing healthcare costs.

### Personalized Treatment Plans

No two patients are alike, and what works for one may not work for another. Machine learning algorithms can analyze diverse datasets to identify optimal treatment strategies tailored to individual patients. By considering factors such as genetic predispositions, biomarkers, and treatment response patterns, these algorithms can help healthcare providers make more informed decisions about medication dosages, therapy options, and surgical interventions, leading to better outcomes and reduced adverse effects.

### Predictive Analytics for Better Patient Management

Predictive analytics powered by machine learning is revolutionizing patient management by forecasting disease progression, hospital readmissions, and adverse events. By analyzing patient data in real-time, these algorithms can identify high-risk individuals who may require intensive monitoring or intervention, enabling healthcare providers to intervene proactively and prevent adverse outcomes. This proactive approach not only improves patient outcomes but also enhances operational efficiency within healthcare systems.

### Drug Discovery and Development

The traditional drug discovery process is time-consuming, expensive, and often yields suboptimal results. Machine learning is revolutionizing this process by accelerating the identification of potential drug candidates, predicting their efficacy and safety profiles, and optimizing clinical trial designs. By leveraging vast datasets from genomics, proteomics, and chemical libraries, machine learning algorithms can identify novel drug targets and repurpose existing drugs for new indications, leading to faster and more cost-effective drug development.

### Healthcare Resource Optimization

Machine learning algorithms can optimize resource allocation within healthcare systems by predicting patient demand, staffing needs, and equipment utilization. By analyzing historical data and real-time metrics, these algorithms can identify inefficiencies, streamline workflows, and ensure that resources are allocated where they are most needed. This not only improves patient access to care but also reduces wait times, enhances patient satisfaction, and maximizes the efficiency of healthcare delivery.

### Ethical and Regulatory Considerations

While the potential benefits of machine learning in healthcare are undeniable, it is essential to address ethical and regulatory considerations to ensure responsible deployment and safeguard patient privacy and autonomy. Issues such as data security, algorithm bias, and transparency in decision-making must be carefully addressed to maintain trust in machine learning systems and mitigate potential harms.

In conclusion, the integration of machine learning into healthcare holds tremendous promise for transforming the delivery of care, from early disease detection and diagnosis to personalized treatment plans and predictive analytics. By harnessing the power of data-driven insights, healthcare providers can deliver more precise, efficient, and patient-centered care, ultimately improving outcomes and enhancing the quality of life for millions of individuals worldwide. As we continue to push the boundaries of innovation in healthcare, the possibilities for improving human health are truly limitless."""

# Preprocess the document by removing stop words and converting to lowercase
stop_words = set(stopwords.words('english'))
processed_words = [word.lower() for word in re.findall(r'\b\w+\b', document) if word.lower() not in stop_words]
processed_document = ' '.join(processed_words)

# Calculate word count
word_count = Counter(processed_words)

# Display individual word counts
print("\nIndividual Word Counts:")
for word, count in word_count.items():
    print(f"{word}: {count}")

# Display total word count
total_word_count = sum(word_count.values())
print(f"\nTotal Word Count: {total_word_count}")

# Display top 10 frequent words and their frequencies
top_words = word_count.most_common(10)
print("\nTop 10 frequent words and their frequencies:")
for word, freq in top_words:
    print(f"{word}: {freq}")

# Create Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_document)

# Display Word Cloud graphically
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Plot frequency distribution graph as a line graph for top 10 words
plt.figure(figsize=(12, 6))
top_words_dict = dict(top_words)
plt.plot(list(top_words_dict.keys()), list(top_words_dict.values()), marker='o', linestyle='-', color='b')
plt.title('Frequency Distribution of Top 10 Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
