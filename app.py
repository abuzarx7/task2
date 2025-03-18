import streamlit as st
import nltk
import os

# Ensure NLTK data is stored in a valid location
nltk_data_path = os.path.expanduser("~/.nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download only the necessary NLTK resources
resources = ["punkt", "stopwords", "reuters"]
for resource in resources:
    nltk.download(resource, download_dir=nltk_data_path)

# Streamlit App Title
st.title("Information Retrieval with Word2Vec")

# Load necessary libraries
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare the corpus
st.write("Loading Reuters corpus...")
corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokenized_sentence = [word for word in word_tokenize(raw_text) if word.isalnum()]
    corpus_sentences.append(tokenized_sentence)

st.write(f"Number of sentences in the Reuters corpus: {len(corpus_sentences)}")

# Train Word2Vec model
st.write("Training Word2Vec model...")
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")

# User input
query_word = st.text_input("Enter a word to find similar words:")

# Find Similar Words
if st.button("Search"):
    if query_word in model.wv:
        similar_words = model.wv.most_similar(query_word, topn=10)
        st.write("Top similar words:")
        for word, score in similar_words:
            st.write(f"{word}: {score:.4f}")
    else:
        st.write("Word not found in vocabulary.")

# Visualizing Word Embeddings
st.write("Visualizing word embeddings...")
words = list(model.wv.index_to_key)[:200]  # Limit to top 200 words
word_vectors = np.array([model.wv[word] for word in words])  # Convert to NumPy array

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(word_vectors)

df = pd.DataFrame(reduced_vectors, columns=["x", "y"])
df["word"] = words

# Plot with Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["x"], df["y"])

for i, txt in enumerate(df["word"]):
    ax.annotate(txt, (df["x"][i], df["y"][i]), fontsize=8)

st.pyplot(fig)
