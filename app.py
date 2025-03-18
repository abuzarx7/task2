import streamlit as st
#!/usr/bin/env python
# coding: utf-8

# In[52]:

import streamlit as st

import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# In[7]:


corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum() and word]
    corpus_sentences.append(tokenized_sentence)
st.write(f"Number of sentences in the Reuters corpus: {len(corpus_sentences)}")


# In[9]:


model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
# Print vocabulary size
st.write(f"Vocabulary size: {len(model.wv.index_to_key)}")


# In[11]:


import numpy as np
# Extract the learned word vectors and their corresponding words for visualization.
words = list(model.wv.index_to_key)[:200]  # Limit to top 200 words for better visualization
word_vectors = np.array([model.wv[word] for word in words])  # Convert to NumPy array for compatible 


# In[13]:


# Use t-SNE to project the high-dimensional word embeddings into a 2D space.
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
word_vectors_2d = tsne.fit_transform(word_vectors)


# In[15]:


# Plot the 2D t-SNE visualization of the word embeddings with their labels.
def plot_embeddings(vectors, labels):
    plt.figure(figsize=(16, 12))
    for i, label in enumerate(labels):
        x, y = vectors[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.1, y + 0.1, label, fontsize=9)
    plt.title("Word2Vec Embeddings Visualized with t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    st.pyplot()
plot_embeddings(word_vectors_2d, words)


# 
# --------------------------
# 
# key question for student:
# 
# 
# 
# ques: What do you observe about the clusters in the t-SNE plot?
# 
# ans:
#      
#      words that have similar meaning or words that are generally being used in a same bracket  for eg months like jaunauary 
#      february are in one clusters
# 
# 
# 
# ques:How do you think the choice of parameters (e.g., window size, vector size) affects the
# embeddings?
# 
# ans: 
# 
#      small vector size creates compact embeddings
# 
#      increeasing num of epoh can increase embedding but might lead to overfitting 
#      
# 
# 
# ques: What are the limitations of using Word2Vec and t-SNE for NLP tasks?     
#       
# ans: 
# 
#     word2vec has typos putting words that arent there in the vocab 
# 
#      word2vec needs a substranital amount of texts to generate meaningfull embeddings  
#      
#      t-sne is hard to  hard to interpet visualsization of tsne the 2d and 3d graphs 
# 
#      t=sne depends on parameters like learning rate which if chosen incoreectly could give a abrupt results 
# 
# 
#      

# In[40]:


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]



# In[47]:


def get_average_embedding(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  

def compute_document_embeddings(corpus_sentences, model):
    document_embeddings = []
    for sentence in corpus_sentences:
        embedding = get_average_embedding(sentence, model)
        document_embeddings.append(embedding)
    return document_embeddings


# In[49]:


def find_top_n_documents(query, document_embeddings, corpus_sentences, model, N=5):
    query_tokens = preprocess_text(query)
    query_embedding = get_average_embedding(query_tokens, model)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-N:][::-1]
    return [(reuters.fileids()[idx], similarities[idx], corpus_sentences[idx][:10]) for idx in top_indices]


nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

corpus_sentences = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    corpus_sentences.append(preprocess_text(raw_text))


# In[54]:


model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

document_embeddings = compute_document_embeddings(corpus_sentences, model)

query = "stock market performance"
N = 5
results = find_top_n_documents(query, document_embeddings, corpus_sentences, model, N)


st.write(f"Top {N} most relevant documents for query '{query}':\n")
for doc_id, similarity, preview in results:
    preview_text = ' '.join(preview) + "..."
    st.write(f"Document ID: {doc_id}")
    st.write(f"Similarity Score: {similarity:.4f}")
    st.write(f"Document Preview: {preview_text}\n")


# In[1]:


get_ipython().system('jupyter nbconvert --to script lab2.ipynb')


# In[ ]:




