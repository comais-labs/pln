import spacy
import numpy as np

nlp =spacy.blank("pt")

words = []
vectors = []

with open("glove_s300.txt", "r", encoding="utf-8") as arquivo:
    for linha in arquivo:
        linha = linha.replace('\n', '')
        values = linha.split(' ')
        if len(values) == 301: 
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            nlp.vocab.set_vector(word, vector)  

nlp.vocab.vectors.to_disk("RepresentacaoVetorial")
nlp.vocab.vectors.from_disk("RepresentacaoVetorial")
word1 = nlp("lula")
word2 = nlp("presidente")
word3 = nlp("vereador")
word4 = nlp("gerente")
word5 = nlp("rogério")
word6 = nlp("dilma")
word7 = nlp("lava jato")


vectors = [nlp("lula").vector,
nlp("presidente").vector,
nlp("vereador").vector,
nlp("gerente").vector,
nlp("rogério").vector,
nlp("dilma").vector,
nlp("lava jato").vector]

words = [word1.text, word2.text, word3.text, word4.text, word5.text, word6.text, word7.text]


word1.similarity(word6)

vector_lula = np.asarray([nlp.vocab["lula"].vector])
proximos_lula = nlp.vocab.vectors.most_similar(vector_lula, n=30)
hash_proximos = proximos_lula[0][0]

[nlp.vocab.strings[prox] for prox in hash_proximos]

# gerar grafico de similaridade
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

import plotly.express as px

pca = PCA(n_components=2)
X_pca = pca.fit_transform(vectors)

# converter para um dataframe

df = pd.DataFrame(X_pca, columns=["x", "y"])

df.index = words

# Plotar os vetores
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    text=df.index
))

fig.show()
