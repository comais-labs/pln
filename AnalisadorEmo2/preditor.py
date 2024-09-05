import streamlit as st
import joblib
import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
import spacy

map_emocoes = {'alegria': 0, 'medo': 1, 'raiva': 2, 'tristeza': 3, 'surpresa':4,'nojo':5} 

model = joblib.load('model.pkl')

nlp = spacy.load('pt_core_news_md')


nlp.vocab.vectors.from_disk("./vetores_glove")


st.title("Analisador de Sentimentos 2.0")
input_comentario = st.text_input('Digite seu comentário')
if st.button('Analisar'):
    if input_comentario: 

        coment = nlp(input_comentario).vector
        print(coment)
        emocao_predita = model.predict([coment])
        for emocao, valor in map_emocoes.items():
            if valor == emocao_predita[0]:
                st.write(f'Emoção predita: **{emocao}**')
                break
    else:
        st.write('Digite um comentário para ser analisado')