import streamlit as st
from preprocessamento import limpar_texto, map_emocoes
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

map_emocoes = {'alegria': 0, 'medo': 1, 'raiva': 2, 'tristeza': 3, 'surpresa':4,'nojo':5} 
model = joblib.load('model.pkl')
vetorizador = joblib.load('vetorizador.pkl')
coment = ' meu dia foi muito ruim estou triste com isso '
coment = limpar_texto(coment)
coment = vetorizador.transform([coment])
emocao_predita = model.predict(coment)

for emocao, valor in map_emocoes.items():
    if valor == emocao_predita[0]:
        print(emocao)
        break

st.title("Analisador de Sentimentos")
input_comentario = st.text_input('Digite seu comentário')
if st.button('Analisar'):
    if input_comentario: 
        coment = limpar_texto(input_comentario)
        print(coment)
        coment = vetorizador.transform([coment])
        emocao_predita = model.predict(coment)
        for emocao, valor in map_emocoes.items():
            if valor == emocao_predita[0]:
                st.write(f'Emoção predita: **{emocao}**')
                break
    else:
        st.write('Digite um comentário para ser analisado')
    



