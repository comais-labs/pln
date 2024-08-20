import pandas as pd
from data_emotions import df_emocoes, emojis_para_palavras,emoticons
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import re
from unidecode import unidecode
from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('portuguese'))
nltk.download('rslp')
nltk.download('punkt')


map_emocoes = {'alegria': 0, 'medo': 1, 'raiva': 2, 'tristeza': 3, 'surpresa':4,'nojo':5} 

stem = RSLPStemmer()

#df_emocoes['comentario'] = df_emocoes['comentario'].apply(lambda x: x.lower()) 

def substituir_emoticons(texto):
    for emoticon, significado in emoticons.items():
        texto = texto.replace(emoticon, significado)
    return texto

def substituir_emjis(texto):
    for emojis, significado in emojis_para_palavras.items():
        texto = texto.replace(emojis, significado)
    return texto

        
texto = 'meu dia foi muito bom 😊'

def limpar_texto(texto):
    texto = texto.lower()
    
    texto = substituir_emoticons(texto)
    texto = substituir_emjis(texto)
    #texto = word_tokenize(texto, language='portuguese')
    texto = texto.split()
    texto = [unidecode(palavra) for palavra in texto]
    #texto =  [re.sub(r'[^a-z]',' ',palavra) for palavra in texto]
    texto  = [palavra for palavra in texto if palavra not in stopwords]
    texto = [palavra for palavra in texto if len(palavra) > 2]
    texto = [stem.stem(palavra) for palavra in texto]
    return ' '.join(texto)

df_emocoes['comentario'] = df_emocoes['comentario'].apply(limpar_texto)
df_emocoes['emocao'] = df_emocoes['emocao'].map(map_emocoes)

vetorizador = CountVectorizer(analyzer='word')
vetorizador.fit(df_emocoes['comentario'])

#vetorizador.get_feature_names_out()

X = vetorizador.transform(df_emocoes['comentario'])
y = df_emocoes['emocao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
#model = SVC()
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred =  model.predict(X_test)   
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

coment = 'meu dia foi muito ruim estou triste com isso '

coment = limpar_texto(coment)
coment = vetorizador.transform([coment])
emocao_predita = model.predict(coment)
for emocao, valor in map_emocoes.items():
    if valor == emocao_predita[0]:
        print(emocao)
        break


#save model
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(vetorizador, 'vetorizador.pkl')



    
