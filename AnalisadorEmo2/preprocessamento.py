import spacy
from data_emotions import df_emocoes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

nlp = spacy.load('pt_core_news_md')



glove_file_path = "glove_s300.txt"
len(nlp.vocab)

with open(glove_file_path, "r", encoding="utf-8") as arquivo:
    for linha in arquivo:
        linha = linha.replace('\n', '')
        values = linha.split(' ')
        if len(values) == 301: 
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            nlp.vocab.set_vector(word, vector)
nlp.vocab.vectors.to_disk("vetores_glove")

nlp.vocab.vectors.from_disk("vetores_glove")

len(nlp.vocab)



map_emocoes = {'alegria': 0, 'medo': 1, 'raiva': 2, 'tristeza': 3, 'surpresa': 4, 'nojo': 5}

def vetorizador(texto):
    vetor = nlp(texto).vector
    return vetor


df_emocoes['comentario'] = df_emocoes['comentario'].apply(lambda x : vetorizador(x))
df_emocoes['emocao'] = df_emocoes['emocao'].map(map_emocoes)


#vetorizador.get_feature_names_out()

X = np.stack(df_emocoes['comentario'].to_numpy())

y = df_emocoes['emocao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC()
model.fit(X_train, y_train)

y_pred =  model.predict(X_test)   

accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

coment = 'meu dia foi muito ruim estou triste com isso 👏👏 '

coment = nlp(coment)

for token in coment:
    print(token.text, token.vector[:2])




#save model
import joblib
joblib.dump(model, 'model.pkl')




    