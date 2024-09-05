import spacy
from spacy.tokens import Token
#

fruit_getter = lambda token: token.text in ("apple", "pear", "banana")
Token.set_extension("e_fruta", getter=fruit_getter)
    

nlp = spacy.load('pt_core_news_md')
#new atttribute to the token

doc = nlp('apple is a fruit')


nlp.pipe_names
#disable pipe
nlp.disable_pipes('ner')
#enable pipe
nlp.enable_pipe('ner')
nlp.pipe_names

doc = nlp('O rato roeu a roupa do rei de Roma')
doc.ents[0].label_

#Create a pipe

from spacy.language import Language
@Language.component("custom_component")
def custom_component(doc):
    if doc[0]._.e_fruta == True:
        print("First token is a fruit!")

    return doc 
  
nlp.add_pipe("custom_component", name="custom_component",  before='lemmatizer')

nlp('roupa do rei de Roma')

nlp('apple is a fruit')