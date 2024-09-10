import spacy
from spacy.tokens import Token, Doc
from data_emotions import emojis_para_palavras 

def check_emoji(token):
    return token.text in emojis_para_palavras.keys()    
 

emojis_getter = lambda token: token.text in emojis_para_palavras.keys()
Token.set_extension("is_emoji", getter=emojis_getter)

nlp = spacy.load('pt_core_news_md')

doc = nlp('que maravilha 😂')

for token in doc:
    print(f"o token: {token.text},é emoji: {token._.is_emoji}")

nlp.pipe_names


from spacy.language import Language

@Language.component("convert_emoji")
def convert_emoji(doc):
    words = [token.text for token in doc]
    for token in doc:
        if token._.is_emoji:
            words[token.i] = emojis_para_palavras[token.text].strip().lower()    
    return Doc(nlp.vocab, words=words)

nlp.add_pipe("convert_emoji", name="convert_emoji",first=True)
doc2 = nlp('que maravilha 😂')

for token in doc2:
    print(token.text)


    

            
    