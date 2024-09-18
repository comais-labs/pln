from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain.document_loaders import DirectoryLoader, TextLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,  HTMLHeaderTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import Optional, List
import torch
import os
from bs4 import BeautifulSoup
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import BaseModel
from langchain.chains import RetrievalQA


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# Preparando os dados
docs_path = "docs"

if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    with open(os.path.join(docs_path, "documento1.txt"), "w", encoding="utf-8") as f:
        f.write("A inteligência artificial é o campo da ciência da computação que se concentra na criação de máquinas inteligentes que trabalham e reagem como seres humanos.")
    with open(os.path.join(docs_path, "documento2.txt"), "w", encoding="utf-8") as f:
        f.write("O aprendizado de máquina é um subcampo da inteligência artificial que dá às máquinas a habilidade de aprender sem serem explicitamente programadas.")
    with open(os.path.join(docs_path, "documento3.txt"), "w", encoding="utf-8") as f:
        f.write("O aprendizado de máquina é um disciplina ministrada pelo professor Rogério Nogueira de Sousa, o professor tem 42 anos, e trabalha na Universidade Federal do Tocantins.")
loader = DirectoryLoader(
    docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()


# Dividindo os documentos em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap  = 50)
texts = text_splitter.split_documents(documents)

# Configurando as embeddings
embedding_model_name = "rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts"
tokenizer_kwargs = {'clean_up_tokenization_spaces':True}
model_kwargs = {'device': torch_device,'similarity_fn_name': 'cosine', 'tokenizer_kwargs': tokenizer_kwargs}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                   model_kwargs=model_kwargs, 
                                   encode_kwargs=encode_kwargs)
vectorstore = FAISS.from_documents(texts, embeddings)   

model_name = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=torch_device
)

#testando o pipeline
# response = qa_pipeline({
#     'question': "O que é inteligência artificial ?",
#     'context': "A inteligência artificial é o campo da ciência da computação que se concentra na criação de máquinas inteligentes que trabalham e reagem como seres humanos."
# }) 

#definindo um limiar
# threshold = 0.5 # Limiar de similaridade
# if response['score'] < threshold:
#     print("Não foi possível encontrar uma resposta.")
# else:
#     print("Resposta:", response['answer'])
#     print("Score:", response['score'])


#tipo float

class CustomQALLM(LLM, BaseModel):
    pipeline: Any  # O pipeline do HuggingFace
    threshold: float
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # O prompt contém o contexto e a pergunta
        # Precisamos extrair ambos
        try:
            context_part, question_part = prompt.split("Question:")
            context = context_part.replace("Context:", "").strip()
            question = question_part.strip()
        except Exception as e:
            raise ValueError("Formato do prompt inválido. Certifique-se de que o prompt contém 'Context:' e 'Question:'")

        input = {
            'question': question,
            'context': context
        }

        result = self.pipeline(input)
        print(f"Score:{result['score']}")
        if result['score'] < self.threshold:
            return "Não foi possível encontrar uma resposta."
        return result['answer']

    @property
    def _identifying_params(self):
        return {"name": "CustomQALLM"}

    @property
    def _llm_type(self):
        return "custom"


llm = CustomQALLM(pipeline=qa_pipeline, threshold=0.0001)
prompt_template = """Context: {context}
Question: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    prompt=PROMPT
)

response = qa_chain.invoke("qual o nome do professor?")
print(response['result'])
print(f"Pergunta:{response['query']} \nResposta:{response['result']}", "\n", "-"*50)

response = qa_chain.invoke("O que é aprendizado de máquina?")
response['result']
