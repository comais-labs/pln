
# Importações
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from typing import Optional, List
import os

# Preparando os dados
docs_path = "docs"

if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    with open(os.path.join(docs_path, "documento1.txt"), "w", encoding="utf-8") as f:
        f.write("A inteligência artificial é o campo da ciência da computação que se concentra na criação de máquinas inteligentes que trabalham e reagem como seres humanos.")
    with open(os.path.join(docs_path, "documento2.txt"), "w", encoding="utf-8") as f:
        f.write("O aprendizado de máquina é um subcampo da inteligência artificial que dá às máquinas a habilidade de aprender sem serem explicitamente programadas.")

# Carregando os documentos
loader = DirectoryLoader(
    docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

# Dividindo os documentos em pedaços menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Configurando as embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Criando o vetor de índice
vectorstore = FAISS.from_documents(texts, embeddings)

# Configurando o modelo de linguagem
model_name = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

# Criando uma classe LLM personalizada
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import BaseModel

class CustomQALLM(LLM, BaseModel):
    pipeline: Any  # O pipeline do HuggingFace

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
        print(result['score'])
        return result['answer']

    @property
    def _identifying_params(self):
        return {"name": "CustomQALLM"}

    @property
    def _llm_type(self):
        return "custom"

# Criando o LLM personalizado
custom_llm = CustomQALLM(pipeline=qa_pipeline)

# Definindo o prompt template
prompt_template = """Context: {context}

Question: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Configurando o chain
from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(
    llm=custom_llm,
    chain_type="stuff",
    prompt=PROMPT
)

# Criando o retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Função para responder perguntas
def answer_question(question):
    # Recupera documentos relevantes
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Informação não disponível nos documentos."

    # Executa o chain com os documentos e a pergunta
    result = qa_chain.run(
        input_documents=docs,
        question=question
    )
    return result

# Teste do sistema
question = "O que é aprendizado de máquina?"
response = answer_question(question)
print("Pergunta:")
print(question)
print("\nResposta:")
print(response)

# Pergunta sem resposta nos documentos
question = "Qual é a capital da França?"
response = answer_question(question)
print("\nPergunta:")
print(question)
print("\nResposta:")
print(response)
