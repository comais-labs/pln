from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM
import torch

model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Use GPU

prompt = "O que inteligência artificial?"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

output = model.generate( 
    inputs, 
    max_length=50, 
    do_sample=True,
    num_return_sequences=1, 
    no_repeat_ngram_size=2,
    early_stopping=True,
    temperature=0.1,

)

output = tokenizer.decode(output[0], skip_special_tokens=True)
print(output)




from transformers import pipeline

model_name = "csebuetnlp/mT5_multilingual_XLSum"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline('summarization',  
    model=model,
    tokenizer=tokenizer,

)
saida = pipe("""Nesta sexta-feira, 29 de março, a banda Supercombo lançou seu novo álbum de estúdio Adeus, Aurora. O trabalho leva o mesmo nome da revista
 em quadrinhos que o grupo lançou em dezembro de 2018.
As músicas podem ser ouvidas como trilha sonora da HQ,
 e ao mesmo tempo funcionam de forma individual, 
 com temas ligados à nossa sociedade atual.""")

##----

from huggingface_hub import notebook_login
notebook_login()

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

tokenizer = AutoTokenizer.from_pretrained("maritaca-ai/sabia-7b")

model = AutoModelForCausalLM.from_pretrained("maritaca-ai/sabia-7b").to("cuda")

pipe = pipeline("text-generation", 
    model=model, 
    tokenizer="maritaca-ai/sabia-7b",
    model_kwargs={"torch_dtype": torch.bfloat16},
    max_length=100,
    device="cuda", 
    truncation=True
    
)

pipe("O que é inteligência artificial?")
