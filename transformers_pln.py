from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#pega o dispositivo que está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "pierreguillou/gpt2-small-portuguese"



# Carregando o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) # modelo de linguagem causal

model.to(device)

#prompt = input("Digite um texto inicial para a geração: ")
prompt = "quem foi o cantor cazuza?"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

output = model.generate(
    input_ids,
    max_length=100,  # comprimento máximo do texto gerado
    num_return_sequences=1,  # número de sequências geradas
    no_repeat_ngram_size=2,  # evita repetição de n-gramas
    early_stopping=True # para a geração quando o modelo para de prever o token de fim de texto
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nTexto Gerado:")
print(generated_text)
# Definindo os parâmetros de geração
temperature = 0.7  # ajustável entre 0 e 1
top_k = 50         # número de palavras a considerar

# Gerando o texto com os novos parâmetros
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
    temperature=temperature,
    top_k=top_k
)

# Decodificando e exibindo o texto gerado
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nTexto Gerado com Parâmetros Ajustados:")
print(generated_text)
