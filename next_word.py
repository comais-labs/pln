import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Configurações básicas
nlp = spacy.load("pt_core_news_md")


# Configurações básicas torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")

# Exemplo simples de dataset

#corpus = df_emocoes['comentario'].values
corpus = [
    "eu gosto de aprender",
    "eu gosto de programação",
    "eu gosto de pizza",
    "eu gosto de python",
    "eu gosto de redes neurais",
    "gosto de assistir filmes",
    "filmes de ficção científica são legais"
]

docs = nlp(', '.join(corpus))

# Criação de um vocabulário e mapeamento de índices usando spacy
docs_vocab = set([ token.text for token in docs])

{ word:i for i, word in enumerate(docs_vocab)}

word2idx = {word: i for i, word in enumerate(docs_vocab)}
idx2word = {i:word for i, word in enumerate(docs_vocab)}

# Criando pares de sequência de entrada (x) e palavra de saída (y)
def create_sequences(corpus, word2idx):
    sequences = []
    for sentence in corpus:
        tokens = [token.text for token in nlp(sentence)]
        for i in range(1, len(tokens)):
            sequence = tokens[:i]
            sequence = [word2idx[word] for word in sequence]
            target = tokens[i]
            target = word2idx[target]
            sequences.append((sequence, target))
    return sequences

sequences = create_sequences(corpus, word2idx) 

# Criação de um Dataset PyTorch personalizado
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence), torch.tensor(target) #retorna dois tensores pytorch
    

train_dataset, test_dataset = train_test_split(sequences, test_size=0.2)
train_dataset = TextDataset(train_dataset)

test_dataset  = TextDataset(test_dataset)  

#train_dataset = TextDataset(sequences) #Addiciona todos exemplos ao treinamento

batch_size = 2 # lote de tamanho 1, processamento em paralelo

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# Dataloader para criar mini-lotes de dados	
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definição do Modelo de Linguagem contendo uma camada de embedding, uma camada LSTM e uma camada linear
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])# o retorno da camada LSTM é uma tupla contendo a saída e o estado oculto, mas estamos interessados apenas na saída
        return x
 
vocab_size = len(docs_vocab)# Tamanho do vocabulário
embed_size = 10 # Define o tamanho dos vetores de embedding. Cada palavra será representada por um vetor de 10 dimensões, o que ajuda a capturar a semântica das palavras no espaço vetorial.
hidden_size = 20 # Define o tamanho da camada oculta no LSTM. Este valor indica quantas unidades ocultas (neurônios) estarão presentes na camada LSTM, influenciando a capacidade do modelo de aprender padrões complexos.
learning_rate = 0.01 # Define a taxa de aprendizado para o otimizador. Esse valor controla o quão grandes serão os passos dados durante a atualização dos pesos do modelo, impactando a velocidade e a estabilidade do treinamento.
num_epochs = 100 # Define o número de épocas, ou seja, quantas vezes o modelo verá todo o conjunto de treinamento. Um maior número de épocas pode melhorar o aprendizado, mas também aumenta o risco de overfitting.

# Inicialização do modelo, critério de perda e otimizador
model = LanguageModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()# Função de perda de entropia cruzada, que é comumente usada para problemas de classificação multiclasse.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)# Otimizador Adam, que é uma variação do gradiente descendente estocástico que se adapta automaticamente a diferentes taxas de aprendizado para cada parâmetro do modelo.

# Função de treinamento
def train_model(model,criterion, train_data_loader,optimizer, num_epochs):
    model.train()
    epoch_loss = []
    for epoch in range(num_epochs):
        for sequence, target in train_data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            output = model(sequence)
            loss = criterion(output, target)
            optimizer.zero_grad()# Zera os gradientes acumulados
            loss.backward()# Calcula os gradientes
            optimizer.step()# Atualiza os pesos do modelo
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            epoch_loss.append((epoch+1,loss.item()))# Salva o valor da perda para cada época  
    return epoch_loss

                
#train_model(model, criterion, train_data_loader, optimizer, num_epochs)

# Avaliação do modelo
def evaluate_model(model,test_data_loader):
    model.eval()
    with torch.no_grad():# Desativa o cálculo de gradientes durante a avaliação
        total_loss = 0
        for sequence, target in test_data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            output = model(sequence)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_data_loader)
    print(f"Avaliação: Loss médio: {avg_loss:.4f}")
    print(f"Output: {output}" )
      
#evaluate_model(model, test_data_loader)

# Teste de geração de texto

def generete_text( model, start_text, word2idx, idx2word, max_len=5):
    model.eval()
    words = [token.text for token in nlp(start_text)]
    vector = [word2idx[word] for word in words]
    input_seq = torch.tensor(vector, dtype=torch.long).unsqueeze(0)
    for _ in range(max_len):
        saida = model(input_seq).argmax(dim=1).item()
        next_word = idx2word[saida]
        words.append(next_word)
        input_seq = torch.cat([input_seq, torch.tensor([[saida]], dtype=torch.long)], dim=1)
    return ' '.join(words)
start_text = "assistir"
generete_text(model, "assistir", word2idx, idx2word, max_len=5)

# avaliar o processo de aprendizado
import matplotlib.pyplot as plt
epoch_loss = train_model(model, criterion, train_data_loader, optimizer, num_epochs)
epochs, losses = zip(*epoch_loss)
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
