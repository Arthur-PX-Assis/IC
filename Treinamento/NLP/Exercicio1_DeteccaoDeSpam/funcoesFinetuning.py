from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

# Classe necessária para o pyTorch entender os dados
class SpamDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def realizar_finetuning(textos_treino, labels_treino):
    print("\n--- INICIANDO FINE-TUNING DO BERT ---")
    print("Isso vai demorar! Certifique-se de estar usando GPU (T4) no Colab.")

    # Configuração
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenização dos dados de treino
    print("Tokenizando dados de treino...")
    train_encodings = tokenizer(textos_treino, truncation=True, padding=True, max_length=128)
    
    # Cria um dataset de validação interno (20% do treino) para saber quando parar
    # Converte labels para lista se forem series do pandas
    labels_treino = list(labels_treino)
    
    # Criar o dataset do pyTorch
    full_dataset = SpamDataset(train_encodings, labels_treino)
    
    # Dividir em treino real e validação para o BERT
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Carregar o modelo
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    model.train()

    # Configurar o treinamento
    training_args = TrainingArguments(
        output_dir='./results_finetuning',
        num_train_epochs=2,              # 2 épocas é suficiente para não viciar (overfitting)
        per_device_train_batch_size=16,  # Batch pequeno para não estourar memória
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no"               # Não salvar checkpoints intermediários para economizar espaço
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Rodar o treino
    trainer.train()
    
    print("Fine-Tuning concluído! Modelo ajustado está na memória.")
    return model, tokenizer, device

def gerar_vetores_finetuned(textos, model, tokenizer, device, batch_size=64):
    print("Gerando representações com o modelo ajustado...")
    model.eval() # Modo de avaliação
    
    todos_vetores = []

    for i in tqdm(range(0, len(textos), batch_size), desc="Extraindo"):
        batch = textos[i : i + batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Pega 'output_hidden_states=True' para extrair o vetor, não a classificação
            outputs = model(**inputs, output_hidden_states=True)
        
        # Pegar o vetor do token [CLS] da última camada
        # hidden_states[-1] é a última camada. [:, 0, :] pega o primeiro token de todas as frases.
        cls_embeddings = outputs.hidden_states[-1][:, 0, :]
        
        todos_vetores.append(cls_embeddings.cpu().numpy())

    return np.vstack(todos_vetores)