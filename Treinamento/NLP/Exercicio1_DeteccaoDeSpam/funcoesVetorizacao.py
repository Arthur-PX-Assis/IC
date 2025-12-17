from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import nltk

_cache_bert = None  # Variável interna para guardar o modelo

def criar_features_tfidf(df):
    # Instanciar o vetorizador
    # Ignora palavras que aparecem em menos de 5 mensagens (ajuda a limpar erros de digitação)
    vectorizer = TfidfVectorizer(min_df=5, max_features=5000) # max_features pra limitar por conta da demora

    # Transformar o texto em números
    # Aprende todas as palavras do seu vocabulário e calcula a matematica TF-IDF para cada msg
    tfidf_matrix = vectorizer.fit_transform(df['clean_msg'])

    # O resultado é uma matriz esparsa (formato comprimido para economizar memória)
    print(f"Matriz Criada: {tfidf_matrix.shape}")

    # Visualizando os pesos de uma mensagem específica
    index_msg = 0
    vetor_mensagem = tfidf_matrix[index_msg]

    # Criando um dataframe temporário só para visualizar bonito
    df_tfidf = pd.DataFrame(
        vetor_mensagem.T.todense(), 
        index=vectorizer.get_feature_names_out(), 
        columns=["tfidf"]
    )

    # Mostrando as palavras com maior peso nessa mensagem (Top 5)
    # print(f"\n--- Palavras mais importantes da mensagem {index_msg} ---")
    # print(f"Texto original: {df.iloc[index_msg]['clean_msg']}")
    # print(df_tfidf[df_tfidf['tfidf'] > 0].sort_values(by=["tfidf"], ascending=False).head())

    return tfidf_matrix

def criar_features_word2vec(df, tamanho_vetor=100):
    # Garantir que o tokenizer está baixado
    nltk.download('punkt', quiet=True)

    print("Treinando modelo Word2Vec...")
    
    # O gensim precisa de uma lista de listas de palavras
    frases_tokenizadas = [frase.split() for frase in df['clean_msg']]
    
    # Treinando o modelo
    modelo_w2v = Word2Vec(sentences=frases_tokenizadas,
                          window=5,                     # Quantas palavras vizinhas ele olha
                          min_count=1,                  # Ignora palavras que aparecem menos de X vezes
                          workers=4)
    
    # Transformar frases em média de vetores
    def vetorizar_frase(frase):
        palavras = frase.split()
        vetores_palavras = []
        
        for palavra in palavras:
            if palavra in modelo_w2v.wv:
                vetores_palavras.append(modelo_w2v.wv[palavra])
        
        if len(vetores_palavras) > 0:
            # Tira a média dos vetores das palavras encontradas
            return np.mean(vetores_palavras, axis=0)
        else:
            # Se a frase não tiver nenhuma palavra conhecida, retorna vetor de zeros
            return np.zeros(tamanho_vetor)

    # Aplica em todo o dataframe
    array_vetores = np.array([vetorizar_frase(frase) for frase in df['clean_msg']])
    
    return array_vetores

def criar_features_fasttext(df, tamanho_vetor=100):
    print("Treinando modelo FastText...")
    
    # Prepara as frases (lista de listas de palavras)
    frases_tokenizadas = [frase.split() for frase in df['clean_msg']]
    
    # Treinando o modelo
    
    modelo_ft = FastText(sentences=frases_tokenizadas, 
                         vector_size=tamanho_vetor,
                         window=5, 
                         min_count=1,               # Aprende até palavras que apareceram uma única vez
                         workers=4,
                         min_n=3,                   # Tamanho dos pedacinhos
                         max_n=6)                   # n-gram padrão é 3 a 6 letras
    
    def vetorizar_frase(frase):
        palavras = frase.split()
        vetores_palavras = []
        
        for palavra in palavras:
            # FastText sempre devolve um vetor mesmo que a palavra seja nova (ele monta pelos pedaços).
            # Por isso não precisa do "if palavra in modelo.wv"
            try:
                vetores_palavras.append(modelo_ft.wv[palavra])
            except KeyError:
                pass # Só por segurança extrema
        
        if len(vetores_palavras) > 0:
            return np.mean(vetores_palavras, axis=0)
        else:
            return np.zeros(tamanho_vetor)

    # Aplica em todo o dataframe
    array_vetores = np.array([vetorizar_frase(frase) for frase in df['clean_msg']])
    
    return array_vetores

def carregar_bert():
    global _cache_bert  # Avisa que vamos mexer na variável lá de cima

    # Verifica se o cofre já tem o modelo guardado
    if _cache_bert is not None:
        print("Modelo BERT recuperado da memória (Cache).")
        # Desempacota o que está guardado e retorna
        return _cache_bert

    # Se o cofre estiver vazio, faz o carregamento pesado (só roda na 1ª vez)
    print("\n--- Inicializando DistilBERT (Carregando do disco/internet) ---")
    
    # Define o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo selecionado: {device}")
    
    # Carrega pré-treinado
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Envia o modelo para a GPU
    model.to(device)
    
    # Guarda tudo no cofre para a próxima vez
    _cache_bert = (model, tokenizer, device)
    
    return model, tokenizer, device

def criar_features_bert(df, model, tokenizer, device, batch_size=100):
    print("Gerando representações vetoriais...")
    
    mensagens = df['clean_msg'].tolist()
    todos_vetores = []

    # O modelo já deve estar em .eval() para extração (desliga dropout, etc)
    model.eval()

    for i in tqdm(range(0, len(mensagens), batch_size), desc="Processando"):
        batch_textos = mensagens[i : i + batch_size]
        
        # Tokenização
        inputs = tokenizer(batch_textos, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # IMPORTANTE: Mover os inputs para a mesma GPU do modelo
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        vetores_medios = last_hidden_states.mean(dim=1)
        
        # Traz de volta para CPU para salvar no numpy (RAM normal)
        todos_vetores.append(vetores_medios.cpu().numpy())

    X_bert = np.vstack(todos_vetores)
    print(f"Dimensão gerada: {X_bert.shape}")
    
    return X_bert