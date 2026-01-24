import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def analisar_corpus(lista_textos, titulo):
    series = pd.Series(lista_textos)
    contagem_palavras = series.apply(lambda x: len(str(x).split()))
    contagem_chars = series.apply(lambda x: len(str(x)))

    print(f"--- Análise: {titulo} ---")
    print(f"Média de Palavras: {contagem_palavras.mean():.2f}")
    print(f"Média de Caracteres: {contagem_chars.mean():.2f}")
    print(f"Tamanho Máximo (palavras): {contagem_palavras.max()}")
    print(f"Tamanho Mínimo (palavras): {contagem_palavras.min()}")
    print("-" * 30)
    return contagem_palavras

def limpeza_inicial(df):
    print("Realizando limpeza inicial (segurança)...")
    df = df.dropna(subset=['texto_original']).copy() # Remove linhas vazias
    df['texto_original'] = df['texto_original'].astype(str) # Garante string
    return df

def funcao_preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', '', texto) # Remove pontuação e números
    tokens = texto.split() if isinstance(texto, str) else [] # Tokeniza
    tokens = [word for word in tokens if word not in STOP_WORDS] # Remove stopwords
    return " ".join(tokens) # Reconstrói a frase

def aplicar_preprocessamento(df):
    print("Aplicando pré-processamento (removendo stopwords e pontuação)...")
    df['texto_limpo'] = df['texto_original'].apply(funcao_preprocessar_texto)

    vazios = df[df['texto_limpo'] == ''].shape[0] # Conta os vazios
    print(f"Documentos que ficaram vazios após limpeza: {vazios}")
    
    df = df[df['texto_limpo'].str.strip() != ''].copy() # Remove os vazios
    print(f"Total final de documentos úteis: {len(df)}")
    
    return df