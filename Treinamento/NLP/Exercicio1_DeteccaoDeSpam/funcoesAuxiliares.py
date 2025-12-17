import matplotlib.pyplot as plt
import seaborn as sns
import string
import os

def preparar_caminho(pasta, nome_arquivo):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    return os.path.join(pasta, nome_arquivo)

# Pre processamento (tudo minusculo, sem pontuacao e sem stop word)
def pre_processar_texto(texto, stop_words):
    texto = texto.lower()
    
    # Remover pontuação
    texto = "".join([char for char in texto if char not in string.punctuation])
    
    # Remover stopwords
    # Quebra o texto em palavras, filtra e junta de volta
    palavras = texto.split()
    palavras_filtradas = [p for p in palavras if p not in stop_words]
    
    return " ".join(palavras_filtradas)

def analise_exploratoria(df):
    # Visão geral dos dados
    print("\n--- Primeiras 5 linhas ---")
    print(df.head())

    print("\n--- Informações Gerais ---")
    print(df.info())

    # Número de documentos (linhas)
    total_docs = len(df)
    print(f"\nNúmero total de documentos (mensagens): {total_docs}")

    # Distribuição de classes (spam vs ham)
    print("\n--- Distribuição das Classes ---")
    print(df['label'].value_counts())

    print("\nPorcentagem:")
    print(df['label'].value_counts(normalize=True) * 100)

    # Plotando a distribuição de classes
    #plt.figure(figsize=(6, 4))
    #sns.countplot(x='label', data=df, palette='viridis')
    #plt.title('Distribuição de Classes (Ham vs Spam)')
    #plt.xlabel('Classe')
    #plt.ylabel('Contagem')
    #plt.show()

    # Análise de Palavras por Documento
    # Criando uma nova coluna 'word_count' contando o número de palavras em cada mensagem
    # Lambda pega linha a linha da coluna message e chama a funcao
    df['word_count'] = df['message'].apply(lambda mensagem_df: len(str(mensagem_df).split()))

    print("\n--- Estatísticas de Palavras por Documento ---")
    print(df['word_count'].describe())

    # Estatísticas separadas por classe
    print("\n--- Estatísticas de Palavras (Média) por Classe ---")
    print(df.groupby('label')['word_count'].describe())

    # Plotando a distribuição do número de palavras
    #plt.figure(figsize=(10, 6))
    #sns.histplot(data=df, x='word_count', hue='label', bins=50, kde=True, palette='bright')
    #plt.title('Distribuição do Número de Palavras por Mensagem')
    #plt.xlabel('Número de Palavras')
    #plt.ylabel('Frequência')
    #plt.xlim(0, 100)
    #plt.show()

    # Visualizando o Antes e Depois
    print("\n--- Antes e Depois ---")
    print(df[['message', 'clean_msg']].head())