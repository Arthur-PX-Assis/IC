import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import testeT_Bonferroni
import arvoreDecisao
import gridSearch
import tf_idf
import knn
import svm


# Pre processamento (tudo minusculo, sem pontuacao e sem stop word)
def pre_processar_texto(texto, stop_words):
    texto = texto.lower()
    
    # Remover pontuação
    texto = "".join([char for char in texto if char not in string.punctuation])
    
    # Remover Stopwords
    # Quebramos o texto em palavras, filtramos e juntamos de volta
    palavras = texto.split()
    palavras_filtradas = [p for p in palavras if p not in stop_words]
    
    return " ".join(palavras_filtradas)

# Tecninca de Holdout (dividir em treino e teste)
def divisao_treino_teste(df):
    # Definindo quem é X (Dados) e quem é y (Respostas)
    X = tf_idf.TF_IDF(df)

    # y: A coluna com a classificação real (ham/spam)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    print(f"\nClasses convertidas: {le.classes_}")

    # Dividindo os dados
    # test_size=0.2: Separa 20% dos dados para testar depois (prova final)
    # random_state=42: Garante que a divisão seja sempre a mesma (para seus resultados não mudarem a cada execução)
    return train_test_split(X, y, test_size=0.2, random_state=42)

caminho_arquivo = 'smsspamcollection/SMSSpamCollection'

try:
    df = pd.read_csv(caminho_arquivo, sep='\t', header=None, names=['label', 'message'])
    print("Base de dados carregada com sucesso!")
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado no caminho: {caminho_arquivo}")
    exit()

# Visão Geral dos Dados
print("\n--- Primeiras 5 linhas ---")
print(df.head())

print("\n--- Informações Gerais ---")
print(df.info())

# Análise Inicial: Número de Documentos (Linhas)
total_docs = len(df)
print(f"\nNúmero total de documentos (mensagens): {total_docs}")

# Distribuição de Classes (Spam vs Ham)
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

nltk.download("stopwords")

# Definindo as Stopwords (palavras vazias) em inglês (linguagem do dataset)
stop_words = set(stopwords.words('english'))

# Aplicando a limpeza
# Criamos uma nova coluna 'clean_msg' para comparar com a original
df['clean_msg'] = df['message'].apply(lambda mensagem_df: pre_processar_texto(mensagem_df, stop_words))

# Visualizando o Antes e Depois
print("--- Antes e Depois ---")
print(df[['message', 'clean_msg']].head())

X_train, X_test, y_train, y_test = divisao_treino_teste(df)

knn.KNN(X_train, y_train, X_test, y_test)
svm.SVM(X_train, y_train, X_test, y_test)
arvoreDecisao.arvore_decisao(X_train, y_train, X_test, y_test)

melhores_parametros = gridSearch.rodar_grid(X_train, y_train, X_test, y_test)
testeT_Bonferroni.teste_estatistico(X_train, y_train, melhores_parametros)