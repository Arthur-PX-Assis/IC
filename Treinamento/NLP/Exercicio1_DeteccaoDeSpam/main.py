# Bibliotecas externas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from datasets import load_dataset

import pandas as pd
import nltk

# Meus módulos
import funcoesVetorizacao as vet
import funcoesAuxiliares as aux
import funcoesFinetuning as ft
import funcoesModelos as mod
import funcoesAnalise as ana

def divisao_treino_teste(df, tecnica_representacao):
    # y: A coluna com a classificação real (ham/spam)
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    textos_todos = df['clean_msg'].tolist()
    print(f"\nClasses convertidas: {le.classes_}")

    # Definindo quem é X (dados) e quem é y (respostas)
    if tecnica_representacao == "tfidf":
        print(f"\n--- Gerando representação: TF-IDF ---")
        X = vet.criar_features_tfidf(df)

    elif tecnica_representacao == "word2vec":
        print(f"\n--- Gerando representação: Word2Vec ---")
        X = vet.criar_features_word2vec(df)

    elif tecnica_representacao == "fasttext":
        print(f"\n--- Gerando representação: FastText ---")
        X = vet.criar_features_fasttext(df)

    elif tecnica_representacao == "bert":
        print(f"\n--- Gerando representação: BERT ---")
        modelo_bert, tokenizer_bert, device_bert = vet.carregar_bert()
        X = vet.criar_features_bert(df, modelo_bert, tokenizer_bert, device_bert)

    elif tecnica_representacao == "bert_finetuned":
        print(f"\n--- Preparando Fine-Tuning do BERT ---")

        # Dividie o texto puro primeiro (antes de vetorizar)
        # Isso garante que o fine-tuning nunca veja os dados de teste
        textos_train, textos_test, y_train, y_test = train_test_split(
            textos_todos, y, test_size=0.2, random_state=42
        )

        # Treina apenas com os textos de treino
        modelo_ft, tokenizer_ft, device_ft = ft.realizar_finetuning(textos_train, y_train)

        # Gera os vetores para o treino e para o teste usando o modelo treinado
        print("Gerando vetores para Treino...")
        X_train = ft.gerar_vetores_finetuned(textos_train, modelo_ft, tokenizer_ft, device_ft)

        print("Gerando vetores para Teste...")
        X_test = ft.gerar_vetores_finetuned(textos_test, modelo_ft, tokenizer_ft, device_ft)

        # Retorna direto (pois já dividiu)
        return X_train, X_test, y_train, y_test

    else:
        raise ValueError(f"Técnica '{tecnica_representacao}' não reconhecida. Use 'tfidf', 'word2vec' ou 'fasttext'.")

    # Dividindo os dados
    # test_size=0.2: Separa 20% dos dados para testar depois (prova final)
    # random_state=42: Garante que a divisão seja sempre a mesma (para seus resultados não mudarem a cada execução)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def rodar_tecnica_representacao(df, tecnica_representacao, pasta_resultado):

    print(f"\n--- INICIANDO CICLO COM {tecnica_representacao.upper()} ---")

    if tecnica_representacao == "tfidf":
        caminho_escrita = aux.preparar_caminho(pasta_resultado, "resultadoTfidf.txt")

    elif tecnica_representacao == "word2vec":
        caminho_escrita = aux.preparar_caminho(pasta_resultado, "resultadoWord2Vec.txt")

    elif tecnica_representacao == "fasttext":
        caminho_escrita = aux.preparar_caminho(pasta_resultado, "resultadoFastText.txt")

    elif tecnica_representacao == "bert":
        caminho_escrita = aux.preparar_caminho(pasta_resultado, "resultadoBert.txt")

    elif tecnica_representacao == "bert_finetuned":
        caminho_escrita = aux.preparar_caminho(pasta_resultado, "resultadoBertFineTuned.txt")

    else:
        raise ValueError(f"Técnica '{tecnica_representacao}' não reconhecida. Use 'tfidf', 'word2vec', 'fasttext' ou 'bert'.")

    X_train, X_test, y_train, y_test = divisao_treino_teste(df, tecnica_representacao)

    with open(caminho_escrita, 'w', encoding='utf-8') as f:
        pass

    mod.KNN(X_train, y_train, X_test, y_test, caminho_escrita)
    mod.SVM(X_train, y_train, X_test, y_test, caminho_escrita)
    mod.arvore_decisao(X_train, y_train, X_test, y_test, caminho_escrita)

    melhores_parametros = ana.rodar_grid(X_train, y_train, X_test, y_test, caminho_escrita)
    ana.teste_estatistico(X_train, y_train, melhores_parametros, caminho_escrita)

if __name__ == "__main__":
    # Carrega
    # df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    # pasta_resultado = "resultadosSMSSpamCollection"

    dataset_dict = load_dataset("imdb")
    df = dataset_dict['train'].to_pandas()
    pasta_resultado = "resultadosIMDB"

    # Visualiza como ele veio (para descobrir os nomes)
    print("Colunas originais:", df.columns)
    print(df.head())

    # Renomear para o padrão do projeto ('message' e 'label')
    # Supondo que as colunas originais sejam 'text' e 'label'
    # Se forem diferentes basta ajustar
    df = df.rename(columns={
        'text': 'message',   # Renomeia o texto para 'message'
        'label': 'label'     # 'label' geralmente já vem certo, mas garante
    })

    # Garante só o que precisa
    df = df[['label', 'message']]

    # Limpa
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words('english'))

    df['clean_msg'] = df['message'].apply(lambda x: aux.pre_processar_texto(x, stop_words))

    print("Dados prontos!")

    # Análise dos dados
    aux.analise_exploratoria(df)

    rodar_tecnica_representacao(df, 'tfidf', pasta_resultado)
    rodar_tecnica_representacao(df, 'word2vec', pasta_resultado)
    rodar_tecnica_representacao(df, 'fasttext', pasta_resultado)
    rodar_tecnica_representacao(df, 'bert', pasta_resultado)
    rodar_tecnica_representacao(df, 'bert_finetuned', pasta_resultado)