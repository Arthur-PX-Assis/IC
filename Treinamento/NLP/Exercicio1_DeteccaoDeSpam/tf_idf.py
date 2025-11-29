from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def TF_IDF(df):
    # Instanciar o Vetorizador
    # Ignora palavras que aparecem em menos de 5 mensagens (ajuda a limpar erros de digitação)
    vectorizer = TfidfVectorizer(min_df=5)

    # Transformar o Texto em Números
    # Aprende todas as palavras do seu vocabulário e calcula a matematica TF-IDF para cada msg
    tfidf_matrix = vectorizer.fit_transform(df['clean_msg'])

    # O resultado é uma matriz esparsa (formato comprimido para economizar memória)
    print(f"Matriz Criada: {tfidf_matrix.shape}")

    # Visualizando os Pesos de uma Mensagem Específica
    index_msg = 0
    vetor_mensagem = tfidf_matrix[index_msg]

    # Criando um DataFrame temporário só para visualizar bonito
    df_tfidf = pd.DataFrame(
        vetor_mensagem.T.todense(), 
        index=vectorizer.get_feature_names_out(), 
        columns=["tfidf"]
    )

    # Mostrando as palavras com maior peso nessa mensagem (Top 5)
    print(f"\n--- Palavras mais importantes da mensagem {index_msg} ---")
    print(f"Texto original: {df.iloc[index_msg]['clean_msg']}")
    print(df_tfidf[df_tfidf['tfidf'] > 0].sort_values(by=["tfidf"], ascending=False).head())

    return tfidf_matrix