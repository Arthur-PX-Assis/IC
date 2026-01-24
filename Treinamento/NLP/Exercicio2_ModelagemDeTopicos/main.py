# Adicione o novo import no topo
from src.dataset import carregar_dados
from src.evaluation import avaliar_todos_modelos
from src.models import gerar_tfidf, treinar_nmf, gerar_embeddings, treinar_bertopic
from src.preprocessing import analisar_corpus, limpeza_inicial, aplicar_preprocessamento
from src.visualization import plotar_comparacao_distribuicoes, extrair_topicos_dataframe, exibir_info_bertopic, visualizar_bertopic_interativo, plotar_comparacao_npmi

CAMINHO_ZIP = "./data/TREC.zip"
PASTA_DESTINO = "./data/TREC"

def main():

    # =====================================================================================

    print(">>> ETAPA 1: CARREGAMENTO")

    df = carregar_dados(CAMINHO_ZIP, PASTA_DESTINO)
    df = limpeza_inicial(df)

    # =====================================================================================

    print("\n>>> ETAPA 2: PRÉ-PROCESSAMENTO")

    df = aplicar_preprocessamento(df)

    # =====================================================================================

    print("\n>>> ETAPA 3: ANÁLISE ESTATÍSTICA")

    analisar_corpus(df['texto_original'], "TEXTO ORIGINAL")
    analisar_corpus(df['texto_limpo'], "TEXTO PRÉ-PROCESSADO")

    # =====================================================================================

    print("\n>>> ETAPA 4: VISUALIZAÇÃO")

    plotar_comparacao_distribuicoes(df)
    print("\nFim da etapa de pré-processamento e análise.")

    # =====================================================================================

    print("\n>>> ETAPA 5: MODELAGEM NMF")

    tfidf_matrix, vectorizer, feature_names = gerar_tfidf(df['texto_limpo'])
    
    print("\n--- Experimento NMF (5 Tópicos) ---")

    nmf_5 = treinar_nmf(tfidf_matrix, n_topicos=5)
    df_topicos_5 = extrair_topicos_dataframe(nmf_5, feature_names, 10)

    print(df_topicos_5)

    print("\n--- Experimento NMF (10 Tópicos) ---")

    nmf_10 = treinar_nmf(tfidf_matrix, n_topicos=10)
    df_topicos_10 = extrair_topicos_dataframe(nmf_10, feature_names, 10)

    print(df_topicos_10)
    print("\nFim do treinamento NMF.")

    # =====================================================================================

    print("\n>>> ETAPA 6: MODELAGEM BERTOPIC")
    
    embeddings, sentence_model = gerar_embeddings(df['texto_original'].tolist())
    
    print("\n--- Experimento BERTopic (5 Tópicos) ---")

    bert_5 = treinar_bertopic(df['texto_original'].tolist(), embeddings, sentence_model, n_topicos=5)

    exibir_info_bertopic(bert_5, "BERTopic 5 Tópicos")
    visualizar_bertopic_interativo(bert_5, "BERTopic (5)", top_n_bar=5, n_words_bar=5) # Visualiza 5 tópicos, com 5 palavras cada

    print("\n--- Experimento BERTopic (10 Tópicos) ---")

    bert_10 = treinar_bertopic(df['texto_original'].tolist(), embeddings, sentence_model, n_topicos=10)

    exibir_info_bertopic(bert_10, "BERTopic 10 Tópicos")
    visualizar_bertopic_interativo(bert_10, "BERTopic (10)", top_n_bar=10, n_words_bar=10)
    print("\nFim do treinamento e visualização dos modelos.")

    # =====================================================================================

    print("\n>>> ETAPA 7: AVALIAÇÃO E COMPARAÇÃO")
    
    s_nmf5, s_nmf10, s_bert5, s_bert10 = avaliar_todos_modelos(
        df, 
        nmf_5, nmf_10, 
        bert_5, bert_10, 
        feature_names
    )
    
    plotar_comparacao_npmi(s_nmf5, s_nmf10, s_bert5, s_bert10)
    print("\n=== PROJETO CONCLUÍDO COM SUCESSO! ===")

if __name__ == "__main__":
    main()