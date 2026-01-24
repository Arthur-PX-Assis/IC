import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotar_comparacao_distribuicoes(df):
    print("Gerando gráficos de distribuição...")
    len_original = df['texto_original'].apply(lambda x: len(str(x).split()))
    len_limpo = df['texto_limpo'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 5)) # Configuração do gráfico

    # Gráfico 1: Original
    plt.subplot(1, 2, 1)
    sns.histplot(len_original, bins=30, color='blue', kde=True)
    plt.title('Distribuição de Palavras (Original)')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')

    # Gráfico 2: Limpo
    plt.subplot(1, 2, 2)
    sns.histplot(len_limpo, bins=30, color='green', kde=True)
    plt.title('Distribuição de Palavras (Após Limpeza)')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

def extrair_topicos_dataframe(modelo, nomes_features, n_palavras=10):
    topics_dict = {}
    for topic_idx, topic in enumerate(modelo.components_):
        # Pega os índices das palavras com maior peso
        top_indices = topic.argsort()[:-n_palavras - 1:-1]
        topics_dict[f"Tópico {topic_idx}"] = [nomes_features[i] for i in top_indices]
    return pd.DataFrame(topics_dict)

def exibir_info_bertopic(model, titulo):
    print(f"\n--- Info: {titulo} ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(model.get_topic_info().head(11)) # Imprime as top 10 linhas (+1 pq a linha 0 conta)

def visualizar_bertopic_interativo(model, titulo_geral, top_n_bar=8, n_words_bar=5):
    print(f"\n--- Iniciando Visualização Interativa: {titulo_geral} ---")
    print("Prepare-se: Gráficos serão abertos no seu navegador...")

    # Gráfico de barras
    try:
        print(f"Gerando Barchart para {titulo_geral}...")
        fig_bar = model.visualize_barchart(top_n_topics=top_n_bar, n_words=n_words_bar) # Gera a figura
        fig_bar.update_layout(title_text=f"{titulo_geral} - Top Tópicos (Barras)") # Adiciona um título para saber qual é qual
        fig_bar.show() 
    except Exception as e:
        print(f"Não foi possível gerar o gráfico de barras: {e}")

    # Mapa de distância intertópica
    try:
        print(f"Gerando Mapa de Distância para {titulo_geral}...")
        fig_map = model.visualize_topics()
        fig_map.update_layout(title_text=f"{titulo_geral} - Mapa de Distância")
        fig_map.show()
    except Exception as e:
        print(f"Não foi possível gerar o mapa de distância: {e}")

    print(f"Visualizações de {titulo_geral} enviadas para o navegador.")

def plotar_comparacao_npmi(s1, s2, s3, s4):
    import pandas as pd # Import local caso não esteja no topo
    import matplotlib.pyplot as plt

    print("\nGerando gráfico comparativo de NPMI...")
    
    # Cria df para o gráfico
    resultados = pd.DataFrame({
        'Modelo': ['NMF (5)', 'NMF (10)', 'BERTopic (5)', 'BERTopic (10)'],
        'NPMI Score': [s1, s2, s3, s4]
    })

    # Plota
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'steelblue', 'lightgreen', 'forestgreen']
    bars = plt.bar(resultados['Modelo'], resultados['NPMI Score'], color=colors)

    # Adiciona linha de base no zero
    plt.axhline(0, color='black', linewidth=0.8)

    # Adiciona os valores
    for bar in bars:
        height = bar.get_height()
        offset = -0.015 if height < 0 else 0.005
        plt.text(bar.get_x() + bar.get_width()/2, height + offset, 
                 f'{height:.3f}', ha='center', va='center', fontweight='bold')

    plt.title('Comparação de Coerência (NPMI) - Maior é Melhor', fontsize=14)
    plt.ylabel('Score NPMI')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.show()