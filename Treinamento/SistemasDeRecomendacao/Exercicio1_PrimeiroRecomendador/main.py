from src.recommender import MostPopular, ContentBasedRecommender, CollaborativeRecommender
from src.analysis import DatasetAnalyzer
from src.visualizer import Visualizer
from src.metrics import RankEvaluator
from src.loader import DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import traceback
import time

def evaluate_model(model, test_data, k=10):
    evaluator = RankEvaluator()
    metrics = {'precision': [], 'recall': [], 'ndcg': []}   
    test_ground_truth = test_data.groupby('userId')['movieId'].apply(list) # Agrupa o gabarito (o que o usuário viu no conjunto de teste)
    
    for user_id, relevant_items in test_ground_truth.items():
        try:
            recs_df = model.recommend(user_id, n=k, remove_seen=True) # Pede recomendação ao modelo (remove itens vistos no treino)
            
            if recs_df.empty:
                continue

            recommended_ids = recs_df['movieId'].values.tolist()
            
            # Calcula as métricas
            p = evaluator.precision_at_k(recommended_ids, relevant_items, k)
            r = evaluator.recall_at_k(recommended_ids, relevant_items, k)
            ndcg = evaluator.ndcg_at_k(recommended_ids, relevant_items, k)
            
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['ndcg'].append(ndcg)
            
        except Exception:
            continue
            
    # Retorna a média das métricas
    return {
        'Precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
        'Recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
        'NDCG': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0
    }

def main():
    DATA_PATH = './data'
    start_time = time.time()
    loader = DataLoader(DATA_PATH)
    
    try:
        # 1. CARREGAMENTO E ANÁLISE EXPLORATÓRIA
        print("1. Carregando dados completos...")
        df_ratings = loader.load_ratings()
        df_movies = loader.load_movies()
        df_tags = loader.load_tags()
        
        analyzer = DatasetAnalyzer(df_ratings, df_movies, df_tags)
        viz = Visualizer()
        
        # Gera estatísticas básicas
        stats = analyzer.get_basic_stats()
        print(f"   Dataset carregado: {stats['n_ratings']:,} avaliações de {stats['n_users']:,} usuários.")
        
        # 2. GERANDO GRÁFICOS
        print("\n2. Gerando gráficos (verifique a pasta 'plots')...")
        viz.plot_rating_distribution(analyzer.get_rating_distribution())
        viz.plot_long_tail(analyzer.get_item_popularity(), entity_name="Filmes")
        viz.plot_long_tail(analyzer.get_user_activity(), entity_name="Usuarios")
        viz.plot_ratings_over_time(analyzer.get_ratings_per_year())
        viz.plot_cumulative_users(analyzer.get_user_entry_cumulative())
        viz.plot_top_movies(analyzer.get_top_movies_by_title())
        viz.plot_popular_genres(analyzer.get_most_rated_genres())
        print("   Gráficos gerados com sucesso.")

        # 3. DEMONSTRAÇÃO (SINGLE USER)
        print("\n3. Demonstração Rápida (Most Popular)")
        # Treina rápido na base toda só para mostrar um exemplo
        demo_model = MostPopular(df_ratings, df_movies)
        demo_model.fit()
        test_user_id = 1
        print(f"   Recomendações para Usuário {test_user_id} (Top 5):")
        recs = demo_model.recommend(test_user_id, n=5, remove_seen=True)
        print(recs[['title', 'genres']].to_string(index=False))

        # 4. PREPARAÇÃO PARA COMPARAÇÃO (Redução de Dataset)
        print("\n4. Preparando Amostra para Comparação de Modelos...")
        print("   (Filtrando para os 2.000 usuários mais ativos e 5.000 filmes mais populares)")
        
        top_users = df_ratings['userId'].value_counts().head(2000).index
        top_movies = df_ratings['movieId'].value_counts().head(5000).index
        
        # Filtra o dataframe principal
        df_sample = df_ratings[
            (df_ratings['userId'].isin(top_users)) & 
            (df_ratings['movieId'].isin(top_movies))
        ].copy()
        
        # Filtra o dataframe de filmes também (necessário para o Content-Based)
        df_movies_sample = df_movies[df_movies['movieId'].isin(top_movies)].copy()
        
        print(f"   Amostra final: {len(df_sample)} ratings.")
        
        train_data, test_data = train_test_split(df_sample, test_size=0.2, random_state=42) # Split Treino/Teste (80/20)
        results = []

        # 5. MODELO 1: MOST POPULAR
        print("\n--- [1/3] Treinando Most Popular ---")
        pop_model = MostPopular(train_data, df_movies_sample)
        pop_model.fit()
        res_pop = evaluate_model(pop_model, test_data)
        res_pop['Model'] = 'Most Popular'
        results.append(res_pop)
        print(f"   NDCG: {res_pop['NDCG']:.4f}")

        # 6. MODELO 2: CONTENT-BASED (TF-IDF)
        print("\n--- [2/3] Treinando Content-Based")
        cb_model = ContentBasedRecommender(train_data, df_movies_sample)
        cb_model.fit()
        res_cb = evaluate_model(cb_model, test_data)
        res_cb['Model'] = 'Content-Based'
        results.append(res_cb)
        print(f"   NDCG: {res_cb['NDCG']:.4f}")

        # 7. MODELO 3: COLLABORATIVE FILTERING (SVD)
        print("\n--- [3/3] Treinando Collaborative Filtering (SVD) ---")
        cf_model = CollaborativeRecommender(train_data, df_movies_sample)
        cf_model.fit()
        res_cf = evaluate_model(cf_model, test_data)
        res_cf['Model'] = 'Collaborative (SVD)'
        results.append(res_cf)
        print(f"   NDCG: {res_cf['NDCG']:.4f}")

        # 8. RESULTADO FINAL
        print("\n" + "="*40)
        print("COMPARAÇÃO FINAL DE MODELOS (Top-10)")
        print("="*40)
        df_results = pd.DataFrame(results).set_index('Model')
        print(df_results)
        
        print(f"\nTempo total de execução: {time.time() - start_time:.2f}s")

    except Exception as e:
        print("Ocorreu um erro:")
        traceback.print_exc()

if __name__ == "__main__":
    main()