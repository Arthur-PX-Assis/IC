import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

class MostPopular:
    def __init__(self, df_ratings, df_movies):
        self.df_ratings = df_ratings
        self.df_movies = df_movies
        self.popularity_df = None
        self.name = "Most Popular"

    def fit(self):
        print(f"Treinando modelo {self.name}...")
        
        # Conta quantas vezes cada filme foi avaliado
        pop_counts = self.df_ratings['movieId'].value_counts().reset_index()
        pop_counts.columns = ['movieId', 'score'] # 'score' é a quantidade de views
        
        # Faz o merge para pegar os títulos (opcional, mas bom para exibição)
        self.popularity_df = pop_counts.merge(
            self.df_movies[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        # Ordena do maior para o menor (ranking global)
        self.popularity_df = self.popularity_df.sort_values('score', ascending=False)
        print("Treinamento concluído.")

    def recommend(self, user_id, n=10, remove_seen=True):
        if self.popularity_df is None:
            raise Exception("O modelo não foi treinado. Execute .fit() primeiro.")
            
        
        recs = self.popularity_df.copy() # Começa com a lista global de populares        
        if remove_seen:
            # Descobre quais filmes esse usuário já viu
            seen_mask = self.df_ratings['userId'] == user_id
            seen_items = self.df_ratings.loc[seen_mask, 'movieId'].unique()
            
            # Filtra removendo os vistos (~ é negação)
            recs = recs[~recs['movieId'].isin(seen_items)]
            
        return recs.head(n) # Retorna os top N
    
class ContentBasedRecommender:
    def __init__(self, df_ratings, df_movies):
        self.df_ratings = df_ratings
        self.df_movies = df_movies.reset_index(drop=True) # Garante índice alinhado
        self.name = "Content-Based (Gêneros)"
        self.movie_indices = None
        self.tfidf_matrix = None
        
    def fit(self):
        print(f"Treinando {self.name}...")
        # Cria a "sopa" de gêneros
        self.df_movies['genres_str'] = self.df_movies['genres'].str.replace('|', ' ')
        
        # Calcula TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df_movies['genres_str'])
        
        # Mapeamento Reverso: ID do Filme -> Índice da Matriz (Row Number)
        # Tem que estar alinhado com o dataframe resetado
        self.movie_indices = pd.Series(self.df_movies.index, index=self.df_movies['movieId'])
        
        print("Treinamento (Indexação) concluído.")

    def recommend(self, user_id, n=10, remove_seen=True):
        # Pega histórico positivo do usuário
        user_history = self.df_ratings[(self.df_ratings['userId'] == user_id) & (self.df_ratings['rating'] >= 4)]
        
        if user_history.empty:
            return pd.DataFrame(columns=['movieId', 'score'])
        
        liked_movies_ids = user_history['movieId'].tolist()
        
        # Recupera os índices da matriz correspondentes aos filmes que ele gostou
        valid_indices = [self.movie_indices[mid] for mid in liked_movies_ids if mid in self.movie_indices]
        
        if not valid_indices:
            return pd.DataFrame(columns=['movieId', 'score'])

        # Em vez de fazer um loop, calcula a similaridade de TODOS os filmes curtidos
        # contra TODOS os filmes do banco de uma vez só.
        # Resultado: Uma matriz (N_curtidos x N_total_filmes)
        user_sim_matrix = linear_kernel(self.tfidf_matrix[valid_indices], self.tfidf_matrix)
        
        # Soma as similaridades verticalmente (axis=0)
        # Se o filme X é similar ao curtido A e ao curtido B, ele ganha pontos dos dois.
        total_scores = user_sim_matrix.sum(axis=0)
        
        # Cria o dataframe direto do array numpy (Muito rápido)
        recs = pd.DataFrame({
            'movieId': self.df_movies['movieId'].values,
            'score': total_scores
        })
        
        recs = recs.sort_values('score', ascending=False) # Ordena
        
        # Filtra vistos
        if remove_seen:
            seen_items = self.df_ratings[self.df_ratings['userId'] == user_id]['movieId'].unique()
            recs = recs[~recs['movieId'].isin(seen_items)]
            
        return recs.head(n)

class CollaborativeRecommender:
    def __init__(self, df_ratings, df_movies):
        self.df_ratings = df_ratings
        self.df_movies = df_movies
        self.name = "Collaborative Filtering (SVD)"
        self.algo = None
        self.user_ids_map = None
        self.item_ids_map = None
        self.matrix_u_i = None

    def fit(self):
        print(f"Treinando {self.name}...")
        
        # Cria a Matriz Esparsa (Pivot Table) com usuários nas linhas r filmes nas colunas
        # Preenche com 0 onde não tem nota
        pivot_table = self.df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Mapeamento para saber qual linha é qual usuário
        self.user_ids_map = {id: i for i, id in enumerate(pivot_table.index)}
        self.index_to_user = {i: id for id, i in self.user_ids_map.items()}
        self.item_ids_map = {id: i for i, id in enumerate(pivot_table.columns)}
        self.index_to_item = {i: id for id, i in self.item_ids_map.items()}
        
        self.matrix_u_i = pivot_table.values
        
        # Aplica SVD (Decomposição de Valor Singular)
        # Reduz a dimensionalidade para descobrir "tópicos latentes"       
        self.algo = TruncatedSVD(n_components=20, random_state=42)  # n_components=20 é um chute inicial
        self.matrix_reduced = self.algo.fit_transform(self.matrix_u_i)
        
        # Matriz de correlação aproximada (reconstrução)
        self.corr_matrix = np.corrcoef(self.matrix_reduced)
        print("Fatoração de Matriz concluída.")

    def recommend(self, user_id, n=10, remove_seen=True):
        if user_id not in self.user_ids_map:
            return pd.DataFrame(columns=['movieId', 'score'])
            
        user_idx = self.user_ids_map[user_id]
        user_vector = self.matrix_reduced[user_idx].reshape(1, -1)        
        predicted_ratings = np.dot(user_vector, self.algo.components_).flatten() # Reconstrói as notas para esse usuário
        
        # Cria dataframe com as predições
        recs = pd.DataFrame({
            'movieId': [self.index_to_item[i] for i in range(len(predicted_ratings))],
            'score': predicted_ratings
        })
        
        recs = recs.sort_values('score', ascending=False)
        
        if remove_seen:
            seen_mask = self.df_ratings['userId'] == user_id
            seen_items = self.df_ratings.loc[seen_mask, 'movieId'].unique()
            recs = recs[~recs['movieId'].isin(seen_items)]
            
        return recs.head(n)