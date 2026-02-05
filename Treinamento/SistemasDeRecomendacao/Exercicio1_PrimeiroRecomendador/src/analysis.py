import pandas as pd

class DatasetAnalyzer:
    def __init__(self, df_ratings, df_movies=None, df_tags=None):
        self.df_ratings = df_ratings
        self.df_movies = df_movies
        self.df_tags = df_tags

    def get_basic_stats(self):
        n_users = self.df_ratings['userId'].nunique()
        n_items = self.df_ratings['movieId'].nunique()
        n_ratings = len(self.df_ratings)
        
        total_elements = n_users * n_items
        sparsity = 1 - (n_ratings / total_elements)
        
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": n_ratings,
            "sparsity": sparsity
        }
    
    def get_genre_distribution(self):
        if self.df_movies is None:
            return {}
            
        # Separa a string pelo pipe '|'
        # 'explode' transforma cada elemento da lista em uma nova linha
        # Conta os valores
        genres_counts = self.df_movies['genres'].str.split('|').explode().value_counts()
        return genres_counts

    def get_top_tags(self, n=10):
        if self.df_tags is None:
            return {}
            
        # Tudo minúsculo
        tags_normalized = self.df_tags['tag'].astype(str).str.lower()
        return tags_normalized.value_counts().head(n)

    def get_user_activity_stats(self):
        user_counts = self.df_ratings['userId'].value_counts()
        return {
            "min_ratings": user_counts.min(),
            "max_ratings": user_counts.max(),
            "mean_ratings": user_counts.mean(),
            "median_ratings": user_counts.median()
        }
    
    def get_rating_distribution(self):
        return self.df_ratings['rating'].value_counts().sort_index()

    def get_item_popularity(self):
        return self.df_ratings.groupby('movieId').size().sort_values(ascending=False)

    def get_user_activity(self):
        return self.df_ratings.groupby('userId').size().sort_values(ascending=False)

    def get_ratings_per_year(self):
        # Converte timestamp para datetime apenas aqui para economizar memória no resto
        temp_df = pd.to_datetime(self.df_ratings['timestamp'], unit='s') # unit='s' pq MovieLens usa Unix Timestamp
        return temp_df.dt.year.value_counts().sort_index()

    def get_user_entry_cumulative(self):
        # Acha a data mínima (entrada) de cada usuário
        user_entry_dates = self.df_ratings.groupby('userId')['timestamp'].min()
        user_entry_dates = pd.to_datetime(user_entry_dates, unit='s').sort_values()
        
        # Cria um dataframe para facilitar o plot acumulado
        entry_df = pd.DataFrame({
            'date': user_entry_dates,
            'count': range(1, len(user_entry_dates) + 1) # O range(1, len+1) cria a contagem: usuário 1, usuário 2, ...
        })
        return entry_df

    def get_top_movies_by_title(self, n=10):
        # Conta quantas avaliações cada ID teve
        popularity_counts = self.df_ratings['movieId'].value_counts()
        
        # Pega os top N IDs
        top_ids = popularity_counts.head(n)
        
        # Filtra o dataframe de filmes para pegar apenas esses IDs
        # Faz o merge para trazer o título
        top_movies = self.df_movies[self.df_movies['movieId'].isin(top_ids.index)].copy()
        
        # Adiciona a contagem de views no dataframe final para ordenar
        top_movies['count'] = top_movies['movieId'].map(top_ids)
        
        return top_movies.sort_values('count', ascending=False)[['title', 'count']]

    def get_most_rated_genres(self):
        # Conta ratings por filme
        movie_counts = self.df_ratings['movieId'].value_counts()
        
        # Cria uma cópia leve apenas com ID e Gêneros
        genres_df = self.df_movies[['movieId', 'genres']].copy()
        
        # Mapeia a contagem para cada filme (filmes sem rating ficam com 0)
        genres_df['popularity'] = genres_df['movieId'].map(movie_counts).fillna(0)
        
        # Separa os gêneros (explode) e soma a popularidade
        # Se Toy Story tem 100 ratings e é "Adventure|Animation", soma +100 para Adventure e +100 para Animation.
        genre_popularity = genres_df.assign(
            genres=genres_df['genres'].str.split('|')
        ).explode('genres').groupby('genres')['popularity'].sum().sort_values(ascending=False)
        
        return genre_popularity