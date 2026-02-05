import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_plot(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close() # Libera memória
        print(f"Gráfico salvo em: {path}")

    def plot_rating_distribution(self, data):
        plt.figure(figsize=(10, 6))
        data.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Distribuição de Frequência das Avaliações')
        plt.xlabel('Nota')
        plt.ylabel('Quantidade de Avaliações')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        self.save_plot('1_distribuicao_notas.png')

    def plot_long_tail(self, data, entity_name="Itens"):
        plt.figure(figsize=(10, 6))
        # Plota apenas os valores (ignorando os IDs) para ver a curva
        plt.plot(data.values, color='blue')
        plt.title(f'Cauda Longa - Popularidade dos {entity_name}')
        plt.xlabel(f'{entity_name} (Ordenados por Popularidade)')
        plt.ylabel('Número de Avaliações')
        plt.yscale('log') # Escala logarítmica ajuda a ver melhor a cauda longa
        plt.grid(True, which="both", ls="-", alpha=0.2)
        self.save_plot(f'2_long_tail_{entity_name.lower()}.png')

    def plot_ratings_over_time(self, data):
        plt.figure(figsize=(10, 6))
        data.plot(kind='bar', color='salmon')
        plt.title('Número de Avaliações por Ano')
        plt.xlabel('Ano')
        plt.ylabel('Total de Avaliações')
        self.save_plot('3_avaliacoes_por_tempo.png')

    def plot_cumulative_users(self, entry_df):
        plt.figure(figsize=(10, 6))
        plt.plot(entry_df['date'], entry_df['count'], color='green', linewidth=2)
        plt.title('Crescimento da Base de Usuários (Acumulado)')
        plt.xlabel('Tempo')
        plt.ylabel('Total de Usuários no Sistema')
        plt.grid(True)
        self.save_plot('4_crescimento_usuarios.png')

    def plot_top_movies(self, df_top_movies):
        plt.figure(figsize=(12, 8))        
        data = df_top_movies.sort_values('count', ascending=True) # Inverte a ordem para o mais popular ficar no topo do gráfico
        plt.barh(data['title'], data['count'], color='purple')
        plt.title('Top 10 Filmes com Mais Avaliações')
        plt.xlabel('Número de Avaliações')
        plt.ylabel('Filme')
        plt.tight_layout() # Ajusta para o texto não cortar
        self.save_plot('5_top_movies_titles.png')

    def plot_popular_genres(self, genre_series):
        plt.figure(figsize=(12, 6))        
        genre_series.head(15).plot(kind='bar', color='orange', edgecolor='black') # Pega os top 15 gêneros para o gráfico não ficar cheio
        plt.title('Gêneros Mais Avaliados (Soma de Avaliações)')
        plt.xlabel('Gênero')
        plt.ylabel('Total de Avaliações Recebidas')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self.save_plot('6_most_rated_genres.png')