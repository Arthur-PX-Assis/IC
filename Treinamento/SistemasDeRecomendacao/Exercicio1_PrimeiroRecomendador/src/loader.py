import pandas as pd
import zipfile
import os

class DataLoader:
    def __init__(self, data_path, zip_filename='ml-32m.zip'):
        self.data_path = data_path
        self.zip_path = os.path.join(data_path, zip_filename)        
        self.extracted_folder = os.path.join(data_path, 'ml-32m') # O MovieLens extrai criando uma pasta com o nome do dataset

    def _check_and_extract(self):
        # Verifica se o ratings.csv já existe para não extrair toda vez
        target_file = os.path.join(self.extracted_folder, 'ratings.csv')
        
        if not os.path.exists(target_file):
            if os.path.exists(self.zip_path):
                print(f"Extraindo {self.zip_path}...")
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_path)
                print("Extração concluída.")
            else:
                raise FileNotFoundError(f"Arquivo {self.zip_path} não encontrado na pasta data.")
        else:
            print("Arquivos já extraídos. Carregando...")

    def load_ratings(self):
        self._check_and_extract()
        csv_path = os.path.join(self.extracted_folder, 'ratings.csv')
        return pd.read_csv(csv_path, usecols=['userId', 'movieId', 'rating', 'timestamp'], nrows=5000000) # Carregam apenas colunas essenciais

    def load_movies(self):
        self._check_and_extract()
        csv_path = os.path.join(self.extracted_folder, 'movies.csv')
        return pd.read_csv(csv_path, dtype={'movieId': 'int32'})

    def load_tags(self):
        self._check_and_extract()
        csv_path = os.path.join(self.extracted_folder, 'tags.csv')
        return pd.read_csv(csv_path, usecols=['userId', 'movieId', 'tag']) # Tag é texto, então o pandas detecta object automaticamente