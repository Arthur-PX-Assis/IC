import os
import zipfile
import pandas as pd

def carregar_dados(caminho_zip_externo, pasta_destino):

    # Criar pasta de destino se não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Descompactar o zip externo (TREC.zip)
    print(f"Extraindo {caminho_zip_externo}...")
    with zipfile.ZipFile(caminho_zip_externo, 'r') as zip_ref:
        zip_ref.extractall(pasta_destino)

    # Procurar e descompactar o zip interno (trec.zip)
    arquivo_interno_zip = None
    for root, dirs, files in os.walk(pasta_destino):
        if "trec.zip" in files:
            arquivo_interno_zip = os.path.join(root, "trec.zip")
            break

    if arquivo_interno_zip:
        print(f"Extraindo zip interno: {arquivo_interno_zip}...")
        with zipfile.ZipFile(arquivo_interno_zip, 'r') as zip_ref:
            zip_ref.extractall(pasta_destino) # Extrair para a pasta destino principal
        
        os.remove(arquivo_interno_zip) # Remover o zip interno após extrair
        print("Arquivo zip interno removido.")

    # Localizar o texts.txt para carregar o df
    caminho_txt = None
    for root, dirs, files in os.walk(pasta_destino):
        if "texts.txt" in files:
            caminho_txt = os.path.join(root, "texts.txt")
            break

    if not caminho_txt:
        raise FileNotFoundError("Arquivo 'texts.txt' não encontrado após a extração.")

    # Carregar no pandas
    with open(caminho_txt, 'r', encoding='utf-8') as f:
        texts_raw = [line.strip() for line in f.readlines()]

    df = pd.DataFrame({'texto_original': texts_raw})
    print(f"Dataset carregado com {len(df)} linhas.")
    return df