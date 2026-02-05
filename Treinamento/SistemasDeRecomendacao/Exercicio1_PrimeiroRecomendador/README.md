# Primeiro Recomendador - Sistema de Recomendação com MovieLens

Projeto desenvolvido para estudo de sistemas de recomendação utilizando o dataset MovieLens 32M.

## 📂 Como obter os dados

Devido ao tamanho do dataset, os arquivos de dados **não estão incluídos no repositório**.

1. Baixe o dataset **MovieLens 32M** no site oficial:
   - Link: https://grouplens.org/datasets/movielens/32m/
2. Salve o arquivo `ml-32m.zip` dentro da pasta `data/` neste projeto.
3. Ao rodar o código, ele extrairá os dados automaticamente.

## 🚀 Como rodar

```bash
# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instale as dependências
pip install -r requirements.txt

# Execute
python main.py