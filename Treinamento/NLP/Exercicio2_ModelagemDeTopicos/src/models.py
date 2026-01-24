from bertopic import BERTopic
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def gerar_tfidf(textos_limpos):
    print("Gerando matriz TF-IDF...")

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2) # min_df=2 ignora palavras que aparecem em menos de 2 documentos
    tfidf_matrix = vectorizer.fit_transform(textos_limpos)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Matriz Criada: {tfidf_matrix.shape} (Documentos x Palavras)")
    return tfidf_matrix, vectorizer, feature_names

def treinar_nmf(tfidf_matrix, n_topicos):
    print(f"Treinando NMF com {n_topicos} tópicos...")
    nmf_model = NMF(n_components=n_topicos, random_state=42, init='nndsvd')
    nmf_model.fit(tfidf_matrix)
    return nmf_model

def gerar_embeddings(lista_textos, nome_modelo="all-MiniLM-L6-v2"):
    print(f"--- Carregando modelo de embeddings: {nome_modelo} ---")
    sentence_model = SentenceTransformer(nome_modelo)
    
    print("Gerando embeddings (pode demorar um pouco)...")
    embeddings = sentence_model.encode(lista_textos, show_progress_bar=True)
    
    return embeddings, sentence_model

def treinar_bertopic(lista_textos, embeddings, sentence_model, n_topicos=5):
    print(f"Treinando BERTopic com {n_topicos} tópicos...")
    topic_model = BERTopic(language="english", 
                           nr_topics=n_topicos, # número desejado de tópicos
                           embedding_model=sentence_model, 
                           verbose=True)
    
    topics, probs = topic_model.fit_transform(lista_textos, embeddings=embeddings) # retorna os tópicos e as probabilidades

    return topic_model