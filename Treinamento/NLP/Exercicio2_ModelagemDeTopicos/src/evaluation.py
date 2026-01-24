import gensim
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

def preparar_dicionario(textos_limpos):

    print("--- Preparando Dicionário Gensim ---")

    tokenized_texts = [text.split() for text in textos_limpos]
    dictionary = Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5) # no_below=2 para permitir palavras mais raras
    
    print(f"Dicionário criado com {len(dictionary)} tokens únicos.")
    return dictionary, tokenized_texts

def get_nmf_topics(model, feature_names, dictionary, n_top_words=10):
    topics = []
    valid_tokens = set(dictionary.token2id.keys())

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        raw_words = [feature_names[i] for i in top_features_ind]
        filtered_words = [w for w in raw_words if w in valid_tokens]
        if filtered_words:
            topics.append(filtered_words)  
    return topics

def get_bertopic_topics(model, dictionary, n_top_words=10):
    topics = []
    valid_tokens = set(dictionary.token2id.keys())
    topic_info = model.get_topic_info()
    
    for topic_id in topic_info['Topic']:
        if topic_id != -1:
            raw_words = [word for word, _ in model.get_topic(topic_id)][:50] # Pega as top 50 palavras para aumentar a chance de match
            filtered_words = [w for w in raw_words if w in valid_tokens][:n_top_words]  
            if len(filtered_words) >= 3: # Só adiciona se tiver pelo menos 3 palavras
                topics.append(filtered_words)
    return topics

def calcular_coerencia(topics, tokenized_texts, dictionary, name):
    if not topics:
        print(f"{name}: Sem tópicos válidos.")
        return 0.0
        
    try:
        cm = CoherenceModel(topics=topics, texts=tokenized_texts, dictionary=dictionary, coherence='c_npmi')
        score = cm.get_coherence()
    except Exception as e:
        print(f"{name}: Erro ao calcular ({e})")
        return 0.0
        
    print(f"{name}: {score:.4f}")
    return score

def avaliar_todos_modelos(df, nmf_5, nmf_10, bert_5, bert_10, tfidf_feature_names):
    print("\n--- Iniciando Avaliação de Coerência (NPMI) ---")
    
    dictionary, tokenized_texts = preparar_dicionario(df['texto_limpo'])
    
    print("Extraindo tópicos e validando vocabulário...")
    topics_nmf_5 = get_nmf_topics(nmf_5, tfidf_feature_names, dictionary)
    topics_nmf_10 = get_nmf_topics(nmf_10, tfidf_feature_names, dictionary)
    topics_bert_5 = get_bertopic_topics(bert_5, dictionary)
    topics_bert_10 = get_bertopic_topics(bert_10, dictionary)
    
    print("\nCalculando scores...")
    s1 = calcular_coerencia(topics_nmf_5, tokenized_texts, dictionary, "NMF (5)")
    s2 = calcular_coerencia(topics_nmf_10, tokenized_texts, dictionary, "NMF (10)")
    s3 = calcular_coerencia(topics_bert_5, tokenized_texts, dictionary, "BERTopic (5)")
    s4 = calcular_coerencia(topics_bert_10, tokenized_texts, dictionary, "BERTopic (10)")
    
    return s1, s2, s3, s4