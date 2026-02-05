import math

class RankEvaluator:
    def __init__(self):
        pass

    def precision_at_k(self, recommended_items, relevant_items, k):
        recommended_at_k = recommended_items[:k]
        relevant_count = len(set(recommended_at_k) & set(relevant_items))
        return relevant_count / k

    def recall_at_k(self, recommended_items, relevant_items, k):
        # Proteção contra divisão por zero
        if len(relevant_items) == 0:
            return 0.0
            
        recommended_at_k = recommended_items[:k]
        relevant_count = len(set(recommended_at_k) & set(relevant_items))
        return relevant_count / len(relevant_items)

    def f1_score_at_k(self, precision, recall):
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def ndcg_at_k(self, recommended_items, relevant_items, k):
        recommended_at_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        # Calcula DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item_id in enumerate(recommended_at_k):
            if item_id in relevant_set:
                # Fórmula padrão: rel_i / log2(i + 2)
                # Assume relevância binária (1 se viu, 0 se não viu)
                dcg += 1.0 / math.log2(i + 2)

        # Calcula IDCG (Ideal DCG - cenário perfeito)
        # O cenário perfeito é ter todos os itens relevantes nas primeiras posições
        idcg = 0.0
        num_relevant_at_k = min(len(relevant_items), k)
        for i in range(num_relevant_at_k):
            idcg += 1.0 / math.log2(i + 2)

        if idcg == 0:
            return 0.0
            
        return dcg / idcg