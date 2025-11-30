import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# X e y sao train
def teste_estatistico(X, y, lista_best_params):
    print("\n--- Iniciando Teste T com correção de Bonferroni ---")

    params_knn = lista_best_params[0]
    params_svm = lista_best_params[1]
    params_tree = lista_best_params[2]

    # Recriando os modelos com os parâmetros campeões
    # O ** serve para "desembrulhar" o dicionário dentro dos parênteses
    knn_otimizado = KNeighborsClassifier(**params_knn)
    svm_otimizado = SVC(**params_svm)
    tree_otimizado = DecisionTreeClassifier(random_state=42, **params_tree)

    modelos = {
        "KNN": knn_otimizado,
        "SVM": svm_otimizado,
        "Arvore": tree_otimizado
    }

    # Coletando as 10 notas (Cross Validation)
    resultados_folds = {}
    
    print("Recalculando os 10 folds para os modelos otimizados...")
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=10, scoring='f1_macro', n_jobs=-1)
        resultados_folds[nome] = scores
        print(f"-> {nome}: Média {scores.mean():.4f} (Desvio: {scores.std():.4f})")

    # Executando o Teste Estatístico
    print("\n--- Resultados do Teste de Hipótese ---")
    
    alpha = 0.05
    pares = [("KNN", "SVM"), ("KNN", "Arvore"), ("SVM", "Arvore")]
    n_comparacoes = len(pares)
    alpha_bonferroni = alpha / n_comparacoes
    
    print(f"Alpha Bonferroni (Critério): {alpha_bonferroni:.6f}\n")
    print(f"{'Comparação':<20} | {'P-Value':<10} | {'Conclusão'}")

    for modelo_A, modelo_B in pares:
        scores_A = resultados_folds[modelo_A]
        scores_B = resultados_folds[modelo_B]
        
        # Teste T Pareado
        t_stat, p_value = stats.ttest_rel(scores_A, scores_B)
        
        if p_value < alpha_bonferroni:
            conclusao = "Diferença de desempenho real e não sorte"
        else:
            conclusao = "Empate Técnico"
            
        print(f"{modelo_A} vs {modelo_B:<13} | {p_value:.6f}   | {conclusao}")