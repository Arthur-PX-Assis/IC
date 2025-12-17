from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from scipy import stats

def rodar_grid(X_train, y_train, X_test, y_test, caminho_escrita):

    def grid_individual(modelo, params, nome, X_train, y_train, X_test, y_test, caminho_escrita):

        print(f"\n--- Otimizando {nome} ---")
        
        # K fold cross validation utilizado por meio de cv, definindo n conjuntos
        # n_jobs=-1 usa todos os núcleos do processador
        grid = GridSearchCV(modelo, params, cv=3, n_jobs=-1, scoring='f1_macro') # cv=3 por conta das limitações
        grid.fit(X_train, y_train)
        
        index_melhor = grid.best_index_
        resultados = grid.cv_results_
        media_f1 = resultados['mean_test_score'][index_melhor]
        desvio_padrao = resultados['std_test_score'][index_melhor]
        
        # Cálculo do intervalo de confiança (95% ~= 2 desvios)
        limite_inferior = media_f1 - (2 * desvio_padrao)
        limite_superior = media_f1 + (2 * desvio_padrao)
        
        # Teste final com o melhor modelo encontrado
        melhor_modelo = grid.best_estimator_
        predicoes = melhor_modelo.predict(X_test)
        f1_final = f1_score(y_test, predicoes, average='macro')
        dentro_do_intervalo = limite_inferior <= f1_final <= limite_superior

        relatorio = (
            f"Melhores Parâmetros {nome}: {grid.best_params_}\n"
            f"Melhor Macro F1 (Validação): {grid.best_score_:.4f}\n"
            f"Desvio Padrão: {desvio_padrao:.4f}\n"
            f"Intervalo de Confiança (95%): [{limite_inferior:.4f} - {limite_superior:.4f}]\n"
            f"Macro F1 no Teste (Real): {f1_final:.4f}\n"
            f"O resultado final está dentro do intervalo previsto? {'Sim' if dentro_do_intervalo else 'Não'}\n"
            f"--------------------------------------------------\n"
        )

        with open(caminho_escrita, 'a', encoding='utf-8') as f:
            f.write(relatorio)

        return grid.best_params_

    params_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    params_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'sigmoid'], 
        'gamma': ['scale', 'auto']
    }

    params_tree = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10]
    }

    melhores_parametros = []
    melhores_parametros.append(grid_individual(KNeighborsClassifier(), params_knn, "KNN", X_train, y_train, X_test, y_test, caminho_escrita))
    melhores_parametros.append(grid_individual(SVC(), params_svm, "SVM", X_train, y_train, X_test, y_test, caminho_escrita))
    melhores_parametros.append(grid_individual(DecisionTreeClassifier(random_state=42), params_tree, "Arvore de Decisao", X_train, y_train, X_test, y_test, caminho_escrita))
    return melhores_parametros

# X e y sao train
def teste_estatistico(X, y, lista_best_params, caminho_escrita):
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

    # Coletando as 10 notas (cross validation)
    resultados_folds = {}
    
    print("Recalculando os 10 folds para os modelos otimizados...")

    relatorio = "Médias nos 10 Folds:\n"

    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=10, scoring='f1_macro', n_jobs=-1)
        resultados_folds[nome] = scores
        relatorio += f"-> {nome}: Média {scores.mean():.4f} (Desvio: {scores.std():.4f})\n"

    # Executando o teste estatístico
    relatorio += "\n--- Resultados do Teste de Hipótese ---"
    
    alpha = 0.05
    pares = [("KNN", "SVM"), ("KNN", "Arvore"), ("SVM", "Arvore")]
    n_comparacoes = len(pares)
    alpha_bonferroni = alpha / n_comparacoes
    
    relatorio += f"Alpha Bonferroni (Critério): {alpha_bonferroni:.6f}\n\n"
    relatorio += f"{'Comparação':<20} | {'P-Value':<10} | {'Conclusão'}\n"

    for modelo_A, modelo_B in pares:
        scores_A = resultados_folds[modelo_A]
        scores_B = resultados_folds[modelo_B]
        
        # Teste T pareado
        t_stat, p_value = stats.ttest_rel(scores_A, scores_B)
        
        if p_value < alpha_bonferroni:
            conclusao = "Diferença de desempenho real e não sorte"
        else:
            conclusao = "Empate Técnico"
            
        relatorio += f"{modelo_A} vs {modelo_B:<13} | {p_value:.6f}   | {conclusao}\n"

    with open(caminho_escrita, 'a', encoding='utf-8') as f:
        f.write(relatorio)