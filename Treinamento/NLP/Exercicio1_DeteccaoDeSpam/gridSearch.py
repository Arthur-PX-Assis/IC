from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# --- CONFIGURAÇÃO DO GRID SEARCH ---

# Dicionários de Parâmetros
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

# --- FUNÇÃO DE EXECUÇÃO ---

def grid_individual(modelo, params, nome, X_train, y_train, X_test, y_test):
    print(f"\n--- Otimizando {nome} ---")
    
    # K Fold Cross Validation utilizado por meio de cv=10, definindo 10 conjuntos
    # n_jobs=-1 usa todos os núcleos do processador
    grid = GridSearchCV(modelo, params, cv=10, n_jobs=-1, scoring='f1_macro')
    grid.fit(X_train, y_train)
    
    index_melhor = grid.best_index_
    resultados = grid.cv_results_
    media_f1 = resultados['mean_test_score'][index_melhor]
    desvio_padrao = resultados['std_test_score'][index_melhor]
    
    # Cálculo do Intervalo de Confiança (95% ~= 2 desvios)
    limite_inferior = media_f1 - (2 * desvio_padrao)
    limite_superior = media_f1 + (2 * desvio_padrao)

    print(f"Melhores Parâmetros: {grid.best_params_}")
    print(f"Melhor Macro F1 (Validação): {grid.best_score_:.4f}")
    print(f"Desvio Padrão: {desvio_padrao:.4f}")
    print(f"Intervalo de Confiança (95%): [{limite_inferior:.4f} - {limite_superior:.4f}]")
    
    # Teste final com o melhor modelo encontrado
    melhor_modelo = grid.best_estimator_
    predicoes = melhor_modelo.predict(X_test)
    f1_final = f1_score(y_test, predicoes, average='macro')
    dentro_do_intervalo = limite_inferior <= f1_final <= limite_superior

    print(f"Macro F1 no Teste (Real): {f1_final:.4f}")
    print(f"O resultado final está dentro do intervalo previsto? {'Sim' if dentro_do_intervalo else 'Não'}")

    return grid.best_params_

def rodar_grid(X_train, y_train, X_test, y_test):
    melhores_parametros = []
    melhores_parametros.append(grid_individual(KNeighborsClassifier(), params_knn, "KNN", X_train, y_train, X_test, y_test))
    melhores_parametros.append(grid_individual(SVC(), params_svm, "SVM", X_train, y_train, X_test, y_test))
    melhores_parametros.append(grid_individual(DecisionTreeClassifier(random_state=42), params_tree, "Arvore de Decisao", X_train, y_train, X_test, y_test))
    return melhores_parametros