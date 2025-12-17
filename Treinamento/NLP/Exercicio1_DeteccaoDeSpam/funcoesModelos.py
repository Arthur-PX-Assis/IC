from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def KNN(X_train, y_train, X_test, y_test, caminho_escrita):
    # Instanciar o modelo KNN
    # n_neighbors=5: O modelo olha para os 5 vizinhos mais próximos para decidir
    knn_modelo = KNeighborsClassifier(n_neighbors=5)

    print("\nTreinando KNN...")
    # Treinar (fit)
    knn_modelo.fit(X_train, y_train)

    # Prever (predict)
    # O KNN calcula a distância matemática entre o texto de teste e todos os textos de treino
    y_pred_knn = knn_modelo.predict(X_test)

    # Avaliar
    acuracia_knn = accuracy_score(y_test, y_pred_knn)
    f1_macro_knn = f1_score(y_test, y_pred_knn, average='macro')

    relatorio = (
        f"--- Resultados do KNN ---\n"
        f"Acurácia: {acuracia_knn:.4f}\n"
        f"Macro F1: {f1_macro_knn:.4f}\n"
        f"----------------------------------\n"
    )

    with open(caminho_escrita, 'a', encoding='utf-8') as f:
        f.write(relatorio)

def SVM(X_train, y_train, X_test, y_test, caminho_escrita):
    # Instanciar (kernel 'linear' ou 'sigmoid' costumam ser ótimos para texto)
    svm_modelo = SVC(kernel='linear', random_state=42)

    print("\nTreinando SVM...")
    svm_modelo.fit(X_train, y_train)

    # Prever
    y_pred_svm = svm_modelo.predict(X_test)

    # Avaliar
    acuracia_svm = accuracy_score(y_test, y_pred_svm)
    f1_macro_svm = f1_score(y_test, y_pred_svm, average='macro')

    relatorio = (
        f"--- Resultados do SVM ---\n"
        f"Acurácia: {acuracia_svm:.4f}\n"
        f"Macro F1: {f1_macro_svm:.4f}\n"
        f"----------------------------------\n"
    )

    with open(caminho_escrita, 'a', encoding='utf-8') as f:
        f.write(relatorio)

def arvore_decisao(X_train, y_train, X_test, y_test, caminho_escrita):
    # Instanciar
    tree_modelo = DecisionTreeClassifier(random_state=42)

    print("\nTreinando Árvore de Decisão...")
    tree_modelo.fit(X_train, y_train)

    # Prever
    y_pred_tree = tree_modelo.predict(X_test)

    # Avaliar
    acuracia_tree = accuracy_score(y_test, y_pred_tree)
    f1_macro_tree = f1_score(y_test, y_pred_tree, average='macro')

    relatorio = (
        f"--- Resultados da Árvore de Decisão ---\n"
        f"Acurácia: {acuracia_tree:.4f}\n"
        f"Macro F1: {f1_macro_tree:.4f}\n"
        f"----------------------------------\n"
    )

    with open(caminho_escrita, 'a', encoding='utf-8') as f:
        f.write(relatorio)