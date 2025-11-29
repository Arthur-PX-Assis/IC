from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def KNN(X_train, y_train, X_test, y_test):
    # Instanciar o Modelo KNN
    # n_neighbors=5: O modelo olhará para os 5 vizinhos mais próximos para decidir
    knn_modelo = KNeighborsClassifier(n_neighbors=5)

    print("\nTreinando KNN...")
    # Treinar (Fit)
    knn_modelo.fit(X_train, y_train)

    # Prever (Predict)
    # O KNN calcula a distância matemática entre o texto de teste e todos os textos de treino
    y_pred_knn = knn_modelo.predict(X_test)

    # Avaliar
    acuracia_knn = accuracy_score(y_test, y_pred_knn)
    f1_macro_knn = f1_score(y_test, y_pred_knn, average='macro')

    print(f"--- Resultados do KNN ---")
    print(f"Acurácia: {acuracia_knn:.4f}")
    print(f"Macro F1: {f1_macro_knn:.4f}")