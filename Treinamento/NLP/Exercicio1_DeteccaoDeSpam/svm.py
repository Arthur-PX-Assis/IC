from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def SVM(X_train, y_train, X_test, y_test):
    # Instanciar (Kernel 'linear' ou 'sigmoid' costumam ser ótimos para texto)
    svm_modelo = SVC(kernel='linear', random_state=42)

    print("\nTreinando SVM...")
    svm_modelo.fit(X_train, y_train)

    # Prever
    y_pred_svm = svm_modelo.predict(X_test)

    # Avaliar
    acuracia_svm = accuracy_score(y_test, y_pred_svm)
    f1_macro_svm = f1_score(y_test, y_pred_svm, average='macro')

    print(f"--- Resultados do SVM ---")
    print(f"Acurácia: {acuracia_svm:.4f}")
    print(f"Macro F1: {f1_macro_svm:.4f}")