from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

def arvore_decisao(X_train, y_train, X_test, y_test):
    # Instanciar
    tree_modelo = DecisionTreeClassifier(random_state=42)

    print("\nTreinando Árvore de Decisão...")
    tree_modelo.fit(X_train, y_train)

    # Prever
    y_pred_tree = tree_modelo.predict(X_test)

    # Avaliar
    acuracia_tree = accuracy_score(y_test, y_pred_tree)
    f1_macro_tree = f1_score(y_test, y_pred_tree, average='macro')

    print(f"--- Resultados da Árvore de Decisão ---")
    print(f"Acurácia: {acuracia_tree:.4f}")
    print(f"Macro F1: {f1_macro_tree:.4f}")