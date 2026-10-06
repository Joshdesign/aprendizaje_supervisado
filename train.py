import os
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    print("=" * 60)
    print("INICIANDO EL ENTRENAMIENTO DEL PIPELINE DE MACHINE LEARNING")
    print("=" * 60)

    # 1. Cargar los datos limpios
    data_path = 'data/titanic_clean.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en {data_path}. Corre los notebooks de limpieza primero.")
    
    print(f"Cargando datos desde: {data_path}...")
    df = pd.read_csv(data_path)
    
    # Definir características y variable objetivo
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df['Survived']

    # 2. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de prueba: {len(X_test)}")
    print("-" * 50)

    # 3. Definir transformadores de columnas
    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    categorical_cols = ['Sex', 'Embarked']

    print("Definiendo el pipeline...")
    print(f"--> Númerico (StandardScaler): {numerical_cols}")
    print(f"--> Categórico (OneHotEncoder): {categorical_cols}")
    print("-" * 50)

    # Crear transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=
        [('num', StandardScaler(), numerical_cols),
         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)]
    )

 # 4. Crear los pipelines para ambos modelos
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6))
    ])

    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # 5. Entrenar y evaluar Random Forest
    print("\nEntrenando Random Forest...")
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"¡Random Forest entrenado con éxito!")
    print(f"Exactitud (Accuracy) en prueba: {acc_rf:.4f}")
    print("\nReporte de Clasificación (Random Forest):")
    print(classification_report(y_test, y_pred_rf))
    print("-" * 50)

    # 6. Entrenar y evaluar Regresión Logística
    print("\nEntrenando Regresión Logística...")
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"¡Regresión Logística entrenada con éxito!")
    print(f"Exactitud (Accuracy) en prueba: {acc_lr:.4f}")
    print("\nReporte de Clasificación (Regresión Logística):")
    print(classification_report(y_test, y_pred_lr))
    print("-" * 50)

    # 7. Guardar los modelos entrenados
    print("\nGuardando los pipelines entrenados (Preprocesador + Modelo)...")
    rf_model_filename = 'model_rf.joblib'
    lr_model_filename = 'model_lr.joblib'
    
    joblib.dump(pipeline_rf, rf_model_filename)
    joblib.dump(pipeline_lr, lr_model_filename)
    
    print(f" -> Guardado exitosamente: '{rf_model_filename}'")
    print(f" -> Guardado exitosamente: '{lr_model_filename}'")
    print("=" * 60)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("=" * 60)

if __name__ == '__main__':
    train_models()