import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

app = Flask(__name__, static_folder='static', template_folder='templates')

# Paths
DATA_PATH = 'data/titanic_clean.csv'
RF_MODEL_PATH = 'model_rf.joblib'
LR_MODEL_PATH = 'model_lr.joblib'

# Global references for models and dataset
model_rf = None
model_lr = None
df_titanic = None

def load_resources():
    global model_rf, model_lr, df_titanic
    
    # 1. Check/Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset clean no encontrado en {DATA_PATH}. Por favor ejecuta la limpieza de datos primero.")
    
    df_titanic = pd.read_csv(DATA_PATH)
    
    # 2. Check/Load Models
    if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(LR_MODEL_PATH):
        print("Modelos no encontrados. Entrenando modelos mediante train.py...")
        from train import train_models
        train_models()
        
    model_rf = joblib.load(RF_MODEL_PATH)
    model_lr = joblib.load(LR_MODEL_PATH)
    print("Recursos cargados correctamente.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Genera estadísticas del dataset para los gráficos del frontend."""
    try:
        global df_titanic
        if df_titanic is None:
            load_resources()
            
        # 1. Tasa de supervivencia general
        total_passengers = int(df_titanic.shape[0])
        survived_count = int(df_titanic['Survived'].sum())
        survival_rate = float(survived_count / total_passengers)
        
        # 2. Supervivencia por sexo
        sex_stats = df_titanic.groupby('Sex')['Survived'].mean().to_dict()
        
        # 3. Supervivencia por Clase (Pclass)
        class_stats = df_titanic.groupby('Pclass')['Survived'].mean().to_dict()
        # Convert keys to string for JSON serialization
        class_stats = {str(k): float(v) for k, v in class_stats.items()}
        
        # 4. Distribución por Edades y Supervivencia
        # Crear grupos de edad
        age_bins = [0, 12, 19, 35, 60, 120]
        age_labels = ['Niños (0-12)', 'Jóvenes (13-19)', 'Adultos (20-35)', 'Adultos Mayores (36-60)', 'Tercera Edad (60+)']
        df_titanic['AgeGroup'] = pd.cut(df_titanic['Age'], bins=age_bins, labels=age_labels)
        
        age_group_stats = df_titanic.groupby('AgeGroup', observed=False)['Survived'].agg(['count', 'mean']).to_dict(orient='index')
        age_stats_clean = {k: {'total': int(v['count']), 'rate': float(v['mean'])} for k, v in age_group_stats.items()}
        
        # 5. Supervivencia por Puerto de Embarque
        embarked_stats = df_titanic.groupby('Embarked')['Survived'].mean().to_dict()
        # Mapear nombres completos
        embarked_names = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
        embarked_stats = {embarked_names.get(k, k): float(v) for k, v in embarked_stats.items()}
        
        return jsonify({
            'success': True,
            'summary': {
                'total': total_passengers,
                'survived': survived_count,
                'rate': survival_rate
            },
            'sex': sex_stats,
            'class': class_stats,
            'age': age_stats_clean,
            'embarked': embarked_stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/metrics')
def get_metrics():
    """Calcula y devuelve métricas avanzadas de entrenamiento para ambos modelos."""
    try:
        global model_rf, model_lr, df_titanic
        if model_rf is None or model_lr is None or df_titanic is None:
            load_resources()
            
        # Re-crear la partición de test igual que en train.py para evaluar fielmente
        X = df_titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = df_titanic['Survived']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Métricas Random Forest
        y_pred_rf = model_rf.predict(X_test)
        y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
        acc_rf = accuracy_score(y_test, y_pred_rf)
        prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')
        cm_rf = confusion_matrix(y_test, y_pred_rf).tolist() # [[TN, FP], [FN, TP]]
        
        # Métricas Regresión Logística
        y_pred_lr = model_lr.predict(X_test)
        y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
        acc_lr = accuracy_score(y_test, y_pred_lr)
        prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
        cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()
        
        # Extraer importancia de características para Random Forest
        feature_names = model_rf.named_steps['preprocessor'].get_feature_names_out()
        # Limpiar nombres de características para presentarlas legibles
        clean_feature_names = [
            f.replace('num__', '').replace('cat__', '') for f in feature_names
        ]
        
        importances_rf = model_rf.named_steps['classifier'].feature_importances_.tolist()
        rf_features = sorted(
            [{'name': name, 'value': val} for name, val in zip(clean_feature_names, importances_rf)],
            key=lambda x: x['value'],
            reverse=True
        )
        
        # Extraer coeficientes para Regresión Logística
        coefs_lr = model_lr.named_steps['classifier'].coef_[0].tolist()
        lr_features = sorted(
            [{'name': name, 'value': val} for name, val in zip(clean_feature_names, coefs_lr)],
            key=lambda x: abs(x['value']),
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'rf': {
                'accuracy': float(acc_rf),
                'precision': float(prec_rf),
                'recall': float(rec_rf),
                'f1': float(f1_rf),
                'confusion_matrix': {
                    'tn': cm_rf[0][0], 'fp': cm_rf[0][1],
                    'fn': cm_rf[1][0], 'tp': cm_rf[1][1]
                },
                'feature_importances': rf_features
            },
            'lr': {
                'accuracy': float(acc_lr),
                'precision': float(prec_lr),
                'recall': float(rec_lr),
                'f1': float(f1_lr),
                'confusion_matrix': {
                    'tn': cm_lr[0][0], 'fp': cm_lr[0][1],
                    'fn': cm_lr[1][0], 'tp': cm_lr[1][1]
                },
                'feature_importances': lr_features
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Ejecuta predicciones en base a datos ingresados por el usuario."""
    try:
        global model_rf, model_lr
        if model_rf is None or model_lr is None:
            load_resources()
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No se enviaron datos'}), 400
            
        # Validar y parsear datos recibidos
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Falta el campo requerido: {field}'}), 400
                
        # Crear DataFrame para la predicción
        input_df = pd.DataFrame([{
            'Pclass': int(data['Pclass']),
            'Sex': str(data['Sex']),
            'Age': float(data['Age']),
            'SibSp': int(data['SibSp']),
            'Parch': int(data['Parch']),
            'Fare': float(data['Fare']),
            'Embarked': str(data['Embarked'])
        }])
        
        # Predicción Random Forest
        pred_rf = int(model_rf.predict(input_df)[0])
        prob_rf = float(model_rf.predict_proba(input_df)[0][1])
        
        # Predicción Regresión Logística
        pred_lr = int(model_lr.predict(input_df)[0])
        prob_lr = float(model_lr.predict_proba(input_df)[0][1])
        
        # Generar veredicto o contexto histórico personalizado
        verdict = generate_historical_narrative(data, pred_rf, prob_rf, pred_lr, prob_lr)
        
        return jsonify({
            'success': True,
            'prediction': {
                'rf': {
                    'survived': pred_rf == 1,
                    'probability': prob_rf
                },
                'lr': {
                    'survived': pred_lr == 1,
                    'probability': prob_lr
                }
            },
            'narrative': verdict
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_historical_narrative(data, pred_rf, prob_rf, pred_lr, prob_lr):
    """Genera un veredicto narrativo contextualizado históricamente."""
    sex_es = 'Mujer' if data['Sex'] == 'female' else 'Hombre'
    pclass_name = {1: 'Primera Clase', 2: 'Segunda Clase', 3: 'Tercera Clase'}.get(data['Pclass'], 'Clase Desconocida')
    
    # Calcular promedio de probabilidad de supervivencia
    avg_prob = (prob_rf + prob_lr) / 2
    
    narrative = ""
    if avg_prob >= 0.70:
        narrative += f"### Estatus: Rescatado(a) / Sobreviviente\n\n"
        if data['Sex'] == 'female':
            narrative += f"Como **{sex_es}** en **{pclass_name}**, fuiste priorizada bajo el protocolo oficial de evacuación del Titanic: *'Mujeres y niños primero'*. "
            if data['Pclass'] == 1:
                narrative += "Ubicada en la cubierta superior, tuviste acceso directo e inmediato a los primeros botes salvavidas (como el Bote 4 o el Bote Colapsable D) que salieron antes del hundimiento final. Las probabilidades históricas y estadísticas estuvieron firmemente a tu favor."
            elif data['Pclass'] == 2:
                narrative += "A pesar de la confusión en la cubierta de botes, fuiste conducida a un bote de estribor. La mayoría de las mujeres de segunda clase lograron abordar los botes con éxito y sobrevivir."
            else:
                narrative += "Pese a las barreras lingüísticas, la falta de información y la lejanía de los camarotes de tercera clase en las profundidades del barco, lograste abrirte paso hacia las cubiertas superiores a tiempo para abordar uno de los botes de popa."
        else: # male with high survival probability (usually very young or first class with special conditions)
            if data['Age'] <= 12:
                narrative += f"Al ser un niño de **{int(data['Age'])} años** en **{pclass_name}**, los oficiales te permitieron abordar los botes salvavidas junto a tu madre en cumplimiento de la orden de evacuación. Lograste escapar del naufragio a bordo de un bote seguro."
            else:
                narrative += f"A pesar de ser un hombre adulto en **{pclass_name}**, las circunstancias jugaron a tu favor. Probablemente pudiste abordar uno de los botes de babor bajo la supervisión del Oficial Lightoller (quien permitió hombres si había espacio libre), o bien lograste lanzarte al agua cerca de un bote colapsable y ser rescatado antes de sucumbir a la hipotermia."
    elif avg_prob >= 0.40:
        narrative += f"### Estatus: Destino Incierto / Zona de Riesgo\n\n"
        narrative += f"Tus datos de embarque (**{sex_es}**, **{pclass_name}**, **{int(data['Age'])} años**) te colocan en una franja donde el destino dependió de decisiones de último segundo. "
        if data['Sex'] == 'female' and data['Pclass'] == 3:
            narrative += "Al viajar en Tercera Clase, la demora en abrir las compuertas y el laberinto de pasillos dificultaron tu llegada a cubierta. Tu supervivencia dependió enteramente de si lograste llegar a los botes antes de que se agotaran."
        else:
            narrative += "Es posible que hayas intentado cooperar con la tripulación o buscar a tu familia antes de buscar un bote. Tu supervivencia dependió críticamente de tu cercanía física a la cubierta de botes en los últimos 30 minutos del naufragio."
    else:
        narrative += f"### Estatus: Fallecido(a) / Perdido en el Atlántico\n\n"
        if data['Sex'] == 'male':
            narrative += f"Como **{sex_es}** adulto en **{pclass_name}**, el estricto protocolo de evacuación te impidió acercarte a los botes salvavidas. Permaneciste en la cubierta inclinada ayudando a otros o manteniendo la calma. "
            if data['Pclass'] == 3:
                narrative += "Al estar en tercera clase, las posibilidades de sobrevivir eran mínimas. La tripulación contuvo a muchos pasajeros en las secciones inferiores para evitar el pánico. Probablemente permaneciste con el barco hasta el final, hundiéndote en las gélidas aguas de -2°C a las 2:20 AM."
            else:
                narrative += "A pesar de estar en una clase más acomodada, el Oficial Murdoch o Lightoller no te permitieron abordar debido a la estricta política. Formas parte de la trágica estadística de caballeros que se despidieron con honor en la cubierta."
        else:
            narrative += f"A pesar de ser mujer, viajar en **{pclass_name}** y enfrentar las extremas condiciones del naufragio dificultaron tu evacuación. La lejanía de las cubiertas superiores o el retraso en la alarma resultaron fatales en esta simulación."
            
    return narrative

if __name__ == '__main__':
    # Asegurar que se carguen los recursos antes del arranque
    load_resources()
    app.run(debug=True, port=5000)
