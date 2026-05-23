import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

# Загрузка
input_file = 'ai_student_impact_dataset_processed.xlsx'
df = pd.read_excel(input_file)

# Разделение на тестовую и тренировочную
X = df.drop('Burnout_Risk_Level', axis=1)
y = df['Burnout_Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100, stratify=y
)
print(f"Тренировочный набор: {X_train.shape[0]} примеров")
print(f"Тестовый набор: {X_test.shape[0]} примеров")

# Перебалансировка классов (анализ показал, что целевая переменная не совсем равномерная)
smote = SMOTE(random_state=100, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Функция для метрик
def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

# Первая модели - логистическая регрессия
# Перебор гиперпараметров
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['balanced'],
    'solver': ['lbfgs', 'saga']
}
lr_base = LogisticRegression(
    max_iter=5000,
    random_state=100,
    multi_class='multinomial'
)
grid_search_lr = GridSearchCV(
    lr_base,
    param_grid_lr,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search_lr.fit(X_train_balanced, y_train_balanced)

lr_model = grid_search_lr.best_estimator_
y_pred_lr = lr_model.predict(X_test)
lr_metrics = calculate_metrics(y_test, y_pred_lr)

print(f"Weighted F1 (CV): {grid_search_lr.best_score_:.4f}")
print(f"Accuracy (test): {lr_metrics['accuracy']:.4f}")
print(f"Weighted F1 (test): {lr_metrics['weighted_f1']:.4f}")
print(f"MAE (test): {lr_metrics['mae']:.4f}")
print(f"RMSE (test): {lr_metrics['rmse']:.4f}")
print(f"R^2 (test): {lr_metrics['r2']:.4f}")

# Вторая модель - рандомный лес
# Перебор гиперпараметров
param_grid_rf = {
    'n_estimators': [300, 500],
    'max_depth': [20, 30, 50],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
rf_base = RandomForestClassifier(
    random_state=100,
    n_jobs=-1,
    class_weight='balanced'
)
grid_search_rf = GridSearchCV(
    rf_base,
    param_grid_rf,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search_rf.fit(X_train_balanced, y_train_balanced)

rf_model = grid_search_rf.best_estimator_
y_pred_rf = rf_model.predict(X_test)
rf_metrics = calculate_metrics(y_test, y_pred_rf)

print(f"Weighted F1 (CV): {grid_search_rf.best_score_:.4f}")
print(f"Accuracy (test): {rf_metrics['accuracy']:.4f}")
print(f"Weighted F1 (test): {rf_metrics['weighted_f1']:.4f}")
print(f"MAE (test): {rf_metrics['mae']:.4f}")
print(f"RMSE (test): {rf_metrics['rmse']:.4f}")
print(f"R^2 (test): {rf_metrics['r2']:.4f}")

# Третья модель - градиентный бустинг
# Перебор гиперпараметров
param_dist_lgb = {
    'num_leaves': [100, 200],
    'learning_rate': [0.001, 0.01],
    'n_estimators': [500, 1000],
    'max_depth': [10, 20, 30],
    'min_child_samples': [2, 5],
    'subsample': [0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0.01, 0.1],
    'reg_lambda': [0.1, 0.5]
}
lgb_base = lgb.LGBMClassifier(
    random_state=100,
    n_jobs=-1,
    verbose=-1,
    objective='multiclass',
    class_weight='balanced',
    num_class=3,
    metric='multi_logloss'
)
random_search_lgb = RandomizedSearchCV(
    lgb_base,
    param_dist_lgb,
    n_iter=20,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100,
    verbose=1
)
random_search_lgb.fit(X_train_balanced, y_train_balanced)

lgb_model = random_search_lgb.best_estimator_
y_pred_lgb = lgb_model.predict(X_test)
lgb_metrics = calculate_metrics(y_test, y_pred_lgb)

print(f"Weighted F1 (CV): {random_search_lgb.best_score_:.4f}")
print(f"Accuracy (test): {lgb_metrics['accuracy']:.4f}")
print(f"Weighted F1 (test): {lgb_metrics['weighted_f1']:.4f}")
print(f"MAE (test): {lgb_metrics['mae']:.4f}")
print(f"RMSE (test): {lgb_metrics['rmse']:.4f}")
print(f"R^2 (test): {lgb_metrics['r2']:.4f}")

# Сравнение
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'LightGBM'],
    'Accuracy': [lr_metrics['accuracy'], rf_metrics['accuracy'], lgb_metrics['accuracy']],
    'Weighted F1': [lr_metrics['weighted_f1'], rf_metrics['weighted_f1'], lgb_metrics['weighted_f1']],
    'MAE': [lr_metrics['mae'], rf_metrics['mae'], lgb_metrics['mae']],
    'RMSE': [lr_metrics['rmse'], rf_metrics['rmse'], lgb_metrics['rmse']],
    'R^2': [lr_metrics['r2'], rf_metrics['r2'], lgb_metrics['r2']]
})

print(results.to_string(index=False))

# Выбор лучшей
best_model_name = results.loc[results['Weighted F1'].idxmax(), 'Model']
print(f"Лучшая модель: {best_model_name}")

if best_model_name == 'Logistic Regression':
    best_model = lr_model
    best_predictions = y_pred_lr
    best_metrics = lr_metrics
elif best_model_name == 'Random Forest':
    best_model = rf_model
    best_predictions = y_pred_rf
    best_metrics = rf_metrics
else:
    best_model = lgb_model
    best_predictions = y_pred_lgb
    best_metrics = lgb_metrics

# Анализ лучшей модели

print(f"Итоговые метрики:")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Weighted F1: {best_metrics['weighted_f1']:.4f}")
print(f"MAE: {best_metrics['mae']:.4f}")
print(f"RMSE: {best_metrics['rmse']:.4f}")
print(f"R^2: {best_metrics['r2']:.4f}")

print("\n")
print("Classification Report:")
print(classification_report(y_test, best_predictions,
                            target_names=['Low (0)', 'Medium (1)', 'High (2)']))

print("\n")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
cm_df = pd.DataFrame(
    cm,
    index=['Actual Low', 'Actual Medium', 'Actual High'],
    columns=['Pred Low', 'Pred Medium', 'Pred High']
)
print(cm_df)

# Сохраннение модели
model_path = 'best_model.pkl'
joblib.dump(best_model, model_path)
