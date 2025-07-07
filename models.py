import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')

def preprocess_data(df):
    """
    Предобработка данных Kickstarter
    """
    # Создаем копию данных
    data = df.copy()
    
    # Создаем целевую переменную (1 - успешная, 0 - неуспешная)
    data['success'] = (data['state'] == 'successful').astype(int)
    
    # Удаляем строки с отсутствующими значениями в ключевых столбцах
    data = data.dropna(subset=['usd_goal_real', 'usd_pledged_real', 'backers'])
    
    # Обработка дат
    data['launched'] = pd.to_datetime(data['launched'])
    data['deadline'] = pd.to_datetime(data['deadline'])
    data['campaign_duration'] = (data['deadline'] - data['launched']).dt.days
    
    # Создание дополнительных признаков
    data['launch_year'] = data['launched'].dt.year
    data['launch_month'] = data['launched'].dt.month
    data['launch_day_of_week'] = data['launched'].dt.dayofweek
    
    # Категориальные переменные
    label_encoders = {}
    categorical_columns = ['category', 'main_category', 'currency', 'country']
    
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    # Выбираем признаки для модели
    feature_columns = [
        'usd_goal_real', 'backers', 'campaign_duration',
        'launch_year', 'launch_month', 'launch_day_of_week'
    ]
    
    # Добавляем закодированные категориальные признаки
    for col in categorical_columns:
        if col + '_encoded' in data.columns:
            feature_columns.append(col + '_encoded')
    
    # Удаляем выбросы (используем IQR метод для числовых столбцов)
    numeric_features = ['usd_goal_real', 'backers', 'campaign_duration']
    for col in numeric_features:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    return data, feature_columns, label_encoders

def train_model(data, feature_columns):
    """
    Обучение модели логистической регрессии
    """
    # Подготовка данных
    X = data[feature_columns]
    y = data['success']
    
    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    Оценка качества модели
    """
    # Основные метрики
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("=== ОЦЕНКА КАЧЕСТВА МОДЕЛИ ===")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nДетальный отчет по классификации:")
    print(classification_report(y_test, y_pred))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Неуспешная', 'Успешная'],
                yticklabels=['Неуспешная', 'Успешная'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.show()
    
    return accuracy, roc_auc

def analyze_feature_importance(model, feature_columns, X_train_scaled, y_train):
    """
    Анализ важности признаков
    """
    # Коэффициенты логистической регрессии
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
    print(feature_importance)
    
    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['abs_coefficient'])
    plt.xlabel('Абсолютное значение коэффициента')
    plt.title('Важность признаков (абсолютные значения коэффициентов)')
    plt.tight_layout()
    plt.show()
    
    # Статистический анализ признаков
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X_train_scaled, y_train)
    
    feature_scores = pd.DataFrame({
        'feature': feature_columns,
        'f_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('f_score', ascending=False)
    
    print("\n=== F-СТАТИСТИКА ПРИЗНАКОВ ===")
    print(feature_scores)
    
    return feature_importance, feature_scores

def correlation_analysis(data, feature_columns):
    """
    Анализ корреляций между признаками
    """
    correlation_matrix = data[feature_columns + ['success']].corr()
    
    print("\n=== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ===")
    print("Корреляция признаков с целевой переменной:")
    target_corr = correlation_matrix['success'].sort_values(ascending=False)
    print(target_corr)
    
    # Тепловая карта корреляций
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def plot_roc_curve(y_test, y_pred_proba):
    """
    Построение ROC-кривой
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.show()

def main_analysis(df):
    """
    Основная функция анализа
    """
    print("Начинаем анализ данных Kickstarter...")
    
    # Предобработка данных
    data, feature_columns, label_encoders = preprocess_data(df)
    print(f"Данные обработаны. Итоговый размер: {data.shape}")
    print(f"Используемые признаки: {feature_columns}")
    
    # Обучение модели
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_pred_proba = train_model(data, feature_columns)
    
    # Оценка модели
    accuracy, roc_auc = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Анализ важности признаков
    feature_importance, feature_scores = analyze_feature_importance(model, feature_columns, X_train_scaled, y_train)
    
    # Корреляционный анализ
    correlation_matrix = correlation_analysis(data, feature_columns)
    
    # ROC-кривая
    plot_roc_curve(y_test, y_pred_proba)
    
    # Дополнительная статистика
    print("\n=== ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА ===")
    print(f"Общее количество кампаний: {len(data)}")
    print(f"Успешных кампаний: {data['success'].sum()} ({data['success'].mean()*100:.1f}%)")
    print(f"Неуспешных кампаний: {len(data) - data['success'].sum()} ({(1-data['success'].mean())*100:.1f}%)")
    
    # Статистика по категориям
    if 'main_category' in data.columns:
        category_stats = data.groupby('main_category')['success'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        print("\nСтатистика успешности по основным категориям:")
        print(category_stats)
    
    return model, scaler, feature_columns, label_encoders, data

# Пример использования:
# Предполагается, что у вас есть DataFrame с именем 'df'
# model, scaler, feature_columns, label_encoders, processed_data = main_analysis(df)

# Функция для предсказания новых кампаний
def predict_campaign_success(model, scaler, feature_columns, new_data):
    """
    Предсказание успешности новой кампании
    """
    # Предобработка новых данных (аналогично обучающим)
    new_data_scaled = scaler.transform(new_data[feature_columns])
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[:, 1]
    
    return prediction, probability

print("Код готов к использованию!")
print("Для запуска анализа используйте: main_analysis(your_dataframe)")

main_analysis(df)