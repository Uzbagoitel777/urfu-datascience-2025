import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')

def normalize_columns(df):
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

    return data, categorical_columns, label_encoders


def preprocess_data(df):
    """
    Предобработка данных Kickstarter
    """
    # Создаем копию данных
    data, categorical_columns, label_encoders = normalize_columns(df)
    
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

def train_models(data, feature_columns, pretrained=False):
    """
    Обучение всех моделей: логистическая регрессия, дерево решений, нейронная сеть
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
    
    print("Обучение моделей...")

    if pretrained:
        models = {'Logistic': {},
                  'Tree': {},
                  'MLP':{}}
        data_list = [['Logistic', 'model', 'predictions', 'probabilities'],
                     ['Tree', 'model', 'predictions', 'probabilities', 'best_params'],
                     ['MLP', 'model', 'predictions', 'probabilities', 'best_params']]

        print('Загрузка моделей из памяти...')

        for model in data_list:
            model_name = model.pop(0)
            for file in model:
                try:
                    with open(f"pretrained_models/{model_name}_{file}.pkl", "rb") as f:
                        models[model_name][file] = pickle.load(f)
                    print(f'Файл {model_name}_{file}.pkl загружен')
                except Exception as e:
                    print(f'Произошла ошибка во время загрузки: {e}')

        models['Logistic']['best_params'] = None

        print('Загрузка завершена. Словарь моделей:')
        print(models)

        return models, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    # === 1. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ===
    print("Обучение логистической регрессии...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # === 2. ДЕРЕВО РЕШЕНИЙ ===
    print("Обучение дерева решений...")
    
    # Подбор гиперпараметров для дерева решений
    dt_params = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring='roc_auc', n_jobs=-1)
    dt_grid.fit(X_train, y_train)  # Деревья не требуют масштабирования
    
    dt_best = dt_grid.best_estimator_
    dt_pred = dt_best.predict(X_test)
    dt_pred_proba = dt_best.predict_proba(X_test)[:, 1]
    
    # === 3. НЕЙРОННАЯ СЕТЬ ===
    print("Обучение нейронной сети...")
    
    # Подбор гиперпараметров для нейронной сети
    nn_params = {
        'hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 10)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.00001, 0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # nn_model = MLPClassifier(random_state=42, max_iter=200)
    # nn_grid = GridSearchCV(nn_model, nn_params, cv=3, scoring='roc_auc', n_jobs=-1)
    nn_model = MLPClassifier(random_state=42, max_iter=100)
    nn_grid = GridSearchCV(nn_model, nn_params, cv=2, scoring='roc_auc', n_jobs=-1)
    nn_grid.fit(X_train_scaled, y_train)
    
    nn_best = nn_grid.best_estimator_
    nn_pred = nn_best.predict(X_test_scaled)
    nn_pred_proba = nn_best.predict_proba(X_test_scaled)[:, 1]
    
    models = {
        'Logistic': {
            'model': lr_model,
            'predictions': lr_pred,
            'probabilities': lr_pred_proba,
            'best_params': None
        },
        'Tree': {
            'model': dt_best,
            'predictions': dt_pred,
            'probabilities': dt_pred_proba,
            'best_params': dt_grid.best_params_
        },
        'MLP': {
            'model': nn_best,
            'predictions': nn_pred,
            'probabilities': nn_pred_proba,
            'best_params': nn_grid.best_params_
        }
    }

    for model, files in models.items():
        for file, parameter in files.items():
            with open(f"pretrained_models/{model}_{file}.pkl", "wb") as f:
                pickle.dump(parameter, f)

    print('Обучение моделей завершено, их данные сохранены в pretrained_models')
    
    return models, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

def evaluate_models(models, y_test):
    """
    Оценка качества всех моделей и их сравнение
    """
    results = {}
    
    print("=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
    print("-" * 80)
    
    for name, model_info in models.items():
        y_pred = model_info['predictions']
        y_pred_proba = model_info['probabilities']
        
        # Основные метрики
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Дополнительные метрики из classification_report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        print(f"\n{name}:")
        print(f"  Точность (Accuracy): {accuracy:.4f}")
        print(f"  ROC-AUC Score: {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        
        if model_info['best_params']:
            print(f"  Лучшие параметры: {model_info['best_params']}")
    
    # Создание таблицы сравнения
    comparison_df = pd.DataFrame(results).T
    print("\n=== ТАБЛИЦА СРАВНЕНИЯ ===")
    print(comparison_df.round(4))
    
    # Определение лучшей модели по ROC-AUC
    best_model_name = comparison_df['roc_auc'].idxmax()
    print(f"\nЛучшая модель по ROC-AUC: {best_model_name} ({comparison_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    return results, comparison_df

def plot_model_comparison(results, y_test, models):
    """
    Визуализация сравнения моделей
    """
    # 1. Барplot сравнения метрик
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['accuracy', 'roc_auc', 'precision', 'f1_score']
    metric_names = ['Точность', 'ROC-AUC', 'Precision', 'F1-Score']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        model_names = list(results.keys())
        values = [results[model][metric] for model in model_names]
        
        bars = ax.bar(model_names, values, color=['blue', 'green', 'red'])
        ax.set_title(f'Сравнение по {metric_name}')
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 2. ROC-кривые для всех моделей
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    for i, (name, model_info) in enumerate(models.items()):
        y_pred_proba = model_info['probabilities']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайная модель')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые для всех моделей')
    plt.legend(loc="lower right")
    plt.show()
    
    # 3. Матрицы ошибок для всех моделей
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, model_info) in enumerate(models.items()):
        y_pred = model_info['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Неуспешная', 'Успешная'],
                   yticklabels=['Неуспешная', 'Успешная'])
        axes[i].set_title(f'Матрица ошибок - {name}')
        axes[i].set_xlabel('Предсказанные значения')
        axes[i].set_ylabel('Истинные значения')
    
    plt.tight_layout()
    plt.show()

def analyze_decision_tree(dt_model, feature_columns):
    """
    Анализ дерева решений
    """
    print("\n=== АНАЛИЗ ДЕРЕВА РЕШЕНИЙ ===")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Важность признаков в дереве решений:")
    print(feature_importance)
    
    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Важность признака')
    plt.title('Важность признаков в дереве решений')
    plt.tight_layout()
    plt.show()
    
    print(f"Глубина дерева: {dt_model.get_depth()}")
    print(f"Количество листьев: {dt_model.get_n_leaves()}")
    
    return feature_importance

def analyze_neural_network(nn_model, feature_columns):
    """
    Анализ нейронной сети
    """
    print("\n=== АНАЛИЗ НЕЙРОННОЙ СЕТИ ===")
    print(f"Архитектура: {nn_model.hidden_layer_sizes}")
    print(f"Функция активации: {nn_model.activation}")
    print(f"Количество итераций: {nn_model.n_iter_}")
    print(f"Значение функции потерь: {nn_model.loss_:.6f}")
    
    # График обучения (если доступен)
    if hasattr(nn_model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(nn_model.loss_curve_)
        plt.title('Кривая обучения нейронной сети')
        plt.xlabel('Эпоха')
        plt.ylabel('Значение функции потерь')
        plt.grid(True)
        plt.show()
    
    return nn_model.loss_

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

def main_analysis(df, pretrained=False):
    """
    Основная функция анализа с тремя моделями
    """
    print("Начинаем анализ данных Kickstarter...")
    
    # Предобработка данных
    data, feature_columns, label_encoders = preprocess_data(df)
    print(f"Данные обработаны. Итоговый размер: {data.shape}")
    print(f"Используемые признаки: {feature_columns}")
    
    # Обучение всех моделей
    models, scaler, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = train_models(data, feature_columns, pretrained)
    
    # Оценка и сравнение моделей
    results, comparison_df = evaluate_models(models, y_test)
    
    # Визуализация сравнения
    plot_model_comparison(results, y_test, models)
    
    # Анализ важности признаков для логистической регрессии
    lr_feature_importance, lr_feature_scores = analyze_feature_importance(
        models['Logistic']['model'], feature_columns, X_train_scaled, y_train
    )
    
    # Анализ дерева решений
    dt_feature_importance = analyze_decision_tree(
        models['Tree']['model'], feature_columns
    )
    
    # Анализ нейронной сети
    nn_loss = analyze_neural_network(
        models['MLP']['model'], feature_columns
    )
    
    # Корреляционный анализ
    correlation_matrix = correlation_analysis(data, feature_columns)
    
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
    
    # Сравнение важности признаков между моделями
    print("\n=== СРАВНЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ ===")
    importance_comparison = pd.DataFrame({
        'Logistic': lr_feature_importance.set_index('feature')['abs_coefficient'],
        'Tree': dt_feature_importance.set_index('feature')['importance']
    }).fillna(0)
    
    print(importance_comparison.sort_values('Tree', ascending=False))
    
    # Визуализация сравнения важности
    plt.figure(figsize=(12, 8))
    importance_comparison.plot(kind='barh', figsize=(12, 8))
    plt.title('Сравнение важности признаков между моделями')
    plt.xlabel('Важность')
    plt.tight_layout()
    plt.show()
    
    return models, scaler, feature_columns, label_encoders, data, results, comparison_df

# Функция для предсказания новых кампаний
def predict_campaign_success(models, scaler, feature_columns, new_data, model_name='Logistic'):
    """
    Предсказание успешности новой кампании с выбранной моделью
    """
    if model_name not in models:
        print(f"Модель {model_name} не найдена. Доступные модели: {list(models.keys())}")
        return None, None
    
    model = models[model_name]['model']
    
    # Предобработка новых данных
    if model_name == 'Tree':
        # Дерево решений не требует масштабирования
        new_data_processed = new_data[feature_columns]
    else:
        # Логистическая регрессия и нейронная сеть требуют масштабирования
        new_data_processed = scaler.transform(new_data[feature_columns])
    
    prediction = model.predict(new_data_processed)
    probability = model.predict_proba(new_data_processed)[:, 1]
    
    return prediction, probability

def get_model_recommendations(comparison_df, results):
    """
    Рекомендации по выбору модели на основе результатов
    """
    print("\n=== РЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ ===")
    
    best_accuracy = comparison_df['accuracy'].idxmax()
    best_roc_auc = comparison_df['roc_auc'].idxmax()
    best_precision = comparison_df['precision'].idxmax()
    best_f1 = comparison_df['f1_score'].idxmax()
    
    print(f"Лучшая по точности: {best_accuracy} ({comparison_df.loc[best_accuracy, 'accuracy']:.4f})")
    print(f"Лучшая по ROC-AUC: {best_roc_auc} ({comparison_df.loc[best_roc_auc, 'roc_auc']:.4f})")
    print(f"Лучшая по Precision: {best_precision} ({comparison_df.loc[best_precision, 'precision']:.4f})")
    print(f"Лучшая по F1-Score: {best_f1} ({comparison_df.loc[best_f1, 'f1_score']:.4f})")
    
    # Общие рекомендации
    print("\nОбщие рекомендации:")
    
    if best_roc_auc == 'Logistic':
        print("- Логистическая регрессия показывает хорошие результаты и легко интерпретируется")
    elif best_roc_auc == 'Tree':
        print("- Дерево решений обеспечивает высокую интерпретируемость и хорошее качество")
    else:
        print("- Нейронная сеть показывает лучшие результаты, но менее интерпретируема")
    
    # Анализ разности в производительности
    max_diff = comparison_df['roc_auc'].max() - comparison_df['roc_auc'].min()
    if max_diff < 0.05:
        print("- Разность в производительности моделей невелика (<5%), выбирайте по интерпретируемости")
    elif max_diff < 0.1:
        print("- Умеренная разность в производительности (5-10%), стоит учесть сложность модели")
    else:
        print("- Значительная разность в производительности (>10%), приоритет лучшей модели")
