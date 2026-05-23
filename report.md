# Отчёт

**Проект:** Влияние использования искусственного интеллекта на выгорание у студентов.  

---

## 1. Goal and metrics

Цель проекта – построить мультиклассовую модель машинного обучения, которая предсказывает уровень риска выгорания у студентов в зависимости от меры использования искусственного интеллекта (ИИ) в процессе обучения. 
Вероятность выгорания - категориальный параметр, который может быть представлен как три класса: "Low", "Medium" или "High". 

### Метрики

| Метрика | Использование |
|---|---|
| `Accuracy` | Простая оценка общей точности модели, чувствительна к дисбалансам |
| `Weighted F1 ` | Среднее взвешенное значение F1, объективная оценка модели |
| `MAE` | Средняя ошибка предсказания в виде различия между классами |
| `RMSE` | Квадратный корень из среднего квадрата ошибок |

Основная метрика - **Weighted F1**. Причины:
1. Учитывает дисбаланс классов
2. Учитывает как ложноположительные, так и ложноотрицательные ошибки
3. Работает с мультиклассовыми задачами

---

## 2. Data

Основной датасет: ai_student_impact_dataset.csv
16 параметров, 50000 строк (до препроцессинга)

### Распределение классов Burnout_Risk_Level
| Класс | Значение |
|---|---|
| Medium | 21144 |
| Low | 16369 |
| High | 12487 |

### Пропорции классов Burnout_Risk_Level: 
| Класс | Значение |
|---|---|
| Medium | 0.42288 |
| Low | 0.32738 |
| High | 0.24974 |

### Признаки

| Тип | Признаки |
|---|---|
| Булевые | `Paid_Subscription` |
| Категориальные | `Major_Category`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Institutional_Policy` |
| Числовые | `Student_ID`, `Pre_Semester_GPA`, `Weekly_GenAI_Hours`, `Tool_Diversity`, `Traditional_Study_Hours`, `Perceived_AI_Dependency`, `Anxiety_Level_During_Exams`, `Post_Semester_GPA `, `Skill_Retention_Score` |
| Искомый признак | `Burnout_Risk_Level` |

---

## 3. Validation draft

### Разбиение данных на train и test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=46, stratify=y
)
```

### Балансировка классов

```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### Кросс-валидация

```python
grid_search_lr = GridSearchCV(
    lr_base,
    param_grid_lr,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search_lr.fit(X_train_balanced, y_train_balanced)
```

Такая модификация выбрана из-за того, что в исходном датасете мультиклассовая классификация с вероятным дисбалансом. Данная модификация сохраняет пропорции классов. 

---

## 4. Approach

### Переопределение искомой переменной
Burnout_Risk_Level - категориальная переменная, поэтому для удобства необходимо перевести его в числовые значения:
- Low -> 0
- Medium -> 1
- High -> 2 

### Feature engineering

1. Year_of_Study -> Year_of_Study_Numeric. Замена категориального признака на числовой 
2. Prompt_Engineering_Skill -> Prompt_Engineering_Skill_Numeric. Также замена категориального признака на числовой
3. Policy_Violation_Severity - новый добавленный признак. Рассчитывается на основе Primary_Use_Case и Institutional_Policy. Если студенты используют ИИ при строгой политике использования ИИ, то их уровень стресса выше
4. GPA_Change - новый добавленный признак. Рассчитывается на основе Pre_Semester_GPA и Post_Semester_GPA. Означает то, как изменилась успеваемость студента в течение семестра
5. AI_Study_Ratio - новый добавленный признак. Рассчитывается на основе Weekly_GenAI_Hours и Traditional_Study_Hours. Означает долю времени, затраченного на обучение с помощью ИИ, относительно общего времени обучения

### One-Hot Encoding

```python
df_processed = pd.get_dummies(
    df_processed,
    columns=['Major_Category'],
    prefix='Major',
    drop_first=True
)
```

### Удаление лишних столбцов

1. Student_ID
2. Year_of_Study
3. Prompt_Engineering_Skill
4. Institutional_Policy
5. Primary_Use_Case

### Масштабирование числовых признаков

```python
scaler = StandardScaler()
df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
```

---

## 5. Training

| Модель | Что делает |
|---|---|
| `Logistic Regression` | baseline, хороша для первичной оценки |
| `Random Forest` | ансамблевая модель, устойчив к переобучению |
| `LightGBM` | градиентный бустинг, хорошо работает с большими данными |

### Сравнение моделей

| Model | Accuracy | Weighted F1 | MAE | RMSE |
|---|---|---|---|---|
| Logistic Regression | 0.5176 | 0.5032 | 0.5467 | 0.8218 |
| Random Forest | 0.5207 | 0.5210 | 0.5266 | 0.7881 |
| LightGBM | 0.5277 | 0.5289 | 0.5112 | 0.7674 |

В результате сравнения была выбрана модель LightGBM
Лучшие параметры для данной модели:

```python
{'subsample': 0.9, 
'reg_lambda': 0.5, 
'reg_alpha': 0.1, 
'num_leaves': 200, 
'n_estimators': 1000, 
'min_child_samples': 2, 
'max_depth': 30, 
'learning_rate': 0.01, 
'colsample_bytree': 0.8}
```

Лучшая метрика CV: `Weighted F1: 0.5657`

---

## 6. Validation

Лучшая модель - `LightGBM`.

### Финальные результаты на тестовом датасете:

| Metric | Value |
|---|---|
| Accuracy | 0.5457 |
| Weighted F1 | 0.5384 |
| MAE | 0.5332 |
| RMSE | 0.8313 |

### Classification Report:
| class | precision | recall | f1-score | support |
|---|---|---|---|---|
| Low (0) | 0.52 | 0.53 | 0.52 | 4092 |
| Medium (1)| 0.49 | 0.55 | 0.52 | 5286 |
| High (2) | 0.65 | 0.49 | 0.56 | 3122|
| accuracy | | | 0.53 | 12500 |
| macro avg | 0.55 | 0.52 | 0.53 | 12500 |
| weighted avg | 0.54 | 0.53 | 0.53 | 12500 |

### Confusion Matrix:
| | Pred Low | Pred Medium | Pred High |
|---|---|---|---|
| Actual Low | 2159 | 1791 | 142 |
| Actual Medium | 1686 | 2906 | 694 |
| Actual High | 344 | 1247 | 1531 |

### Выводы
1. Модель правильно классифицирует 53% случаев.
2. Модель лучше всего предсказывает класс High по precision (65%), но плохо находит истинные High (recall = 49%).
3. Основная проблема в том, что модель неправильно оценивает риск для класса High.
4. Классы Low и Medium сильно перекрываются. Модель не может четко разделить их.

---

## 7. Model selection

Финальная модель - `LightGBM`.

Лучшие параметры для данной модели:

```python
{'subsample': 0.9, 
'reg_lambda': 0.5, 
'reg_alpha': 0.1, 
'num_leaves': 200, 
'n_estimators': 1000, 
'min_child_samples': 2, 
'max_depth': 30, 
'learning_rate': 0.01, 
'colsample_bytree': 0.8}
```

Топ-10 важных признаков (LightGBM):
| Feature | Importance |
|---|---|
| Skill_Retention_Score | 68756 |
| Traditional_Study_Hours | 66610 |
| GPA_Change | 66256 |
| Weekly_GenAI_Hours | 63619 |
| Pre_Semester_GPA | 61797 |
| AI_Study_Ratio | 56299 |
| Post_Semester_GPA | 55019 |
| Anxiety_Level_During_Exams | 35986 |
| Perceived_AI_Dependency | 30764 |
| Year_of_Study_Numeric | 26777 |


### Причины выбора модели

1. Гибкость в обработке категориальных и числовых данных
2. Эффективное использование памяти
3. Учёт дисбаланса и сложные зависимости в данных
4. Быстрое обучение
