import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ai_student_impact_dataset.csv.csv')

print("Общая информация о даатсете")
print(f"Форма: {df.shape}"\n)
print(f"Колонки ({len(df.columns)}):")

print("Информация о столбцах")
for col in df.columns:
    print(f"\n")
    print(f"{col}")
    print(f"Тип: {df[col].dtype}")
    print(f"Пропуски: {df[col].isnull().sum()} ({df[col].isnull().sum() / len(df) * 100:.1f}%)")
    print(f"Уникальных значений: {df[col].nunique()}")

    if df[col].dtype == 'object':
        print(f"  Примеры: {df[col].head(10).tolist()}")
    else:
        print(f"  Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}")
