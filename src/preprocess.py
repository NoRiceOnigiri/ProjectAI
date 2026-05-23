import pandas as pd
from sklearn.preprocessing import StandardScaler

input_file = 'ai_student_impact_dataset.xlsx'
output_file = 'ai_student_impact_dataset_processed.xlsx'

df = pd.read_excel(input_file)
df_processed = df.copy()

# Целевая переменная
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_processed['Burnout_Risk_Level'] = df_processed['Burnout_Risk_Level'].map(target_mapping)

# Инженерия признаков
year_mapping = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4, 'Graduate': 5}
df_processed['Year_of_Study_Numeric'] = df_processed['Year_of_Study'].map(year_mapping)
skill_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
df_processed['Prompt_Engineering_Skill_Numeric'] = df_processed['Prompt_Engineering_Skill'].map(skill_mapping)

def calculate_violation_severity(row):
    use_case = row['Primary_Use_Case']
    policy = row['Institutional_Policy']

    risk_level = {'Direct_Answer_Generation': 3, 'Copywriting/Drafting': 2,'Summarizing_Reading': 2,'Debugging/Troubleshooting': 1,'Ideation': 1}.get(use_case, 1)
    if policy == 'Strict_Ban':
        return 2 if risk_level >= 2 else 1
    elif policy == 'Allowed_With_Citation':
        return 1 if risk_level == 3 else 0
    elif 'Actively_Encouraged' in policy:
        return 0
    return 0

df_processed['Policy_Violation_Severity'] = df_processed.apply(calculate_violation_severity, axis=1)
df_processed['Paid_Subscription'] = df_processed['Paid_Subscription'].astype(int)
df_processed['GPA_Change'] = df_processed['Post_Semester_GPA'] - df_processed['Pre_Semester_GPA']
df_processed['AI_Study_Ratio'] = df_processed['Weekly_GenAI_Hours'] / (
    df_processed['Weekly_GenAI_Hours'] + df_processed['Traditional_Study_Hours']
)
df_processed = pd.get_dummies(df_processed, columns=['Major_Category'], prefix='Major', drop_first=True)

# Удаление преобразованного или ненужного
columns_to_drop = ['Student_ID', 'Year_of_Study', 'Prompt_Engineering_Skill', 'Institutional_Policy', 'Primary_Use_Case']
df_processed = df_processed.drop(columns=columns_to_drop)

# Маштабирование числовых признаков
numerical_features = ['Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity', 'Paid_Subscription', 'Traditional_Study_Hours',
    'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams', 'Post_Semester_GPA', 'Skill_Retention_Score', 'Year_of_Study_Numeric',
    'Prompt_Engineering_Skill_Numeric', 'Policy_Violation_Severity', 'GPA_Change', 'AI_Study_Ratio'
]
scaler = StandardScaler()
df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

# Сохранение
df_processed.to_excel(output_file, index=False)
