from pathlib import Path
import pandas as pd

N = 100  # Сколько столбцов нужно сохранить из исходного датасета в новый файл
input_path = Path("ai_student_impact_dataset.xlsx")
output_path = Path(f"ai_student_impact_dataset_{N}.xlsx")

if not input_path.exists():
	raise FileNotFoundError(f"Файл не найден: {input_path}")

in_suffix = output_path.suffix.lower()
if in_suffix in [".xlsx", ".xls"]:
	df = pd.read_excel(input_path, engine="openpyxl", nrows=n)
else:
	df = pd.read_csv(input_path, nrows=n)

out_suffix = output_path.suffix.lower()
if out_suffix in [".xlsx", ".xls"]:
	df.to_excel(output_path.with_suffix(".xlsx"), index=False, engine="openpyxl")
elif out_suffix == ".csv":
	df.to_csv(output_path.with_suffix(".csv"), index=False)

print(f"Сохранено первые {N} строк в {output_path}")

