# Install dependencies as needed:
import kagglehub
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

"""
green = "\033[32m"
  YELLOW = "\033[33m"
  blue = "\033[36m"
  purple = "\033[35m"
"""

def parse_number(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace(",", "")

    if value == "":
        return np.nan

    multiplier = 1

    if value[-1].upper() == "K":
        multiplier = 1_000
        value = value[:-1]

    elif value[-1].upper() == "M":
        multiplier = 1_000_000
        value = value[:-1]

    elif value[-1].upper() == "B":
        multiplier = 1_000_000_000
        value = value[:-1]

    return float(value) * multiplier


def format_string(string, colour):
    print(colour + string + "\033[0m")

# Set the path to the file you'd like to load
path = kagglehub.dataset_download("biggiefats/roblox-games-dataset")
print(path)
file_path = os.path.join(path, "roblox_games.csv")

df = pd.read_csv(file_path)

format_string("Размер датасета: ", "\033[35m")
print(df.shape[0])

format_string("\nСтолбцы данных: ", "\033[35m")
for i in range(df.shape[1]):
    print(df.columns.tolist()[i])

format_string("\nТипы данных", "\033[35m")
print(df.dtypes)

print(df.isnull().sum())

numeric_columns = [
    "Active",
    "Visits",
    "Favourites",
    "Likes",
    "Dislikes"
]

for column in numeric_columns:
    df[column] = df[column].apply(parse_number)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df = df.dropna(subset=numeric_columns + ["Rating"])

print("Количество дубликатов:", df.duplicated().sum())

df = df.drop_duplicates()

target = "Visits"
format_string("Таргет: ", "\033[36m")
print("Visits (количество посещений одной игры в Roblox)")
format_string("Причина: ", "\033[36m")
print("С помощью Visits можно охарактеризовать и популярность игры, и интерес игроков к ней, и успех игры на платформе")

feature_reasons = {
    "Active": "По активности пользователей можно понять популярность игры",
    "Favourites": "По количеству добавлений в избранное можно понять интерес к игре",
    "Likes": "По лайкам можно понять общую положительную оценку игры",
    "Dislikes": "По дизлайкам можно понять общую отрицательную оценку игры",
    "Rating": "Рейтинг является оценкой игры с учётом лайков и дизлайков"
}

features = []

format_string("Были выбраны такие признаки:", "\033[36m")

for feature, reason in feature_reasons.items():
    print("\t--------")
    print("\t" + feature)
    format_string("\tПричина:", "\033[34m")
    print("\t" + reason)
    features.append(feature)

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Размер X_train:", x_train.shape[0])

print("Размер X_test:", x_test.shape[0])

print("Размер y_train:", y_train.shape[0])

print("Размер y_test:", y_test.shape[0])

sns.set_theme(style="whitegrid")

pd.set_option(
    "display.max_columns",
    None
)

train_df = x_train.copy()
train_df["Visits"] = y_train
format_string("Описательная статистика", "\033[35m")
print(train_df.describe())

# todo анализ на выбросы + ассиметрию + отличия признаков по масштабу
train_df.hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.show()
# todo конец анализа

plt.figure(figsize=(10, 5))
sns.histplot(train_df["Visits"], bins=30, kde=True)
plt.title("Распределение Visits (таргета) в обучающей выборке")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=train_df["Visits"])
plt.title("Boxplot Visits")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица TRAIN")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3

    sns.scatterplot(data=train_df, x=feature, y="Visits", ax=axes[row, col])

    axes[row, col].set_title(f"{feature} vs Visits")
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel("Visits")

axes[1, 2].remove()

plt.tight_layout()
plt.show()

print()

# линейная регрессия без регуляризации

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficients": linear_model.coef_
})
print(coefficients)
format_string("Intercept:", "\033[35m")
print(linear_model.intercept_)

y_train_pred = linear_model.predict(x_train)
y_test_pred = linear_model.predict(x_test)

def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{dataset_name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.4f}")
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

linear_train_metrics = calculate_metrics(y_train, y_train_pred, "Linear Regression - TRAIN")
linear_test_metrics = calculate_metrics(y_test, y_test_pred, "Linear Regression - TEST")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred)
plt.xlabel("Реальные Visits")
plt.ylabel("Предсказанные Visits")
plt.title("Linear Regression: реальные vs предсказанные")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_pred)
minn = min(y_test.min(), y_test_pred.min())
maxx = max(y_test.max(), y_test_pred.max())

plt.plot([minn, maxx], [minn, maxx], "r--")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

residuals = y_test - y_test_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Распределение ошибок Linear Regression")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Предсказанные значения")
plt.ylabel("Остатки")
plt.title("Residuals vs Predictions")
plt.show()

train_data = x_train.copy()
train_data["Visits"] = y_train

def remove_outliers_iqr(data, columns):
    result = data.copy()
    for column in columns:
        Q1 = result[column].quantile(0.25)
        Q3 = result[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        result = result[(result[column] >= lower_bound) & (result[column] <= upper_bound)]
    return result

outlier_columns = ["Active", "Favourites", "Likes", "Dislikes"]
train_clean = remove_outliers_iqr(train_data, outlier_columns)
print("До удаления выбросов:", train_data.shape[0])
print("После удаления выбросов:", train_clean.shape[0])

x_train_clean = train_clean[features]
y_train_clean = train_clean[target]

ridge_pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
ridge_pipeline.fit(x_train_clean, y_train_clean)
y_train_ridge_pred = ridge_pipeline.predict(x_train_clean)
y_test_ridge_pred = ridge_pipeline.predict(x_test)
ridge_train_metrics = calculate_metrics(y_train_clean, y_train_ridge_pred, "Ridge Regression - TRAIN")
ridge_test_metrics = calculate_metrics(y_test, y_test_ridge_pred, "Ridge Regression - TEST")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_test_ridge_pred)

min_value = min(y_test.min(), y_test_ridge_pred.min())
max_value = max(y_test.max(), y_test_ridge_pred.max())

plt.plot([min_value, max_value], [min_value, max_value], "r--")
plt.xlabel("Реальные Visits")
plt.ylabel("Предсказанные Visits")
plt.title("Ridge Regression: Actual vs Predicted")
plt.show()

ridge_residuals = y_test - y_test_ridge_pred
plt.figure(figsize=(10, 5))
sns.histplot(ridge_residuals, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Распределение ошибок Ridge Regression")
plt.show()

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression"
    ],
    "RMSE": [
        linear_test_metrics["RMSE"],
        ridge_test_metrics["RMSE"]
    ],
    "MAE": [
        linear_test_metrics["MAE"],
        ridge_test_metrics["MAE"]
    ],
    "R2": [
        linear_test_metrics["R2"],
        ridge_test_metrics["R2"]
    ]
})

print(comparison)

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison, x="Model", y="R2")
plt.title("Сравнение моделей по R2")
plt.xlabel("Модель")
plt.ylabel("R2")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

comparison_plot = comparison.melt(
    id_vars="Model",
    value_vars=["RMSE", "MAE"],
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_plot, x="Metric", y="Value", hue="Model")
plt.title("Сравнение моделей по основным метрикам")
plt.xlabel("Метрика")
plt.ylabel("Значение")
plt.tight_layout()
plt.show()