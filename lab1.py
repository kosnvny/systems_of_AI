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

# todo что-то не так с графиками
plt.figure(figsize=(10, 5))
sns.histplot(train_df["Visits"], bins=10, kde=True)
plt.title("Распределение Visits (таргета) в обучающей выборке")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=train_df["Visits"])
plt.title("Boxplot Visits")
plt.show()
# todo конец неправильных графиков

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