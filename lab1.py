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

numeric_columns = ["Rating"]

df = df.dropna(subset=numeric_columns)

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

