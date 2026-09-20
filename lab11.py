import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats

df = pd.read_csv("roblox_games.csv")

df = df.drop_duplicates().copy()


def convert_number(x):
    x = str(x).replace(",", "").strip()

    if x.endswith("B"):
        return float(x[:-1]) * 1_000_000_000
    elif x.endswith("M"):
        return float(x[:-1]) * 1_000_000
    elif x.endswith("K"):
        return float(x[:-1]) * 1_000
    else:
        return float(x.replace("#", ""))


numeric_cols = ["Rank", "Active", "Visits", "Favourites", "Likes", "Dislikes"]
for col in numeric_cols:
    df[col] = df[col].apply(convert_number)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df["Date"] = pd.Timestamp("2024-10-27")

# todo group by + агрегация
# Как отличаются средние показатели популярных игр от менее популярных в зависимости от диапазона рейтинга?
df["Rating_group"] = pd.cut(df["Rating"], bins=[0, 50, 70, 85, 100], labels=["Низкий", "Средний", "Высокий", "Очень высокий"])
rating_stats = df.groupby("Rating_group", observed=True).agg(games=("Name", "count"), mean_active=("Active", "mean"), mean_visits=("Visits", "mean"), mean_favourites=("Favourites", "mean"), mean_likes=("Likes", "mean")).reset_index()
print(rating_stats.to_string())

print("=" * 50)
# todo resample
# Как меняется среднее количество активных игроков по временным периодам?
# НО истории изменений в датасете НЕТ(
df_demo = df.copy()
df_demo["Date_demo"] = pd.date_range(start="2024-01-01", periods=len(df_demo), freq="D")
df_demo = df_demo.set_index("Date_demo")
monthly_active = df_demo["Active"].resample("ME").mean()
monthly_active.plot(figsize=(12, 5))

plt.title("Среднее количество Active по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Среднее Active")
plt.grid()
plt.show()

# todo merge
# Как соотносятся числовая оценка игры и словесная категория её рейтинга?
rating_groups = pd.DataFrame({
    "Rating_group": [
        "Низкий",
        "Средний",
        "Высокий",
        "Очень высокий"
    ],
    "Description": [
        "Низкая оценка",
        "Средняя оценка",
        "Высокая оценка",
        "Очень высокая оценка"
    ]
})
df_merged = df.merge(rating_groups, on="Rating_group", how="left")
print(df_merged[
    ["Name", "Rating", "Rating_group", "Description"]
].head(10))
print("=" * 50)

# todo seaborn
# Есть ли связь между количеством активных игроков и количеством посещений игры?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Visits", y="Active", alpha=0.6)
plt.title("Связь между Visits и Active")
plt.xlabel("Количество посещений")
plt.ylabel("Активные игроки")
plt.show()

# todo Гистограмма относительных частот
# Как распределено количество активных игроков среди игр?
plt.figure(figsize=(10, 6))
sns.histplot(data=df,x="Active", bins=30, stat="density")
plt.title("Распределение количества активных игроков")
plt.xlabel("Active")
plt.ylabel("Плотность")
plt.show()

# todo kde
# Как выглядит сглаженное распределение рейтингов игр?
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Rating", fill=True)
plt.title("KDE распределения рейтинга игр")
plt.xlabel("Rating")
plt.show()

# todo boxplot
# Есть ли выбросы в количестве активных игроков?
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x=df["Active"], ax=axes[0])
axes[0].set_title("Boxplot Active")
axes[0].set_xlabel("Active")
sns.boxplot(x=np.log1p(df["Active"]), ax=axes[1])
axes[1].set_title("Boxplot log(Active + 1)")
axes[1].set_xlabel("log(Active + 1)")
plt.tight_layout()
plt.show()

# todo Violinplot
# Как различается распределение количества активных игроков между группами рейтинга?
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x="Rating_group", y=np.log1p(df["Active"]))
plt.title("Распределение Active по группам рейтинга")
plt.xlabel("Группа рейтинга")
plt.ylabel("log(Active + 1)")
plt.show()

# todo Scatterplot
# Связано ли количество посещений с количеством избранных пользователей?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Visits", y="Favourites", hue="Rating_group", alpha=0.7)
plt.title("Visits и Favourites")
plt.xlabel("Visits")
plt.ylabel("Favourites")
plt.show()

# todo Plotly
# Какие игры одновременно имеют много активных игроков, посещений и высокую оценку?
fig = px.scatter(
    df,
    x="Visits",
    y="Active",
    size="Favourites",
    color="Rating",
    hover_name="Name",
    hover_data=[
        "Rank",
        "Likes",
        "Dislikes",
        "Rating"
    ],
    title="Популярность игр Roblox"
)
fig.update_xaxes(type="log")
fig.update_yaxes(type="log")
fig.show()

# todo нормальность
# Можно ли считать распределение рейтинга игр нормальным? Рассматриваем Rating
sns.histplot(df["Rating"], kde=True)
plt.title("Распределение Rating")
plt.show()

stats.probplot(df["Rating"], dist="norm", plot=plt)
plt.title("Q-Q plot для Rating")
plt.show()

stat, p_value = stats.jarque_bera(df["Rating"])
print("Statistic:", stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Отклоняем H0: распределение отличается от нормального.")
else:
    print("Нет оснований отклонять H0 о нормальности.")
print("=" * 50)

# todo One Hot Encoding
# Как представить группы рейтинга в числовом виде с помощью One Hot Encoding?
df_ohe = pd.get_dummies(df, columns=["Rating_group"], prefix="rating", dtype=int)
print(df_ohe.head().to_string())