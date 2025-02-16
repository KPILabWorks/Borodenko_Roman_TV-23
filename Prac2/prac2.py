import os
import tarfile
import json
import pandas as pd
import timeit

# Завантаження та розпакування даних (як раніше)
if not os.path.exists("amazon-massive-dataset-1.0.tar.gz"):
    os.system("curl https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz --output amazon-massive-dataset-1.0.tar.gz")

if not os.path.exists("1.0"):
    with tarfile.open("amazon-massive-dataset-1.0.tar.gz", "r:gz") as tar:
        tar.extractall()

data_dir = "1.0/data"
all_data = []

for filename in os.listdir(data_dir):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)

df = pd.DataFrame(all_data)

# Створення MultiIndex (як раніше)
df = df.set_index(["locale", "partition", "scenario", "intent", "id"])
df = df.sort_index()

# --- Приклади вибірки та порівняння швидкості ---

# 1. Вибірка на ОДНОМУ рівні: всі дані для locale='de-DE'
def select_with_xs_single_level():
    return df.xs("de-DE", level="locale")

def select_with_loc_single_level():
    idx = pd.IndexSlice
    return df.loc[idx["de-DE", :, :, :, :], :]

def select_with_naive_single_level(): # Don't do this, for demonstration only!
    return df[df.index.get_level_values('locale') == 'de-DE']

# Вимірювання часу
time_xs_single = timeit.timeit(select_with_xs_single_level, number=1000)  
time_loc_single = timeit.timeit(select_with_loc_single_level, number=1000)
time_naive_single = timeit.timeit(select_with_naive_single_level, number=100)

print(f"Час вибірки (один рівень, 'de-DE'):")
print(f"  .xs():               {time_xs_single:.6f} секунд")
print(f"  .loc + IndexSlice: {time_loc_single:.6f} секунд")
print(f"  Наївне індексування: {time_naive_single:.6f} секунд") # Very slow!


# 2. Вибірка за КОМБІНАЦІЄЮ рівнів: de-DE, train, alarm_query
def select_with_xs_multi_level():
    return df.xs(("de-DE", "train", "alarm", "alarm_query"), level=["locale", "partition", "scenario", "intent"])

def select_with_loc_multi_level():
    idx = pd.IndexSlice
    return df.loc[idx["de-DE", "train", "alarm", "alarm_query", :], :]

#  "Наївного" варіанту для MultiIndex НЕМАЄ.

# Вимірювання часу (тільки .xs() і .loc)
time_xs_multi = timeit.timeit(select_with_xs_multi_level, number=100)
time_loc_multi = timeit.timeit(select_with_loc_multi_level, number=100)

print(f"\nЧас вибірки (комбінація рівнів, 'de-DE', 'train', 'alarm', 'alarm_query'):")
print(f"  .xs():               {time_xs_multi:.6f} секунд")
print(f"  .loc + IndexSlice: {time_loc_multi:.6f} секунд")


# 3. Вибірка на ОДНОМУ рівні: всі дані для partition='train'
def select_with_xs_train():
    return df.xs("train", level="partition")

def select_with_loc_train():
    idx = pd.IndexSlice
    return df.loc[idx[:, "train", :, :, :], :]

def select_with_naive_train():  # Don't do this!
    return df[df.index.get_level_values('partition') == 'train']

time_xs_train = timeit.timeit(select_with_xs_train, number=100)
time_loc_train = timeit.timeit(select_with_loc_train, number=100)
time_naive_train = timeit.timeit(select_with_naive_train, number=100)

print(f"\nЧас вибірки (один рівень, 'train'):")
print(f"  .xs():               {time_xs_train:.6f} секунд")
print(f"  .loc + IndexSlice: {time_loc_train:.6f} секунд")
print(f"  Наївне індексування: {time_naive_train:.6f} секунд")  # Very Slow

#4. Вибірка всіх сценаріїв music
def select_with_xs_music():
    return df.xs("music", level="scenario")

def select_with_loc_music():
    idx = pd.IndexSlice
    return df.loc[idx[:, :, "music", :, :], :]

def select_with_naive_music():
    return df[df.index.get_level_values("scenario") == "music"]

time_xs_music = timeit.timeit(select_with_xs_music, number=100)
time_loc_music = timeit.timeit(select_with_loc_music, number=100)
time_naive_music = timeit.timeit(select_with_naive_music, number=100)

print(f"\nЧас вибірки (один рівень, 'music'):")
print(f"  .xs():               {time_xs_music:.6f} секунд")
print(f"  .loc + IndexSlice: {time_loc_music:.6f} секунд")
print(f"  Наївне індексування: {time_naive_music:.6f} секунд")  # Very Slow