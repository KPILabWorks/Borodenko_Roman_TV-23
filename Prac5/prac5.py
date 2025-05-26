\
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import json

# Створення фіктивних даних (якщо файли не існують)
def create_dummy_files():
    # CSV
    csv_data = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'value_csv': [100, 200, 300]}
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv('data.csv', index=False)
    print("data.csv created.")

    # JSON
    json_data = [
        {"id": 1, "city": "New York", "value_json": 10.5},
        {"id": 2, "city": "London", "value_json": 20.3},
        {"id": 4, "city": "Paris", "value_json": 30.1}
    ]
    with open('data.json', 'w') as f:
        json.dump(json_data, f)
    print("data.json created.")    # SQLite
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        status TEXT,
        value_sql INTEGER
    )
    """)
    users_data = [
        (1, 'active', 50),
        (2, 'inactive', 75),
        (5, 'active', 90)
    ]
    cursor.executemany("INSERT INTO users (id, status, value_sql) VALUES (?,?,?)", users_data)
    conn.commit()
    conn.close()
    print("data.db created with users table.")

# Запустіть цю функцію один раз, щоб створити файли
# create_dummy_files() # Розкоментуйте, якщо файли ще не створені

# 1. Завантаження даних з CSV
try:
    df_csv = pd.read_csv('data.csv')
    print("CSV Data:")
    print(df_csv)
except FileNotFoundError:
    print("Помилка: data.csv не знайдено. Будь ласка, створіть його.")
    df_csv = pd.DataFrame() # Створюємо порожній DataFrame, щоб уникнути помилок далі

# 2. Завантаження даних з JSON
try:
    df_json = pd.read_json('data.json')
    print("\\nJSON Data:")
    print(df_json)
except FileNotFoundError:
    print("Помилка: data.json не знайдено. Будь ласка, створіть його.")
    df_json = pd.DataFrame()

# 3. Завантаження даних з SQL (SQLite)
try:
    engine = create_engine('sqlite:///data.db')
    df_sql = pd.read_sql('SELECT * FROM users', engine)
    print("\\nSQL Data:")
    print(df_sql)
except Exception as e:
    print(f"Помилка завантаження даних з SQL: {e}")
    df_sql = pd.DataFrame()


# Очищення даних (приклад: видалення дублікатів по 'id' перед злиттям, обробка пропусків)
# Це дуже залежить від ваших конкретних даних та вимог
if not df_csv.empty:
    df_csv = df_csv.drop_duplicates(subset=['id'])
if not df_json.empty:
    df_json = df_json.drop_duplicates(subset=['id'])
if not df_sql.empty:
    df_sql = df_sql.drop_duplicates(subset=['id'])

# Заповнення пропущених значень (приклад)
# df_csv.fillna(0, inplace=True) # Заповнити NaN нулями
# df_json.fillna('', inplace=True) # Заповнити NaN порожніми рядками

# 4. Злиття датафреймів
# Почнемо з df_csv і будемо приєднувати інші
merged_df = df_csv

if not df_json.empty:
    # Використовуємо outer join, щоб зберегти всі id з обох таблиць
    # suffixes додає постфікси до назв колонок, що збігаються (крім 'id')
    merged_df = pd.merge(merged_df, df_json, on='id', how='outer', suffixes=('_csv', '_json'))
else:
    print("\\nПропуск злиття з JSON, оскільки df_json порожній.")


if not df_sql.empty:
    merged_df = pd.merge(merged_df, df_sql, on='id', how='outer', suffixes=('', '_sql'))
    # Якщо є колонка 'id_sql' після злиття, і вона дублює 'id', її можна видалити
    if 'id_sql' in merged_df.columns and 'id' in merged_df.columns:
        # Перевіряємо, чи 'id' та 'id_sql' не є однією і тією ж колонкою
        if not merged_df['id'].equals(merged_df['id_sql']):
             # Якщо 'id_sql' містить значення, яких немає в 'id', об'єднаємо їх
            merged_df['id'] = merged_df['id'].combine_first(merged_df['id_sql'])
        merged_df.drop(columns=['id_sql'], inplace=True, errors='ignore')

else:
    print("\\nПропуск злиття з SQL, оскільки df_sql порожній.")


print("\\nMerged DataFrame:")
print(merged_df)

