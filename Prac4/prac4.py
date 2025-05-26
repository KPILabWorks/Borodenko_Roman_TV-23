import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Генерація синтетичних даних
np.random.seed(0)
# Температура (незалежна змінна)
X_temperature = np.sort(np.random.rand(100, 1) * 30, axis=0)  # Температури від 0 до 30 градусів

# Енергоспоживання (залежна змінна)
# Припустимо, що енергоспоживання зменшується з підвищенням температури (менше опалення)
# і збільшується при дуже високих температурах (кондиціонування)
# Для простоти, візьмемо лінійну залежність з шумом: y = a*X + b + noise
true_a = -2.5
true_b = 100
y_energy = true_a * X_temperature.flatten() + true_b + np.random.normal(0, 10, X_temperature.shape[0])
y_energy = y_energy.reshape(-1, 1)

# 2. Класична лінійна регресія
linear_reg = LinearRegression()
linear_reg.fit(X_temperature, y_energy.ravel())
y_linear_pred = linear_reg.predict(X_temperature)
linear_mse = mean_squared_error(y_energy, y_linear_pred)

# 3. Байєсівська регресія
bayesian_reg = BayesianRidge(compute_score=True)
bayesian_reg.fit(X_temperature, y_energy.ravel())
y_bayesian_pred, y_bayesian_std = bayesian_reg.predict(X_temperature, return_std=True)
bayesian_mse = mean_squared_error(y_energy, y_bayesian_pred)

# 4. Порівняння результатів
print("Класична лінійна регресія:")
print(f"  Коефіцієнт (a): {linear_reg.coef_[0]:.2f}")
print(f"  Перетин (b): {linear_reg.intercept_:.2f}")
print(f"  Середньоквадратична помилка (MSE): {linear_mse:.2f}\n")

print("Байєсівська регресія:")
print(f"  Коефіцієнт (a): {bayesian_reg.coef_[0]:.2f}")
print(f"  Перетин (b): {bayesian_reg.intercept_:.2f}")
# Байєсівська регресія також надає оцінку дисперсії для коефіцієнтів
print(f"  Альфа (точність шуму): {bayesian_reg.alpha_:.2f}")
print(f"  Лямбда (точність ваг): {bayesian_reg.lambda_:.2f}")
print(f"  Середньоквадратична помилка (MSE): {bayesian_mse:.2f}\n")

print("Порівняння:")
print(f"Різниця в MSE (Лінійна - Байєсівська): {linear_mse - bayesian_mse:.2f}")
print("Байєсівська регресія надає не тільки точкові оцінки коефіцієнтів,")
print("але й розподіли для них, що дозволяє оцінити невизначеність.")
print("У даному випадку, з достатньою кількістю даних, результати можуть бути схожими.")

# Візуалізація
plt.figure(figsize=(12, 7))
plt.scatter(X_temperature, y_energy, color='black', label='Спостережувані дані', s=20)
plt.plot(X_temperature, y_linear_pred, color='blue', linestyle='--', linewidth=2, label=f'Лінійна регресія (MSE: {linear_mse:.2f})')
plt.plot(X_temperature, y_bayesian_pred, color='red', linewidth=2, label=f'Байєсівська регресія (MSE: {bayesian_mse:.2f})')
plt.fill_between(X_temperature.flatten(), y_bayesian_pred - y_bayesian_std, y_bayesian_pred + y_bayesian_std,
                 color='pink', alpha=0.5, label='Невизначеність байєсівської регресії (1 std)')

plt.xlabel('Температура (°C)')
plt.ylabel('Енергоспоживання (кВт·год)')
plt.title('Порівняння лінійної та байєсівської регресії')
plt.legend()
plt.grid(True)
plt.show()

print("\nПояснення висновків:")
print("1. Коефіцієнти: Обидві моделі повинні дати схожі оцінки коефіцієнтів (нахилу та перетину),")
print("   оскільки байєсівська регресія з неінформативними апріорними розподілами часто збігається з МНК.")
print("2. MSE: Середньоквадратичні помилки також, ймовірно, будуть близькими.")
print("   Невеликі відмінності можуть виникати через різний підхід до оцінки параметрів.")
print("3. Невизначеність: Ключова перевага байєсівської регресії - це кількісна оцінка невизначеності.")
print("   Графік показує довірчий інтервал (або інтервал правдоподібності) для прогнозів байєсівської моделі.")
print("   Це відображає нашу впевненість у прогнозах: там, де даних менше або вони більш розкидані, інтервал буде ширшим.")
print("4. Параметри моделі: Байєсівська регресія оцінює параметри alpha (точність шуму в даних) та lambda (точність ваг моделі).")
print("   Ці параметри допомагають регуляризувати модель та уникнути перенавчання, особливо на малих наборах даних.")
print("\nЗагалом, для простих лінійних залежностей з достатньою кількістю даних, обидві моделі можуть дати схожі точкові прогнози.")
print("Однак байєсівський підхід надає багатшу інформацію про невизначеність моделі та її параметрів,")
print("що є критично важливим у багатьох практичних застосуваннях, де оцінка ризиків та надійності прогнозів є пріоритетом.")

