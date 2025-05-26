import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Визначення функції споживання енергії (оптимізована для Numba)

@njit
def energy_consumption(hour_of_day, temperature, day_of_week, humidity):
    """
    Моделює споживання енергії на основі часу, температури, дня тижня та вологості.
    Це більш складна модель. Ви повинні замінити її моделлю, підігнаною під ваші дані.

    Args:
        hour_of_day: Година дня (0-23).
        temperature: Температура в градусах Цельсія.
        day_of_week: День тижня (0-6, де 0 - понеділок).
        humidity: Відносна вологість (0-100).

    Returns:
        Очікуване споживання енергії.
    """
    base_consumption = 120  # Змінене базове споживання

    # Вплив температури
    temperature_impact = 0.7 * (22 - temperature)  # Більше споживання при холоді

    # Вплив часу доби (синусоїдальний з піками вранці та ввечері)
    time_impact = 25 * np.sin(2 * np.pi * hour_of_day / 24)  # Збільшений вплив

    # Вплив дня тижня (менше у вихідні)
    is_weekend = day_of_week > 4
    weekend_factor = 0.6 if is_weekend else 1.0  # Зменшене споживання у вихідні
    day_of_week_impact = 30 * (1 - weekend_factor)

    # Вплив вологості (незначна позитивна кореляція)
    humidity_impact = 0.1 * humidity

    # Випадковий шум для імітації непередбачуваних варіацій
    noise = np.random.normal(0, 5)  # Середнє 0, стандартне відхилення 5

    total_consumption = base_consumption + temperature_impact + time_impact + day_of_week_impact + humidity_impact + noise

    # Забезпечуємо неід'ємне споживання
    return max(0, total_consumption)


# 2. Функція інтеграла Монте-Карло (з Numba)

@njit
def monte_carlo_integral(num_iterations, min_hour, max_hour, min_temp, max_temp, min_day, max_day, min_humidity, max_humidity):
    """
    Обчислює інтеграл споживання енергії методом Монте-Карло.

    Args:
        num_iterations: Кількість ітерацій Монте-Карло.
        min_hour: Мінімальна година дня.
        max_hour: Максимальна година дня.
        min_temp: Мінімальна температура.
        max_temp: Максимальна температура.
        min_day: Мінімальний день тижня.
        max_day: Максимальний день тижня.
        min_humidity: Мінімальна вологість.
        max_humidity: Максимальна вологість.

    Returns:
        Кортеж: (Оцінка інтеграла споживання енергії, масив окремих зразків споживання)
    """
    sum_consumption = 0.0
    consumption_samples = np.zeros(num_iterations) # Зберігаємо окремі зразки

    for i in range(num_iterations):
        hour = min_hour + np.random.random() * (max_hour - min_hour)
        temp = min_temp + np.random.random() * (max_temp - min_temp)
        day = min_day + np.random.random() * (max_day - min_day)
        humidity = min_humidity + np.random.random() * (max_humidity - min_humidity)

        instant_consumption = energy_consumption(hour, temp, day, humidity)
        sum_consumption += instant_consumption
        consumption_samples[i] = instant_consumption

    volume = (max_hour - min_hour) * (max_temp - min_temp) * (max_day - min_day) * (max_humidity - min_humidity)
    average_consumption = sum_consumption / num_iterations
    integral = average_consumption * volume

    return integral, consumption_samples


# Додаткова функція для прогнозування пікових навантажень
@njit
def predict_peak_loads(num_samples=10000):
    """
    Прогнозує періоди пікового споживання енергії за допомогою симуляції Монте-Карло.
    Повертає часи пікового споживання та значення.
    """
    peak_hours = []
    peak_consumptions = []
    
    for _ in range(num_samples):
        hour = np.random.uniform(0, 24)
        temp = np.random.uniform(-15, 35)
        day = np.random.uniform(0, 7)
        humidity = np.random.uniform(20, 80)
        
        consumption = energy_consumption(hour, temp, day, humidity)
        
        # Вважаємо споживання > 160 піковим
        if consumption > 160:
            peak_hours.append(hour)
            peak_consumptions.append(consumption)
    
    return np.array(peak_hours), np.array(peak_consumptions)

# 3. Основна програма

if __name__ == '__main__':
    # Встановлюємо зерно випадковості для відтворюваності
    np.random.seed(42)
    
    # Параметри симуляції
    min_hour = 0
    max_hour = 24
    min_temp = -15
    max_temp = 35
    min_day = 0
    max_day = 7
    min_humidity = 20
    max_humidity = 80
    num_iterations_list = [1000, 10000, 100000, 1000000]  # Різна кількість ітерацій

    print("Обчислення енергетичних інтегралів методом Монте-Карло")
    print("=" * 60)
    
    # Обчислюємо "справжнє" значення інтегралу (для порівняння)
    print("Обчислення еталонного значення інтегралу...")
    true_integral, _ = monte_carlo_integral(5000000, min_hour, max_hour, min_temp, max_temp, min_day, max_day, min_humidity, max_humidity)
    print(f'"Еталонне" значення інтегралу: {true_integral:.2f}')
    print()

    errors = []
    execution_times = []
    confidence_intervals = []  # Зберігаємо довірчі інтервали

    for num_iterations in num_iterations_list:
        print(f"Обробка {num_iterations:,} ітерацій...")
        start_time = time.time()
        integral_estimate, consumption_samples = monte_carlo_integral(num_iterations, min_hour, max_hour, min_temp, max_temp, min_day, max_day, min_humidity, max_humidity)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        error = abs(integral_estimate - true_integral) / true_integral * 100  # Відсоткова похибка
        errors.append(error)

        # Обчислюємо довірчий інтервал
        mean_consumption = np.mean(consumption_samples)
        std_dev = np.std(consumption_samples)
        standard_error = std_dev / np.sqrt(num_iterations)
        confidence_level = 0.95
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * standard_error
        volume = (max_hour - min_hour) * (max_temp - min_temp) * (max_day - min_day) * (max_humidity - min_humidity)
        confidence_interval = (integral_estimate - margin_of_error * volume,
                               integral_estimate + margin_of_error * volume)

        confidence_intervals.append(confidence_interval)

        print(f"  Ітерації: {num_iterations:,}")
        print(f"  Оцінка інтегралу: {integral_estimate:.2f}")
        print(f"  Похибка: {error:.3f}%")
        print(f"  Час виконання: {execution_time:.4f} с")
        print(f"  Довірчий інтервал (95%): [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
        print(f"  Ширина довірчого інтервалу: {confidence_interval[1] - confidence_interval[0]:.2f}")
        print()

    # 4. Візуалізація результатів
    print("\nСтворення графіків...")
    plt.figure(figsize=(15, 6))
    
    # Стилізація графіків
    plt.style.use('default')

    plt.subplot(1, 3, 1)  # Створюємо сітку 1x3, це перший підграфік
    plt.plot(num_iterations_list, errors, marker='o', linewidth=2, markersize=8, color='red')
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Похибка (%)")
    plt.title("Похибка vs Кількість ітерацій")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)  # Це другий підграфік
    plt.plot(num_iterations_list, execution_times, marker='s', linewidth=2, markersize=8, color='blue')
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Час виконання (секунди)")
    plt.title("Час виконання vs Кількість ітерацій")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3) # Третій підграфік для ширини довірчого інтервалу
    interval_widths = [ci[1] - ci[0] for ci in confidence_intervals]
    plt.plot(num_iterations_list, interval_widths, marker='^', linewidth=2, markersize=8, color='green')
    plt.xlabel("Кількість ітерацій")
    plt.ylabel("Ширина довірчого інтервалу")
    plt.title("Ширина довірчого інтервалу vs Кількість ітерацій")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Підсумковий аналіз
    print("\n" + "="*60)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ:")
    print("="*60)
    print(f"Найкраща точність досягнута при {num_iterations_list[np.argmin(errors)]:,} ітераціях: {min(errors):.3f}%")
    print(f"Найшвидше виконання: {min(execution_times):.4f} с при {num_iterations_list[np.argmin(execution_times)]:,} ітераціях")
    print(f"Найвужчий довірчий інтервал: {min(interval_widths):.2f} при {num_iterations_list[np.argmin(interval_widths)]:,} ітераціях")
    
    # Додатковий аналіз пікових навантажень
    print("\n" + "="*60)
    print("ПРОГНОЗУВАННЯ ПІКОВИХ НАВАНТАЖЕНЬ:")
    print("="*60)
    
    peak_hours, peak_consumptions = predict_peak_loads(50000)
    
    if len(peak_hours) > 0:
        print(f"Виявлено {len(peak_hours)} випадків пікового навантаження (>160 кВт·год)")
        print(f"Середній час піку: {np.mean(peak_hours):.1f} годин")
        print(f"Максимальне споживання: {np.max(peak_consumptions):.1f} кВт·год")
        print(f"Середнє пікове споживання: {np.mean(peak_consumptions):.1f} кВт·год")
        
        # Створюємо візуалізацію пікових навантажень
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(peak_hours, bins=24, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel("Година доби")
        plt.ylabel("Кількість пікових подій")
        plt.title("Розподіл пікових навантажень по годинах")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(peak_consumptions, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel("Споживання (кВт·год)")
        plt.ylabel("Частота")
        plt.title("Розподіл величини пікових навантажень")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Пікових навантажень не виявлено")

    print("\nВисновок: Зі збільшенням кількості ітерацій точність зростає, але час виконання також збільшується.")