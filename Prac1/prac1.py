#!/usr/bin/env python3

def zip_unlimited_iterators(iterators_list):
    """
    Аналог zip(), що працює зі списком ітераторів(необмеежна кількість).
    Об'єднує елементи з ітераторів у списку 'iterators_list'
    поелементно, доки найкоротший ітератор не закінчиться.
    """
    if not iterators_list:
        return  # Якщо список ітераторів порожній, нічого не робимо

    iterators = [iter(it) for it in iterators_list] # Перетворюємо на ітератори на випадок, якщо отримали не ітератори

    while True:
        elements_tuple = []
        try:
            for iterator in iterators:
                elements_tuple.append(next(iterator))
            yield tuple(elements_tuple) # Повертаємо кортеж зібраних елементів
        except StopIteration:
            break # Зупиняємось, якщо хоча б один з ітераторів закінчився

# Приклад використання 1: Списки чисел
number_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
iterators_for_zip = [iter(number_list) for number_list in number_lists] # Створюємо список ітераторів

print("Приклад 1: Списки чисел")
for tuple_result in zip_unlimited_iterators(iterators_for_zip):
    print(tuple_result)
# Вивід:
# (1, 4, 7)
# (2, 5, 8)
# (3, 6, 9)

print("\n" + "="*30 + "\n")

# Приклад використання 2: Ітератори різних типів
iterators_of_different_types = [
    iter(['a', 'b', 'c', 'd']),
    iter((10, 20, 30)),
    iter("XYZ")
]

print("Приклад 2: Ітератори різних типів")
for tuple_result in zip_unlimited_iterators(iterators_of_different_types):
    print(tuple_result)
# Вивід:
# ('a', 10, 'X')
# ('b', 20, 'Y')
# ('c', 30, 'Z')