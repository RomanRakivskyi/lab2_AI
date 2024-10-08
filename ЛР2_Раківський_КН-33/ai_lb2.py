import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Генерація випадкового набору даних
np.random.seed(0)
X = np.random.rand(1000, 1) * 1000  # Генерація 1000 випадкових значень в діапазоні [0, 1000]
y = 2 * X.squeeze() + np.random.normal(0, 100, 1000)  # Лінійна функція з шумом

# 2. Нормалізація значень
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Навчання KNN-регресора з різними значеннями K
k_values = range(1, 21)
mse_values = []  # Список для збереження значень середньоквадратичної помилки
r2_values = []   # Список для збереження значень R^2

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mse_values.append(mse)
    r2_values.append(r2)

# 5. Вибір k для найкращих показників якості регресії
optimal_k = k_values[np.argmin(mse_values)]
print(f"Оптимальне значення K: {optimal_k}")

# 6. Візуалізація результатів
plt.figure(figsize=(12, 10))

# Графік MSE в залежності від K
plt.subplot(3, 2, 1)
plt.plot(k_values, mse_values, marker='o')
plt.title('MSE в залежності від K')
plt.xlabel('K')
plt.ylabel('Середньоквадратична помилка (MSE)')
plt.xticks(k_values)
plt.grid()

# Графік R^2 в залежності від K
plt.subplot(3, 2, 2)
plt.plot(k_values, r2_values, marker='o', color='orange')
plt.title('R^2 в залежності від K')
plt.xlabel('K')
plt.ylabel('Коефіцієнт детермінації (R^2)')
plt.xticks(k_values)
plt.grid()

# Графік фактичних і прогнозованих значень
knn_optimal = KNeighborsRegressor(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_optimal = knn_optimal.predict(X_test)

plt.subplot(3, 2, 3)
plt.scatter(range(len(y_test)), y_test, color='red', alpha=0.5, label='Фактичні значення')  # Червоні крапочки
plt.scatter(range(len(y_pred_optimal)), y_pred_optimal, color='blue', alpha=0.5, label='Прогнозовані значення')  # Сині крапочки
plt.plot([0, len(y_test)], [y.min(), y.max()], 'r--', lw=2)  # Лінія y=x
plt.title(f'Фактичні vs Прогнозовані значення (K={optimal_k})')
plt.xlabel('Індекс')
plt.ylabel('Значення')
plt.legend()
plt.grid()

# Графік помилок
errors = y_test - y_pred_optimal
plt.subplot(3, 2, 4)
plt.scatter(y_pred_optimal, errors, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')  # Лінія на нулі
plt.title('Помилки регресії')
plt.xlabel('Прогнозовані значення')
plt.ylabel('Помилка (Фактичні - Прогнозовані)')
plt.grid()

# Графік з крапочками
plt.subplot(3, 2, 5)
plt.scatter(y_test, y_pred_optimal, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Лінія y=x
plt.title('Фактичні vs Прогнозовані значення (KNN)')
plt.xlabel('Фактичні значення')
plt.ylabel('Прогнозовані значення')
plt.grid()

plt.tight_layout()
plt.show()

