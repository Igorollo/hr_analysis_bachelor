import numpy as np
import matplotlib.pyplot as plt

# Przykładowe dane
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1,1,1000) # Transformata Fishera to funkcja arctanh

# Wykonanie transformaty Fishera
fisher_transform = np.arctanh(y)

# Wizualizacja danych przed i po transformacji
plt.figure(figsize=(14, 7))

# Wykres oryginalnych danych
plt.subplot(1, 2, 1)
plt.plot(x, y, label='Przed transformacją Fishera')
plt.legend()

# Wykres po transformacji Fishera
plt.subplot(1, 2, 2)
plt.plot(x, fisher_transform, label='Po transformacji Fishera', color='red')
plt.legend()

plt.show()
