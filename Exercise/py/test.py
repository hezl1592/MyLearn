import numpy as np
import matplotlib.pyplot as plt

plt.figure()
mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, (1, 100))
s1 = sigma * np.random.randn(1, 100) + mu
x = np.linspace(1, 100, 100).reshape(-1, 100)
plt.plot(x, s, 'r*')
plt.show()