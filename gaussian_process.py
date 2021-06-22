"""
多元高斯分布采样
"""
from matplotlib import pyplot as plt
import numpy as np

# SE协方差函数
kernel_se = np.vectorize(lambda x1, x2, l: np.exp(-(x1 - x2) ** 2 / (2 * l ** 2)))   # np.vectorize()将函数向量化

def sample_se(x, l, mean=0):
    # x为numpy数组，e.g. x = np.arange(-5, 5, 0.05)
    x1, x2 = np.meshgrid(x, x)
    n = len(x)
    sigma = kernel_se(x1, x2, l) + np.identity(n) * 0.000000001
    L = np.linalg.cholesky(sigma)       # 为什么要进行分解
    u = np.random.randn(n)
    y = mean + L @ u      # 这里@是什么意思
    return y

c = ['red', 'green', 'blue']
l = [3, 1, 0.3]

for i in range(len(l)):
    x = np.arange(-5, 5, 0.05)
    y = sample_se(x, l[i])
    plt.plot(x, y, c=c[i], linewidth=1, label='l=%.1f' % l[i])

plt.xlabel('input, x')
plt.ylabel('output, f(x)')
plt.legend(loc='best')
plt.show()



from matplotlib import pyplot as plt
import numpy as np

# 目标函数                                                                                                                                                              
objective = np.vectorize(lambda x, std_n=0: 0.001775 * x**5 - 0.055 * x**4 + \
    0.582 * x**3 - 2.405 * x**2 + 3.152 * x + 4.678 + np.random.normal(0, std_n))

# 超参数                                                                                                                                                                
mean, l, std_f, std_n = 5, 1, 1, 0.0001

# SE协方差函数                                                                                                                                                          
kernel = lambda r_2, l: np.exp(-r_2 / (2 * l**2))

# 训练集，以一维输入为例
X = np.arange(1.5, 10, 1.5)
X = X.reshape(X.size, 1)
Y = objective(X).flatten()

# 未知样本
Xs = np.arange(0, 10, 0.1)
Xs = Xs.reshape(Xs.size, 1)

n, d = X.shape
t = np.repeat(X.reshape(n, 1, d), n, axis=1) - X  # 因为采样点为一维度的数据，所以X=\mu(X)
r_2 = np.sum(t**2, axis=2)
Kf = std_f**2 * kernel(r_2, l)
Ky = Kf + std_n**2 * np.identity(n)
Ky_inv = np.linalg.inv(Ky)

m = Xs.shape[0]
t = np.repeat(Xs.reshape(m, 1, d), n, axis=1) - X   
r_2 = np.sum(t**2, axis=2).T
kf = std_f**2 * kernel(r_2, l)
mu = mean + kf.T @ Ky_inv @ (Y - mean)
std = np.sqrt(std_f**2 - np.sum(kf.T @ Ky_inv * kf.T, axis=1))

x_test = Xs.flatten()
y_obj = objective(x_test).flatten()

plt.plot(x_test, mu, c='black', lw=1, label='predicted mean')
plt.fill_between(x_test, mu + std, mu - std, alpha=0.2, color='#9FAEB2', lw=0)
plt.plot(x_test, y_obj, c='red', ls='--', lw=1, label='objective function')
plt.scatter(X.flatten(), Y, c='red', marker='o', s=20)
plt.legend(loc='best')
plt.show()


