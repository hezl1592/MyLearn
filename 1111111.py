import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

L = 3.95
a1 = 2.1
a2 = 2.95
b = 0.38
h1 = 0.845
h2 = 1.17
f0 = 0.428

f = np.linspace(0, 1, 100)

er1 = (a1/L)/((1 - b) + (f*h1)/ L)
er2 = []
for i in f:
    if i <= f0:
        er = ((L - a2)/L)/(b - (i*h2)/ L)
        er2.append(er)
    else:
        er = (a2/L)/((1 - b) + (i*h2)/ L)
        er2.append(er)
def z(f1):
	z = 0.1 + 0.85 * (f1-0.2)
	return z
	
def f22(x, ax, hx):
	f22 = x * L * (1 - b)/(ax - x * hx)
	return f22
	


# print(er1)
# print(er2)
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.plot(f, er1, 'r-', label = '空载')
plt.plot(f, er2, 'b:', label = '满载')
plt.scatter(f0, ((L - a2)/L)/(b - (f0*h2)/ L),s = 50, color = 'blue')
plt.plot([f0,f0], [0, ((L - a2)/L)/(b - (f0*h2)/ L)], 'g--')
plt.xlabel('附着系数',fontsize = 12)
plt.ylabel('制动效率',fontsize = 12)
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.legend(loc='best')     #开启图示

plt.figure(2)
plt.plot(np.linspace(0.2, 0.8, 60), z(np.linspace(0.2, 0.8, 60)), 'b-')
plt.plot(f, f22(f, a1, h1), 'b:')
plt.xlabel('附着系数',fontsize = 12)
plt.ylabel('制动效率',fontsize = 12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc='best')     #开启图示

plt.show()