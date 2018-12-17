# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2018/12/16

'''
empCount 变量是一个类变量，它的值将在这个类的所有实例之间共享。你可以在内部类或外部类使用 Employee.empCount 访问。

第一种方法__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法

self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
'''
print(__doc__)


class emplyee:
    '所有员工的基类'
    empcount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        emplyee.empcount += 1

    def displaycount(self):
        print("Total employee: %d" % emplyee.empcount)

    def displayemployee(self):
        print('name:', self.name, ', salary:', self.salary)


emp1 = emplyee('zilch', 66)
emp2 = emplyee('liang', 90)

print(emplyee.empcount)
print(emp1.salary)
