# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2018/12/16

'''
---------
the test of *args and **kwargs。
---------
'''
print(__doc__)


def test_var_args(f_arg, *argv):
    print('test of *args')
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)


def test_kwargs(**kwargs):
    print('\ntest of **kwargs')
    for key, value in kwargs.items():
        print('{0} == {1}'.format(key, value))


test_var_args('12', 'sss', 'sss')

ccc = {'the': 3, 'are': 2}
test_kwargs(**ccc)
