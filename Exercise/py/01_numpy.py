import numpy as np


def show_squeeze():
    print('--------------------')
    x = np.array([[[1, 2, 3]], [[1, 2, 3]]])
    print('x:')
    print(x)
    print('x.shape:', x.shape)
    print('------removing dimensions------')
    x1 = np.squeeze(x, axis=1)
    print(x1)
    print('x1.shape:', x1.shape)
    return None


def show_expand_dims():
    print('--------------------')
    x = np.array([[1, 2, 3], [1, 2, 3]])
    print('x:')
    print(x)
    print('x.shape:', x.shape)
    print('------expanding dimensions------')
    x1 = np.expand_dims(x, axis=1)
    print(x1)
    print('x1.shape:', x1.shape)
    return None


def show_tile():
    print('--------------------')
    x = np.array([[1, 2], [1, 2]])
    y = np.array([[5, 6]])
    print('x:', x)
    print('y:', y)
    print('x.shape:', x.shape)
    print("y.shape:", y.shape)
    print('')
    print('------tile------')
    x1 = np.tile(y, (len(x), 1))
    print(x1)
    print('x1.shape:', x1.shape)
    return None


if __name__ == "__main__":
    show_squeeze()
    show_expand_dims()
    show_tile()
