import numpy as np

if __name__ == "__main__":
    testfunc = np.poly1d([1, 2, 3])
    print(testfunc)
    print(testfunc(2))
