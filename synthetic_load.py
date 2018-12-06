from __future__ import print_function
import numpy as np
import time

def load_0():
    A = np.array([[1] * 8000] * 200)
    B = np.array([[2] * 200] * 8000)
    C = A.dot(B)

def load_1():
    A = np.array([[1] * 8000] * 400)
    B = np.array([[2] * 400] * 8000)
    C = A.dot(B)

def load_2():
    A = np.array([[1] * 8000] * 500)
    B = np.array([[2] * 500] * 8000)
    C = A.dot(B)

def load_3():
    A = np.array([[1] * 8000] * 600)
    B = np.array([[2] * 600] * 8000)
    C = A.dot(B)

def load_4():
    A = np.array([[1] * 8000] * 700)
    B = np.array([[2] * 700] * 8000)
    C = A.dot(B)

def load_5():
    A = np.array([[1] * 1200] * 1200)
    B = np.array([[2] * 1200] * 1200)
    C = A.dot(B)

def load_6():
    A = np.array([[1] * 1700] * 1700)
    B = np.array([[2] * 1700] * 1700)
    C = A.dot(B)

def load_7():
    A = np.array([[1] * 2000] * 2000)
    B = np.array([[2] * 2000] * 2000)
    C = A.dot(B)

def test(load_type):
    start = time.time()
    if load_type == 'load_0':
        load_0()
    elif load_type == 'load_1':
        load_1()
    elif load_type == 'load_2':
        load_2()
    elif load_type == 'load_3':
        load_3()
    elif load_type == 'load_4':
        load_4()
    elif load_type == 'load_5':
        load_5()
    elif load_type == 'load_6':
        load_6()
    elif load_type == 'load_7':
        load_7()
    else:
        raise Exception("load_typeument not recognized")
    print(time.time() - start)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        test(sys.argv[1])
        