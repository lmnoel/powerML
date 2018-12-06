import numpy as np
def load_0(self):
    A = np.array([[1] * 2000] * 2000)
    B = np.array([[2] * 2000] * 2000)
    C = A.dot(B)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == 'load_0':
            load_0()
        elif arg == 'load_1':
            load_1()
        elif arg == 'load_2':
            load_2()
        elif arg == 'load_3':
            load_3()
        elif arg == 'load_4':
            load_4()
        elif arg == 'load_5':
            load_5()
        elif arg == 'load_6':
            load_6()
        elif arg == 'load_7':
            load_7()
        else:
            raise Exception("Argument not recognized")