from train_script import train
import sys

if __name__ == '__main__':
    n_splits = sys.argv[2]
    model = sys.argv[1]
    for n in range(int(n_splits)):
        train([sys.argv[0], model,n])


