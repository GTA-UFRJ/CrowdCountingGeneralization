from test_script import test
import sys
import numpy as np
import pickle

if __name__ == '__main__':
    model_train = sys.argv[1]
    model_test = sys.argv[2]
    n_splits = sys.argv[3]
    result = []
    for n in range(int(n_splits)):
        result.append(test(['test',model_train,model_test,str(n)]))
    print(result)
    result = np.array(result)
    print(result.mean(),' +- ', result.std())
    with open(f'./results/{model_train}/{model_train}_{model_test}.pkl','wb') as fp:
        pickle.dump(result, fp)

