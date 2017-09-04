from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def f1():
    X_train = np.array([[2, 2], [2, 0], [0, 2], [0, 0]])
    Y_train = np.array([1, 1, 1, 0])

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    # print(lr.predict(X_train))
    # print(lr.score(X_train, Y_train))
    # print(lr.get_params())
    result = lr.predict_proba(X_train)
    print(list(result[:, 1]))


def f2():
    data = [('i1', 0.1), ('i2', 0.2)]
    df = pd.DataFrame(data, columns=['iid', 'val'])
    for i in range(len(df)):
        df.loc[i, 'new'] = 1
    print(df)
    print(df.values)


def f3():
    df1 = pd.DataFrame([['i1', 1]], columns=['iid', 'val'])
    if 1 in df1['val'].values:
        print(True)

if __name__ == '__main__':
    f3()