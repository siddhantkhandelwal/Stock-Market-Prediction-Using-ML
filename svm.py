from sklearn import svm
import preprocess_data

def train(df):
    '''This function trains the data on 4 different SVC model kernels:
    1. Linear Kernel
    2. Polynomial Kernel
    3. Radial Basis Function Kernel
    4. Sigmoid Kernel
    The hyperparameters are set default in each case.
    The score of the model on the Dev/Test set is returned to the main script.
    '''
    X, y = preprocess_data.addFeatures(df)
    X_train, X_test, y_train, y_test = preprocess_data.splitDataset(X, y)
    X_train, X_test = preprocess_data.featureScaling(X_train, X_test)
    
    model_linear = svm.SVC(kernel='linear')
    model_linear.fit(X_train, y_train)
    score_linear = model_linear.score(X_test, y_test)

    model_poly = svm.SVC(kernel='poly')
    model_poly.fit(X_train, y_train)
    score_poly = model_poly.score(X_test, y_test)
    
    model_rbf = svm.SVC(kernel='rbf')
    model_rbf.fit(X_train, y_train)
    score_rbf = model_rbf.score(X_test, y_test)

    model_sig = svm.SVC(kernel='sigmoid')
    model_sig.fit(X_train, y_train)
    score_sig = model_sig.score(X_test, y_test)
    
    return score_linear, score_poly, score_rbf, score_sig
