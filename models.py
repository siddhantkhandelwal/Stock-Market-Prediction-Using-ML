from sklearn import svm
import preprocess_data
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import VotingClassifier


def train(df):
    '''This function trains the data on 4 different SVC model kernels:
    1. Linear Kernel
    2. Polynomial Kernel
    3. Radial Basis Function Kernel
    4. Sigmoid Kernel
    The RFC model is also implemented.
    
    The hyperparameters are set default in each case.
    The score of the model on the Dev/Test set is returned to the main script.
    '''
    X, y = preprocess_data.addFeatures(df)
    X_train, X_test, y_train, y_test = preprocess_data.splitDataset(X, y)
    X_train, X_test = preprocess_data.featureScaling(X_train, X_test)
    
    model_slinear = svm.SVC(kernel='linear')
    model_slinear.fit(X_train, y_train)
    score_slinear = model_slinear.score(X_test, y_test)

    model_spoly = svm.SVC(kernel='poly')
    model_spoly.fit(X_train, y_train)
    score_spoly = model_spoly.score(X_test, y_test)
    
    model_srbf = svm.SVC(kernel='rbf')
    model_srbf.fit(X_train, y_train)
    score_srbf = model_srbf.score(X_test, y_test)

    model_ssig = svm.SVC(kernel='sigmoid')
    model_ssig.fit(X_train, y_train)
    score_ssig = model_ssig.score(X_test, y_test)

    model_rfc = rfc(max_depth=4, random_state=0)
    model_rfc.fit(X_train, y_train)
    score_rfc = model_rfc.score(X_test, y_test)

    model_abc = abc(n_estimators=500)
    model_abc.fit(X_train, y_train)
    score_abc = model_abc.score(X_test, y_test) 

    model_vc = VotingClassifier(estimators=[('svc', model_srbf,), ('rf', model_rfc)], voting='hard')
    model_vc.fit(X_train, y_train)
    score_vc = model_vc.score(X_test, y_test)

    return score_slinear, score_spoly, score_srbf, score_ssig, score_rfc, score_abc,  score_vc
