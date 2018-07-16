from sklearn import svm
import preprocess_data

def train(df):
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
