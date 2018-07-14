from sklearn import svm
import preprocess_data

def train(df):
    X, y = preprocess_data.addFeatures(df)
    X_train, X_test, y_train, y_test = preprocess_data.splitDataset(X, y)
    X_train, X_test = preprocess_data.featureScaling(X_train, X_test)
    model = svm.SVR(kernel='rbf', gamma='auto')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return(score)