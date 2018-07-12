from sklearn import svm
import preprocess_data

def train(df):
	X, y = preprocess_data.addFeatures(df)
	X_train, X_cv, X_test = preprocess_data.featureScaling(df)
	y_train, y_cv, y_test = preprocess_data.splitDataset(y)
	model = svm.SVR(kernel='linear', C=1)
	model.fit(X_train, y_train)
	score = model.score(X_cv, y_cv)
	print('Confidence score for linear kernel :',score*100)
