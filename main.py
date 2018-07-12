import svm
import preprocess_data

df = preprocess_data.loadDataset('AAPL')
svm.train(df)
