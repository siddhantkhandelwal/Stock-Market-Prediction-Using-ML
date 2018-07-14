import svm
import preprocess_data
import sys

symbols = sys.argv[1:]
for symbol in symbols:
    df = preprocess_data.loadDataset(symbol)
    print(f"Score of RBF SVR model on the dataset {symbol}: {svm.train(df)}")
