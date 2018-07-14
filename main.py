import svm
import preprocess_data
import sys

symbols = sys.argv[1:]
for symbol in symbols:
    df = preprocess_data.loadDataset(symbol)
    scores_svm =  svm.train(df)
    print(f"Scores of SVR model on the dataset {symbol} (Linear, Poly, RBF, Sigmoid): {[score * 100 for score in scores_svm]}")
