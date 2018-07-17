import svm
import preprocess_data
import sys


symbols = sys.argv[1:] #loads symbols from the command line.
for symbol in symbols: #runs the svm models on every symbol.
    df = preprocess_data.loadDataset(symbol)
    svm.train(df)
    scores_svm =  svm.train(df)
    print(f"Scores of SVC model on the dataset {symbol} (Linear, Poly, RBF, Sigmoid): {[score * 100 for score in scores_svm]}")

