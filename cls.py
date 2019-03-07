from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from csv_reader import extract_feature_and_labels
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


def prepare_data(): # make it a binary classfication
    threshold = 0
    file_path = "gwp_brt_2000_2008.csv"
    features, labels = extract_feature_and_labels(file_path, threshold)
    return features, labels

# TODO: grd search for the best params of NN
# TODO: evaluation of SVM, RF and so
def evaluate(features, labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, warm_start=True)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
    predicted = cross_val_predict(clf, features, labels, cv=10)
    return metrics.accuracy_score(labels, predicted), metrics.recall_score(labels, predicted), metrics.precision_score(labels, predicted)


features, labels = prepare_data()
print(evaluate(features, labels))