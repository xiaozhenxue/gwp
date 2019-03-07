import csv


def remove_missing_value(file_path):
    with open(file_path, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        labels = list([])
        features = list([])
        for row in reader:
            if '' not in row and '#REF!' not in row:
                labels.append(row[1])
                features.append(row[2:])
        return features, labels


def remove_head(features, labels):
    return features[1:], labels[1:]


def convert_float(features, labels):
    new_labels = [float(e) for e in labels]
    new_features = list([])
    for row in features:
        new_features.append([float(e) for e in row])
    return new_features, new_labels


def categrize_feature(labels, threshold):
    new_labels = list([])
    for l in labels:
        new_labels.append(0 if l >= threshold else 1)
    return new_labels


def extract_feature_and_labels(file_path, threshold):
    features, labels = remove_missing_value(file_path)
    features, labels = remove_head(features, labels)
    features, labels = convert_float(features, labels)
    labels = categrize_feature(labels, threshold)
    return features, labels
