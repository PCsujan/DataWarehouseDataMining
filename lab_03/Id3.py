import pandas as pd
import numpy as np
from math import log2
import pprint

# ---------- ENTROPY ----------
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = 0
    for i in range(len(elements)):
        prob = counts[i] / np.sum(counts)
        entropy_val += -prob * log2(prob)
    return entropy_val

# ---------- INFO GAIN ----------
def info_gain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        prob = counts[i] / np.sum(counts)
        weighted_entropy += prob * entropy(subset[target_name])
    return total_entropy - weighted_entropy

# ---------- ID3 ALGORITHM ----------
def id3(data, original_data, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(
            np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(
            np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        gains = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(gains)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = id3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree


# ---------- MAIN ----------
if __name__ == "__main__":
    df = pd.read_csv("Data/laptop_buy_data.csv")
    df.columns = df.columns.str.strip()  

    print("Columns:", df.columns.tolist())  

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    features = list(df.columns)
    target = 'Class'

    features.remove(target) 

    tree = id3(df, df, features, target)

    print("Decision Tree:")
    pprint.pprint(tree)
