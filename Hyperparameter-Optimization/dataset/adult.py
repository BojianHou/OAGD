import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def load_adult(batch_size=64):
    # age, workclass, fnlwgt, education, educational-num, marital-status
    # occupation, relationship, race, gender, capital-gain, capital-loss
    # hours-per-week, native-country, income

    data = pd.read_csv('./dataset/adult.csv')
    # data = pd.read_csv('adult.csv')
    data['workclass'] = data['workclass'].replace('?', np.nan)
    data['occupation'] = data['occupation'].replace('?', np.nan)
    data['native-country'] = data['native-country'].replace('?', np.nan)
    df = data.copy()
    df.dropna(how='any', inplace=True)
    df = df.drop_duplicates()

    df1 = df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1)
    label_encoder = preprocessing.LabelEncoder()
    df1['gender'] = label_encoder.fit_transform(df1['gender'])
    df1['workclass'] = label_encoder.fit_transform(df1['workclass'])
    df1['education'] = label_encoder.fit_transform(df1['education'])
    df1['marital-status'] = label_encoder.fit_transform(df1['marital-status'])
    df1['occupation'] = label_encoder.fit_transform(df1['occupation'])
    df1['relationship'] = label_encoder.fit_transform(df1['relationship'])
    df1['race'] = label_encoder.fit_transform(df1['race'])
    df1['native-country'] = label_encoder.fit_transform(df1['native-country'])
    df1['income'] = label_encoder.fit_transform(df1['income'])


    X = df1.drop(columns={"income"}, axis=1)
    y = df1["income"].values
    X = X.to_numpy()
    X = StandardScaler().fit_transform(X)

    X_rem, X_test, y_rem, y_test = \
        train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_rem, y_rem, test_size=0.5, random_state=42, stratify=y_rem)

    dataset_train = torch.utils.data.TensorDataset(torch.FloatTensor(torch.from_numpy(X_train).float()),
                                                   torch.LongTensor(y_train))
    dataset_val = torch.utils.data.TensorDataset(torch.FloatTensor(torch.from_numpy(X_val).float()),
                                                  torch.LongTensor(y_val))
    dataset_test = torch.utils.data.TensorDataset(torch.FloatTensor(torch.from_numpy(X_test).float()),
                                                   torch.LongTensor(y_test))
    # train, val = train_test_split(dataset_train, test_size=0.3,
    #                 random_state=42, stratify=dataset_train.tensors[1])  # dataset.tensors[1] is the target/label

    num_train_samples, num_val_samples = [], []
    num_class = 2
    for i in range(num_class):
        num_train_samples.append(np.sum(y_train == i))
        num_val_samples.append(np.sum(y_test == i))

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    eval_train_loader = train_loader
    eval_val_loader = val_loader

    img_size = dataset_train[0][0].shape

    return train_loader, val_loader, test_loader, eval_train_loader, \
           eval_val_loader, num_train_samples, num_val_samples, img_size


if __name__ ==  '__main__':
    load_adult()