import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_tadpole(batch_size=128):
    # DXCHANGE: 1=Stable: NL to NL; 2=Stable: MCI to MCI; 3=Stable: Dementia to Dementia;
    # 4=Conversion: NL to MCI; 5=Conversion: MCI to Dementia; 6=Conversion: NL to Dementia;
    # 7=Reversion: MCI to NL; 8=Reversion: Dementia to MCI; 9=Reversion: Dementia to NL。
    # MCI: DXCHANGE should be 2, 4, 8; AD: DXCHANGE should be 3, 5, 6
    features = ['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate',
                'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
                'FDG', 'AV45', 'ABETA_UPENNBIOMK9_04_19_17',
                'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17',
                'APOE4', 'AGE', 'ADAS13', 'Ventricles']
    df_tadpole = pd.read_csv('./dataset/TADPOLE_D1_D2.csv')
    df_MCI = df_tadpole[(df_tadpole.DXCHANGE == 2) | (df_tadpole.DXCHANGE == 4) | (df_tadpole.DXCHANGE == 8)]
    # df_MCI = df_tadpole[(df_tadpole.DXCHANGE == 2)]
    df_AD = df_tadpole[(df_tadpole.DXCHANGE == 3) | (df_tadpole.DXCHANGE == 5) | (df_tadpole.DXCHANGE == 6)]
    # df_AD = df_tadpole[(df_tadpole.DXCHANGE == 3)]
    # X = pd.concat([df_MCI, df_AD])
    len_AD = int(1/2*len(df_AD))
    X = pd.concat([df_MCI, df_AD[:len_AD]])  # select part of AD to make more imbalanced data
    X = X[features]
    X = X.apply(pd.to_numeric, errors='coerce')  # fill all the blank cells with NaN
    X = X.dropna(axis=1, how='all')
    X.fillna(X.mean(), inplace=True)
    X = X.to_numpy()
    X = StandardScaler().fit_transform(X)
    y = np.concatenate([np.zeros(len(df_MCI)), np.ones(len_AD)])
    X_rem, X_test, y_rem, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_rem, y_rem, test_size=0.3, random_state=42, stratify=y_rem)

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