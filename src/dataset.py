import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml


def preprocess_train(df, num_rare):
    train_features = pd.concat((df.iloc[:, 1], df.iloc[:, 3:]), axis=1)
    train_features = train_features.drop(columns = ["Address", "Summary", "Bedrooms", "Heating features", "Cooling features",
                                   "Parking features","Listed On","Last Sold On", "City", "State"])

    # deal with numeric features
    numeric_idx = train_features.dtypes[train_features.dtypes != 'object'].index
    numeric = train_features[numeric_idx]
    static_dict = dict()
    for name in numeric_idx:
        feature = numeric[name]
        m = feature.mean()
        s = feature.std()
        numeric.loc[:, name] = (feature - m) / s
        static_dict[name] = (m, s)
    joblib.dump(static_dict, "./static_dict.pkl")
    numeric = numeric.fillna(0)

    # deal with categorical features
    categorical_idx = train_features.dtypes[train_features.dtypes == 'object'].index
    categorical = train_features[categorical_idx]
    le_dict = dict()
    for name in categorical_idx:
        feature = categorical[name].fillna("unknown")
        feature = feature.astype(str).str.lower().str.strip()

        counts = feature.value_counts()
        rare_type = counts[counts < num_rare].index
        feature = feature.replace(rare_type, "other")

        le = LabelEncoder()
        categorical[name] = le.fit_transform(feature)
        le_dict[name] = le
    joblib.dump(le_dict, "./le_dict.pkl")

    # deal with label
    label = torch.tensor(
            df['Sold Price'].values, dtype=torch.float32
        )

    return (torch.tensor(
            numeric.values, dtype=torch.float32
        ),
        torch.tensor(
            categorical.values, dtype=torch.long
        ),
        torch.log1p(label))

def preprocess_test(df):
    test_features = df.iloc[:, 1:]
    test_features = test_features.drop(columns = ["Address", "Summary", "Bedrooms", "Heating features", "Cooling features",
                                    "Parking features","Listed On","Last Sold On", "City", "State"])

    # deal with numeric features
    numeric_idx = test_features.dtypes[test_features.dtypes != 'object'].index
    numeric = test_features[numeric_idx]
    static_dict = joblib.load("./static_dict.pkl")
    for name in numeric_idx:
        feature = numeric[name]
        m = static_dict[name][0]
        s = static_dict[name][1]
        numeric.loc[:,name] = (feature - m) / s
    numeric = numeric.fillna(0)

    # deal with categorical features
    categorical_idx = test_features.dtypes[test_features.dtypes == 'object'].index
    categorical = test_features[categorical_idx]
    le_dict = joblib.load("./le_dict.pkl")
    for name in categorical_idx:
        feature = categorical[name].fillna("unknown")
        feature = feature.astype(str).str.lower().str.strip()

        le = le_dict[name]
        valid_type = le.classes_
        feature = feature.where(feature.isin(valid_type), "other")

        categorical[name] = le.transform(feature)

    return (torch.tensor(
            numeric.values, dtype=torch.float32
        ), torch.tensor(
            categorical.values, dtype=torch.long
        ))


class HouseDataset(Dataset):
    def __init__(self, numeric, categorical, label):
        super().__init__()
        self.numeric = numeric
        self.categorical = categorical
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.numeric[idx], self.categorical[idx], self.label[idx]

def get_data_loader(batch_size, num_rare):
    #load configs
    with open('./configs/configs.yaml', "r") as f:
        configs = yaml.safe_load(f)

    processed_data_train_path = configs['processed_data_path'] + 'processed_train.pth'
    if configs['need_preprocess']:
        raw_data_train_path = configs['raw_data_path'] + 'train.csv'
        train_data = pd.read_csv(raw_data_train_path)
        numeric, categorical, label = preprocess_train(train_data, num_rare=num_rare)
        torch.save((numeric, categorical, label), processed_data_train_path)
    else:
        numeric, categorical, label = torch.load(processed_data_train_path)
    
    num_train = round(0.8 * len(label))
    dataset_train = HouseDataset(numeric[:num_train, :], categorical[:num_train, :], label[:num_train])
    dataset_valid = HouseDataset(numeric[num_train:, :], categorical[num_train:, :], label[num_train:])
    return (DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
            DataLoader(dataset_valid, batch_size=batch_size, shuffle=False))