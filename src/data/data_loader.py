import pandas as pd
import os
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sympy import rf
import torch
from torch.utils.data import TensorDataset,DataLoader
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(cfg , seed ):

    filepath = cfg['dataset']['filepath']
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset at: {os.path.exists(filepath)}. Please run data extraction first.")
    
    df = pd.read_csv(filepath)

    #  Extract targets and IDs
    target_col = cfg['dataset']['target_column']
    id_col = cfg['dataset']['id_column']    

    X_raw = df.drop(columns=[id_col,target_col]) #6 feature
    y_raw = df[target_col]# 4 labels in total 


    encoder = LabelEncoder()
    y_encoder = encoder.fit_transform(y_raw) 

    X_train_raw , X_temp , y_train , y_temp = train_test_split(
        X_raw,y_encoder , test_size=0.3 , random_state=seed , stratify = y_encoder
        )
    
    X_val_raw , X_test_raw , y_val ,y_test = train_test_split(
        X_temp , y_temp , test_size = 0.5 , random_state=seed , stratify = y_temp
    ) 

    #plot_pca(X_raw.values , y_encoder) #pca plotting for visualization of the data distribution in 2D space

    # print("Training Class distribution")
    # print(pd.Series(y_train).value_counts().sort_index())

    # print("Test Class  distribution")
    # print(pd.Series(y_test).value_counts().sort_index())

    # print("VAL Class  distribution")
    # print(pd.Series(y_val).value_counts().sort_index())

    # print("Average of each feature for each class")
    # #print(pd.concat([X_raw, y_raw], axis=1).groupby(target_col).mean())
    # grouped = pd.concat([X_raw, y_raw], axis=1).groupby(target_col).mean()

    # print(grouped.to_string())
    

    # rf = RandomForestClassifier(random_state=42)
    # rf.fit(X_raw, y_raw)

    # feature_names = X_raw.columns

    # print("\nFeature Importances")
    # print("-" * 40)

    # for name, score in sorted(zip(feature_names, rf.feature_importances_),
    #                         key=lambda x: x[1],
    #                         reverse=True):
    #     print(f"{name:30s} {score:.4f}")


    # Standard Scaling (Features MUST be normalized for Neural Networks)
    scaller = StandardScaler()
    X_train = scaller.fit_transform(X_train_raw)
    X_val = scaller.transform(X_val_raw)
    X_test = scaller.transform(X_test_raw)

    # Create PyTorch DataLoadersy_test
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    batch_size = cfg['dataset']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Return both loaders (for NN) and raw scaled arrays (for XGBoost)

    
    return {
        "loaders": (train_loader, val_loader, test_loader),
        "arrays": (X_train, y_train, X_val, y_val, X_test, y_test),
        "label_encoder": encoder
    }




def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6,5))

    for cls in sorted(set(y)):
        idx = y == cls
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {cls}", alpha=0.6)

    plt.legend()
    plt.title("PCA of cfDNA Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
#==========================================
# CALL AND INSPECT THE OUTPUT
# ==========================================
if __name__ == "__main__":
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir,"..", "config.yaml")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    output = load_and_preprocess_data(cfg=config_dict, seed=42)
    
   
    '''
    train_loader, val_loader, test_loader = output["loaders"]
    X_train, y_train, X_val, y_val, X_test, y_test = output["arrays"]
    print("------- INSPECTING NUMPY ARRAYS (For XGBoost) -------")
    print(f"X_train matrix shape: {X_train.shape} | First row preview: {X_train[0]}")
    print(f"y_train labels shape: {y_train.shape} | First row preview: {y_train[0]}")
    print(f"X_test matrix shape:  {X_test.shape}")
    
    print("\n------- INSPECTING PYTORCH DATALOADERS (For FFN/CNN) -------")
    # Grab just one single batch from the DataLoader stream
    first_batch_features, first_batch_targets = next(iter(train_loader))
    
    print(f"Features Batch Tensor Shape: {first_batch_features.shape}") # Should be (32, 6)
    print(f"Targets Batch Tensor Shape:  {first_batch_targets.shape}")  # Should be (32,)
    print(f"First element in batch data:\n{first_batch_features[0]}")
     
       '''
    