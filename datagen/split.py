import pandas as pd

def split_csv(file_path, num_splits):
    data = pd.read_csv(file_path)
    
    split_size = len(data) // num_splits
    for i in range(num_splits):
        start = i * split_size
        if i == num_splits - 1:
            end = len(data)
        else:
            end = start + split_size
        
        split_data = data.iloc[start:end]
        split_data.to_csv(f'{file_path}_part_{i+1}.csv', index=False)

split_csv('/hpc2hdd/home/jzhao815/datagen/cold_train.csv', 5)
