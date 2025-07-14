import pandas as pd

def get_raw_data():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/gokaygokay/prompt-enhancer-dataset/" + splits["train"])
    print(df.head())
    df.to_csv("./prompts_raw.csv")
    print("Downloaded raw dataset.")
    return df

def preprocess_data(data : pd.DataFrame) -> pd.DataFrame:
    # Not too much to do here right now, maybe just trim extra white space on prompts. 
    # Will explore doing more things later. 
    data["short_prompt"] = data["short_prompt"].apply(lambda x: x.lower().strip())
    data["long_prompt"] = data["long_prompt"].apply(lambda x: x.lower().strip())
    return data

def get_splits():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    train_df = pd.read_parquet("hf://datasets/gokaygokay/prompt-enhancer-dataset/" + splits["train"])
    train_df = preprocess_data(train_df)
    # Split into test and val datasets
    val_df = train_df[int(len(train_df)*.9):]
    val_df.reset_index(inplace=True)
    val_df.drop("index", axis=1, inplace=True)
    train_df = train_df[:int(len(train_df)*.9)]
    train_df.to_csv("./splits/train.csv")
    val_df.to_csv("./splits/val.csv")
    test_df = pd.read_parquet("hf://datasets/gokaygokay/prompt-enhancer-dataset/" + splits["test"])
    test_df = preprocess_data(test_df)
    test_df.to_csv("./splits/test.csv")

if __name__ == "__main__":
    # get raw data
#    get_raw_data()
    # get splits
    get_splits()