"""chapter06_classification: prepare spam splits.

Original learning file: 6-2.py. Run as a module from the repository root.
"""

from project_paths import DATA_DIR
import pandas as pd
from pathlib import Path

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]

    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])

    return balanced_df

if __name__ == "__main__":
    extracted_path = DATA_DIR / "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    df = pd.read_csv(
        data_file_path, sep="\t",header=None,names=["Label","text"]
    )
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())


    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

if __name__ == "__main__":
    train_df, validation_df, test_df = random_split(
        balanced_df, 0.7, 0.1
    )

    train_df.to_csv(DATA_DIR / "sms_spam_collection" / "train.csv", index=None)
    validation_df.to_csv(DATA_DIR / "sms_spam_collection" / "validation.csv", index=None)
    test_df.to_csv(DATA_DIR / "sms_spam_collection" / "test.csv", index=None)

import tiktoken

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
