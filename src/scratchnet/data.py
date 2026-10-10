"""Loading, cleaning and splitting the credit default data.

I keep all of it here so the baseline, my own network and the Keras
model all get exactly the same data.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_XLS = PROJECT_ROOT / "data" / "raw" / "default of credit card clients.xls"
TARGET = "default"
SEED = 42


def load_raw():
    """Reads the Excel file into a table.

    The file has two header rows, so I use the second one. I also give the
    target a short name and drop ID, because a customer number tells the
    model nothing about whether someone will pay.
    """
    df = pd.read_excel(RAW_XLS, header=1)
    df = df.rename(columns={"default payment next month": TARGET})
    return df.drop(columns="ID")


def merge_unknown_codes(df):
    """Puts the undocumented codes into the "others" group.

    EDUCATION 0, 5 and 6 become 4, and MARRIAGE 0 becomes 3. Why I chose
    this is written in data/README.md.
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def split(df):
    """Cuts the data into train (70%), dev (15%) and test (15%).

    It's stratified, so every part keeps about 22% defaulters, and seeded,
    so I get exactly the same split every time.
    """
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=SEED
    )
    dev, test = train_test_split(
        rest, test_size=0.50, stratify=rest[TARGET], random_state=SEED
    )
    return train, dev, test
