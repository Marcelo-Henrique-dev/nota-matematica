import pandas as pd

def read_csv(file, top_n_quantity=20):
    chunks = pd.read_csv(file, chunksize=100000, sep=',', low_memory=False)
    df = next(chunks)
    print(df.columns)

    cols_to_keep = ["gender","race/ethnicity","parental level of education","lunch","test preparation course","math score","reading score","writing score"]
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df.dropna(subset="test preparation course")

    cols_for_dummies = ['parental level of education', 'lunch', 'test preparation course']
    existing_cols = [c for c in cols_for_dummies if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cols, dtype=int)

    df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]
    df_encoded.columns = df_encoded.columns.str.strip().str.replace(r"[^\w]", "_", regex=True)

    return df_encoded