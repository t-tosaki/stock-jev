import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

df = pd.read_csv(
    "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv",
    encoding="cp932",
    dtype=str,
)
df = df[df["日付"].str.fullmatch(r"\d{8}", na=False)]


def pick(category):
    return (
        df.loc[df["ニューインデックス区分"].isin([category]), ["コード", "銘柄名"]]
        .rename(columns={"コード": "code", "銘柄名": "name"})
        .to_dict("records")
    )


universe = {
    "as_of": df["日付"].iloc[0],
    "core30": pick("TOPIX Core30"),
    "large70": pick("TOPIX Large70"),
}
print({k: len(v) for k, v in universe.items() if k != "as_of"})

with open(DATA_DIR / "universe.json", "w", encoding="utf-8") as f:
    json.dump(universe, f, ensure_ascii=False, indent=2)
