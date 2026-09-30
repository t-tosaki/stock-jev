import os
from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta

import requests

from typesafe_sdk import Choice, TypeSafeClient

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

BASE_DAY_OFFSET = -1

# log
run_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
log_file = f"{run_time}.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / log_file).unlink(missing_ok=True)

# universe
with open(DATA_DIR / "universe.json") as f:
    universe = json.load(f)

tickers = universe["core30"]
print(universe["as_of"], len(tickers))

# topix
topix = yf.Ticker("1306.T").history(period="1y")
for n in (5, 25, 75):
    topix.ta.roc(length=n, append=True)

# market
market = {}
for ticker in ["NIY=F", "^GSPC", "USDJPY=X"]:
    df = yf.Ticker(ticker).history(period="1y")
    df.ta.roc(length=1, append=True)
    market[ticker] = round(df["ROC_1"].iloc[BASE_DAY_OFFSET].item(), 2)

vix = yf.Ticker("^VIX").history(period="1y")["Close"]
market["^VIX"] = round(vix.iloc[BASE_DAY_OFFSET].item(), 2)

# loop tickers
for t in tickers:
    code, name = t["code"], t["name"]
    print(code, name)

    stock = yf.Ticker(f"{code}.T")
    df = stock.history(period="1y")
    info = stock.info

    # trend
    df.ta.adx(length=14, append=True)
    df.ta.bias(length=25, append=True)
    df["BIAS_SMA_25"] *= 100
    df.ta.bias(length=75, append=True)
    df["BIAS_SMA_75"] *= 100

    # reversal
    df.ta.rsi(length=14, append=True)

    # return
    for n in (5, 25, 75):
        df.ta.roc(length=n, append=True)
        df[f"REL_{n}"] = df[f"ROC_{n}"] - topix[f"ROC_{n}"]

    # volatility
    df.ta.natr(length=14, append=True)

    # volume
    df["RVOL_25"] = df["Volume"] / df.ta.sma(close=df["Volume"], length=25)

    # fundamentals
    per = info.get("trailingPE")
    pbr = info.get("priceToBook")
    revenue_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")

    # info
    sector = info["sector"]
    industry = info["industry"]

    # state
    data_now = df.iloc[BASE_DAY_OFFSET].to_dict()
    data_5 = df.iloc[BASE_DAY_OFFSET - 5].to_dict()
    data_25 = df.iloc[BASE_DAY_OFFSET - 25].to_dict()

    state = {
        "sector": sector,
        "industry": industry,
        "market": {
            "nikkei_futures_roc1_percent": market["NIY=F"],
            "sp500_roc1_percent": market["^GSPC"],
            "usdjpy_roc1_percent": market["USDJPY=X"],
            "vix": market["^VIX"],
        },
        "trend": {
            "bias_sma25_percent": {
                "now": round(data_now["BIAS_SMA_25"], 2),
                "5_days_ago": round(data_5["BIAS_SMA_25"], 2),
                "25_days_ago": round(data_25["BIAS_SMA_25"], 2),
            },
            "bias_sma75_percent": round(data_now["BIAS_SMA_75"], 2),
            "adx14": {
                "now": round(data_now["ADX_14"], 1),
                "5_days_ago": round(data_5["ADX_14"], 1),
                "25_days_ago": round(data_25["ADX_14"], 1),
            },
        },
        "reversal": {
            "rsi14": {
                "now": round(data_now["RSI_14"], 1),
                "5_days_ago": round(data_5["RSI_14"], 1),
                "25_days_ago": round(data_25["RSI_14"], 1),
            }
        },
        "return": {
            "roc5_percent": round(data_now["ROC_5"], 2),
            "roc25_percent": round(data_now["ROC_25"], 2),
            "roc75_percent": round(data_now["ROC_75"], 2),
            "roc5_vs_topix_percent": round(data_now["REL_5"], 2),
            "roc25_vs_topix_percent": round(data_now["REL_25"], 2),
            "roc75_vs_topix_percent": round(data_now["REL_75"], 2),
        },
        "volume": {"rvol25_ratio": round(data_now["RVOL_25"], 2)},
        "volatility": {"natr14_percent": round(data_now["NATR_14"], 2)},
        "fundamentals": {
            "per": round(per, 2) if per is not None else None,
            "pbr": round(pbr, 2) if pbr is not None else None,
            "revenue_growth_percent": (
                round(revenue_growth * 100, 2) if revenue_growth is not None else None
            ),
            "earnings_growth_percent": (
                round(earnings_growth * 100, 2) if earnings_growth is not None else None
            ),
        },
    }

    # call jev
    with TypeSafeClient() as client:
        jev_result = client.system_one(
            state=state,
            questions={
                "signal": Choice(
                    instructions="Should I newly buy this Japanese stock (listed on the Tokyo Stock Exchange) for a short-term hold of 1 week to 1 month?",
                    criteria={
                        "strong_buy": "Likely to rise",
                        "neutral": "Mixed or unclear",
                        "avoid": "Likely to fall or stagnate",
                    },
                )
            },
        )

    # logging
    with open(LOGS_DIR / log_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "date": df.index[BASE_DAY_OFFSET].strftime("%Y-%m-%d"),
                    "code": code,
                    "name": name,
                    "close": df["Close"].iloc[BASE_DAY_OFFSET],
                    "state": state,
                    "model": jev_result.model,
                    "usage": dict(jev_result.usage),
                    "choice": jev_result.answers["signal"].choice,
                    "confidence": jev_result.answers["signal"].confidence,
                    "probabilities": dict(jev_result.answers["signal"].probabilities),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

# send notify
with open(LOGS_DIR / log_file) as f:
    CHOICE_ORDER = {
        "strong_buy": 0,
        "neutral": 1,
        "avoid": 2,
    }
    CHOICE_EMOJI = {
        "strong_buy": "🟢",
        "neutral": "🟡",
        "avoid": "🔴",
    }

    records = sorted(
        [json.loads(line) for line in f if line.strip()],
        key=lambda r: (
            CHOICE_ORDER.get(r["choice"], 99),
            -r["confidence"],
        ),
    )

    content = "\n".join(
        [
            f"{CHOICE_EMOJI.get(r['choice'], '❔')} {r['code']} {r['confidence']:.2f} "
            f"{r['probabilities'].get('strong_buy'):.2f}/{r['probabilities'].get('neutral'):.2f}/{r['probabilities'].get('avoid'):.2f} "
            f"({r['name']})"
            for r in records
        ]
    )
    content = "```\n" + content + "\n```"

    res = requests.post(os.environ["DISCORD_WEBHOOK_URL"], json={"content": content})
    res.raise_for_status()
