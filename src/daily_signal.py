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
    df["RSI_14_PCT"] = df["RSI_14"].expanding().rank(pct=True) * 100
    df.ta.willr(length=200, append=True)
    df.ta.donchian(lower_length=60, upper_length=60, append=True)
    df["DD_60"] = (df["Close"] / df["DCU_60_60"] - 1) * 100

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
            },
            "rsi14_percentile": round(data_now["RSI_14_PCT"], 1),
            "willr200": round(data_now["WILLR_200"], 1),
            "drawdown60_percent": {
                "now": round(data_now["DD_60"], 2),
                "5_days_ago": round(data_5["DD_60"], 2),
                "25_days_ago": round(data_25["DD_60"], 2),
            },
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
                ),
                "rebound": Choice(
                    instructions="Is this Japanese stock (listed on the Tokyo Stock Exchange) oversold and likely to rebound within 1 week to 1 month?",
                    criteria={
                        "likely_rebound": "Likely to rebound",
                        "unclear": "Mixed or unclear",
                        "still_falling": "Likely to keep falling",
                    },
                ),
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
                    "signal": {
                        "choice": jev_result.answers["signal"].choice,
                        "confidence": jev_result.answers["signal"].confidence,
                        "probabilities": dict(
                            jev_result.answers["signal"].probabilities
                        ),
                    },
                    "rebound": {
                        "choice": jev_result.answers["rebound"].choice,
                        "confidence": jev_result.answers["rebound"].confidence,
                        "probabilities": dict(
                            jev_result.answers["rebound"].probabilities
                        ),
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )

# send notify
with open(LOGS_DIR / log_file) as f:
    ORDER = {
        "strong_buy": 0,
        "neutral": 1,
        "avoid": 2,
        "likely_rebound": 0,
        "unclear": 1,
        "still_falling": 2,
    }
    EMOJI = {
        "strong_buy": "🟢",
        "neutral": "🟡",
        "avoid": "🔴",
        "likely_rebound": "🟢",
        "unclear": "🟡",
        "still_falling": "🔴",
    }

    records = [json.loads(line) for line in f if line.strip()]

    def ranked(key):
        return sorted(
            records,
            key=lambda r: (ORDER.get(r[key]["choice"], 99), -r[key]["confidence"]),
        )

    top = "\n".join(
        [
            "**signal**",
            "```",
            *[
                f"[{r['code']}] {r['signal']['probabilities']['strong_buy']:.2f} / {r['signal']['probabilities']['avoid']:.2f} ({r['name']})"
                for r in ranked("signal")
                if r["signal"]["choice"] == "strong_buy"
                and r["signal"]["probabilities"]["strong_buy"] >= 0.5
            ],
            "```",
            "**rebound**",
            "```",
            *[
                f"[{r['code']}] {r['rebound']['probabilities']['likely_rebound']:.2f} / {r['rebound']['probabilities']['still_falling']:.2f} ({r['name']})"
                for r in ranked("rebound")
                if r["rebound"]["choice"] == "likely_rebound"
                and r["rebound"]["probabilities"]["likely_rebound"] >= 0.5
            ],
            "```",
        ]
    )

    full = "\n".join(
        [
            f"[{r['code']}] "
            f"{EMOJI.get(r['signal']['choice'], '❔')} "
            f"({r['signal']['probabilities']['strong_buy']:.2f},{r['signal']['probabilities']['neutral']:.2f},{r['signal']['probabilities']['avoid']:.2f}) "
            f"{EMOJI.get(r['rebound']['choice'], '❔')} "
            f"({r['rebound']['probabilities']['likely_rebound']:.2f},{r['rebound']['probabilities']['unclear']:.2f},{r['rebound']['probabilities']['still_falling']:.2f}) "
            f"¥ {r['close']:>8,.1f} "
            f"({r['name']})"
            for r in ranked("signal")
        ]
    )

    res = requests.post(
        os.environ["DISCORD_WEBHOOK_URL"],
        data={"payload_json": json.dumps({"content": top})},
        files={"file": ("result.txt", full.encode("utf-8"))},
    )
    print(res.status_code, res.text)
    res.raise_for_status()
