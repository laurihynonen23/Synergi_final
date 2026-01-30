import numpy as np
import pandas as pd


def read_prices(path, expected_rows=None):
    df = pd.read_csv(path, sep=";", decimal=",")
    if "Time" not in df.columns:
        raise ValueError("Missing Time column")
    price_cols = [c for c in df.columns if c != "Time"]
    if not price_cols:
        raise ValueError("Missing price column")
    price_col = price_cols[0]
    df = df[["Time", price_col]].rename(columns={price_col: "c_per_kwh"})

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)

    if df["Time"].isna().any():
        raise ValueError("Time column has NaT values")
    if df["c_per_kwh"].isna().any():
        raise ValueError("Price column has NaN values")
    if df["Time"].duplicated().any():
        raise ValueError(
            "Time column has duplicate timestamps (possible DST overlap). "
            "Use UTC timestamps or a range without DST overlaps."
        )

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if len(df) % 4 != 0:
        raise ValueError("Row count must be divisible by 4")

    diffs = df["Time"].diff().dropna()
    if not (diffs == pd.Timedelta(minutes=15)).all():
        raise ValueError(
            "Time steps are not consistently 15 minutes (possible DST shift or missing data). "
            "Use UTC timestamps or a range without DST gaps/overlaps."
        )

    df["eur_per_kwh"] = df["c_per_kwh"] / 100.0
    if df["eur_per_kwh"].isna().any():
        raise ValueError("EUR prices contain NaN values")

    return df, df["eur_per_kwh"].to_numpy(dtype=float)


def make_hourly_price_repeated(df):
    if "Time" not in df.columns or "eur_per_kwh" not in df.columns:
        raise ValueError("Input df must contain Time and eur_per_kwh")
    hour = df["Time"].dt.floor("h")
    sizes = df.groupby(hour)["eur_per_kwh"].size()
    if not (sizes == 4).all():
        raise ValueError("Each hour must have exactly 4 quarter-hour samples")
    p_hour = df.groupby(hour)["eur_per_kwh"].mean()
    p60 = hour.map(p_hour).to_numpy(dtype=float)
    if np.isnan(p60).any():
        raise ValueError("Hourly price series contains NaN values")
    return p60
