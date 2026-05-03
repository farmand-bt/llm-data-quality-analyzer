"""Generate synthetic messy datasets for testing and demo purposes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def generate_housing_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    prices = rng.normal(350_000, 80_000, n).round()
    sqft = rng.normal(1800, 400, n).round()
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n)
    age = rng.integers(0, 80, n)
    neighborhood = rng.choice(["Downtown", "Suburbs", "Rural", "Midtown"], n)
    rating = rng.uniform(1, 10, n).round(1)

    # Inconsistent date formats across rows
    dates = []
    for i in range(n):
        fmt = i % 3
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(i * 365 / n))
        if fmt == 0:
            dates.append(base.strftime("%Y-%m-%d"))
        elif fmt == 1:
            dates.append(base.strftime("%m/%d/%Y"))
        else:
            dates.append(base.strftime("%d-%b-%Y"))

    garage = rng.choice(["Yes", "No", "yes", "no", "Y", "N", "TRUE", "FALSE"], n)

    df = pd.DataFrame({
        "id": range(1, n + 1),
        "price": prices,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "neighborhood": neighborhood,
        "sale_date": dates,
        "garage": garage,
        "school_rating": rating,
    })

    # Missing values — random
    for col in ["price", "sqft", "school_rating"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    # Missing values — structured (older homes lack school_rating)
    df.loc[df["age_years"] > 60, "school_rating"] = np.nan

    # Duplicate rows (20 rows repeated)
    dup_indices = rng.choice(n, size=20, replace=False)
    df = pd.concat([df, df.iloc[dup_indices].copy()], ignore_index=True)

    # Mixed types — replace some sqft values with strings
    df["sqft"] = df["sqft"].astype(object)
    mixed_idx = rng.choice(len(df), size=15, replace=False)
    for idx in mixed_idx:
        df.at[idx, "sqft"] = rng.choice(["N/A", "unknown", "TBD"])

    # Outliers — extreme price values
    outlier_idx = rng.choice(len(df), size=10, replace=False)
    for idx in outlier_idx:
        df.at[idx, "price"] = float(rng.choice([5_000, 9_999_999, -1_000]))

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_clean_sample(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"Person_{i}" for i in range(1, n + 1)],
        "age": rng.integers(20, 65, n),
        "salary": rng.normal(60_000, 15_000, n).round(),
        "department": rng.choice(["Engineering", "Marketing", "Sales", "HR"], n),
    })


def main():
    root = Path(__file__).resolve().parent.parent

    fixtures = root / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    sample_data = root / "sample_data"
    sample_data.mkdir(parents=True, exist_ok=True)

    messy = generate_housing_data()
    messy.to_csv(fixtures / "messy_sample.csv", index=False)
    messy.to_csv(sample_data / "housing_messy.csv", index=False)
    print(f"Written {len(messy)} rows -> tests/fixtures/messy_sample.csv")
    print(f"Written {len(messy)} rows -> sample_data/housing_messy.csv")

    clean = generate_clean_sample()
    clean.to_csv(fixtures / "clean_sample.csv", index=False)
    print(f"Written {len(clean)} rows -> tests/fixtures/clean_sample.csv")


if __name__ == "__main__":
    main()
