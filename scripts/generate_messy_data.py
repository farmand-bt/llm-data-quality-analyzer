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


def generate_titanic_data(n: int = 891, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic Titanic-style dataset with intentional messiness.

    Issues baked in: inconsistent Sex labels, ~20% missing Age, ~77% missing
    Cabin, a few missing Embarked values, Fare values with "$" prefix (type
    mismatch), outlier fares, and 20 duplicate rows.
    """
    rng = np.random.default_rng(seed)

    survived = rng.integers(0, 2, n)
    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])

    titles = ["Mr.", "Mrs.", "Miss.", "Dr.", "Rev.", "Col."]
    first_names = ["James", "Mary", "John", "Anna", "William", "Emily",
                   "Thomas", "Rose", "George", "Edith", "Charles", "Alice"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
                  "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"]
    names = [
        f"{rng.choice(last_names)}, {rng.choice(titles)} {rng.choice(first_names)}"
        for _ in range(n)
    ]

    # Inconsistent Sex labels: male/female/Male/Female/M/F/MALE/FEMALE
    sex_values = []
    for _ in range(n):
        base = "male" if rng.random() < 0.65 else "female"
        fmt = int(rng.integers(0, 5))
        if fmt == 0:
            sex_values.append(base.upper())
        elif fmt == 1:
            sex_values.append(base[0].upper())
        elif fmt == 2:
            sex_values.append(base.capitalize())
        else:
            sex_values.append(base)

    # Age — ~20% missing
    age = rng.normal(29.7, 14.5, n).clip(0.17, 80.0).round(1).astype(float)
    age[rng.random(n) < 0.20] = np.nan

    sib_sp = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.06, 0.01, 0.01, 0.01])
    parch = rng.choice([0, 1, 2, 3, 4], n, p=[0.76, 0.13, 0.09, 0.01, 0.01])

    ticket_prefixes = ["PC", "CA", "A/5", "SOTON/OQ", ""]
    tickets = []
    for _ in range(n):
        pfx = rng.choice(ticket_prefixes)
        num = rng.integers(1000, 999999)
        tickets.append(f"{pfx} {num}".strip() if pfx else str(num))

    # Fare — outliers + some stored as "$X.XX" strings (type mismatch)
    fare_num = rng.exponential(32, n).clip(0.0, 200.0).round(2)
    for idx in rng.choice(n, size=8, replace=False):
        fare_num[idx] = float(rng.choice([0.0, 512.33, 263.0, 227.53]))
    fare_raw = fare_num.astype(object)
    for idx in rng.choice(n, size=25, replace=False):
        fare_raw[idx] = f"${float(fare_num[idx]):.2f}"

    # Cabin — ~77% missing
    cabin_letters = ["A", "B", "C", "D", "E", "F", "G"]
    cabin_vals = [
        np.nan if rng.random() < 0.77
        else f"{rng.choice(cabin_letters)}{rng.integers(1, 148)}"
        for _ in range(n)
    ]

    # Embarked — ~2% missing
    embarked_vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.019:
            embarked_vals.append(np.nan)
        elif r < 0.208:
            embarked_vals.append("C")
        elif r < 0.298:
            embarked_vals.append("Q")
        else:
            embarked_vals.append("S")

    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex_values,
        "Age": age,
        "SibSp": sib_sp,
        "Parch": parch,
        "Ticket": tickets,
        "Fare": fare_raw,
        "Cabin": cabin_vals,
        "Embarked": embarked_vals,
    })

    # 20 duplicate rows
    dup_idx = rng.choice(n, size=20, replace=False)
    df = pd.concat([df, df.iloc[dup_idx].copy()], ignore_index=True)
    df["PassengerId"] = range(1, len(df) + 1)

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

    titanic = generate_titanic_data()
    titanic.to_csv(sample_data / "titanic_messy.csv", index=False)
    print(f"Written {len(titanic)} rows -> sample_data/titanic_messy.csv")

    clean = generate_clean_sample()
    clean.to_csv(fixtures / "clean_sample.csv", index=False)
    print(f"Written {len(clean)} rows -> tests/fixtures/clean_sample.csv")


if __name__ == "__main__":
    main()
