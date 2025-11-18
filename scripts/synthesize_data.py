#!/usr/bin/env python3
"""
Synthetic data generator for DQ + anomaly detection + Snowflake-style schema.

Generates:
  - dim_customers
  - dim_products
  - dim_stores
  - fact_sales

Supports:
  - Data quality issues (--with-issues)
  - Anomalies for detection (--with-anomalies)

Usage examples:

    # Clean(ish) data
    python scripts/synthesize_data.py --rows 5000

    # With DQ issues + anomalies
    python scripts/synthesize_data.py --rows 20000 --with-issues --with-anomalies
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


PAYMENT_METHODS = ["CARD", "UPI", "CASH", "WALLET"]
CHANNELS = ["ONLINE", "OFFLINE"]
STATUSES = ["SUCCESS", "PENDING", "FAILED"]
COUNTRIES = ["IN", "US", "UK", "DE", "FR", "SG"]
CUSTOMER_SEGMENTS = ["RETAIL", "WHOLESALE", "ONLINE_ONLY", "VIP"]
PRODUCT_CATEGORIES = ["GROCERY", "ELECTRONICS", "CLOTHING", "HOME", "TOYS"]
REGIONS = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRAL"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic DQ + anomaly data.")
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of fact_sales rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/",
        help="Directory or prefix for output files (default: data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--with-issues",
        action="store_true",
        help="Introduce data quality issues (nulls, negatives, FK issues, etc).",
    )
    parser.add_argument(
        "--with-anomalies",
        action="store_true",
        help="Introduce anomalies (spikes, extreme values) for anomaly detection.",
    )
    return parser.parse_args()


def ensure_dir(prefix: str):
    # If prefix is like 'data/' ensure that directory exists
    directory = prefix if prefix.endswith("/") else os.path.dirname(prefix)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def random_datetime(start: datetime, end: datetime, faker: Faker) -> datetime:
    return faker.date_time_between(start_date=start, end_date=end)


def generate_dimensions(n_customers: int, n_products: int, n_stores: int, seed: int):
    faker = Faker()
    Faker.seed(seed)
    random.seed(seed)

    # Customers
    customers = []
    for cid in range(1, n_customers + 1):
        profile = faker.simple_profile()
        customers.append(
            {
                "customer_id": cid,
                "first_name": profile["name"].split(" ")[0],
                "last_name": " ".join(profile["name"].split(" ")[1:]) or None,
                "email": profile["mail"],
                "signup_date": faker.date_between(start_date="-3y", end_date="today").isoformat(),
                "country": random.choice(COUNTRIES),
                "segment": random.choice(CUSTOMER_SEGMENTS),
            }
        )
    dim_customers = pd.DataFrame(customers)

    # Products
    products = []
    for pid in range(1000, 1000 + n_products):
        category = random.choice(PRODUCT_CATEGORIES)
        base_price = round(random.uniform(50, 10000), 2)
        products.append(
            {
                "product_id": pid,
                "product_name": f"{category}_ITEM_{pid}",
                "category": category,
                "base_price": base_price,
            }
        )
    dim_products = pd.DataFrame(products)

    # Stores
    stores = []
    for sid in range(1, n_stores + 1):
        region = random.choice(REGIONS)
        stores.append(
            {
                "store_id": sid,
                "store_name": f"STORE_{sid}",
                "region": region,
                "country": random.choice(COUNTRIES),
                "is_online": random.choice([0, 1]),
            }
        )
    dim_stores = pd.DataFrame(stores)

    return dim_customers, dim_products, dim_stores


def generate_fact_sales(
    n_rows: int,
    dim_customers: pd.DataFrame,
    dim_products: pd.DataFrame,
    dim_stores: pd.DataFrame,
    seed: int,
    with_anomalies: bool,
):
    faker = Faker()
    Faker.seed(seed + 1)
    np.random.seed(seed + 1)
    random.seed(seed + 1)

    now = datetime.now()
    start_date = now - timedelta(days=365)

    customer_ids = dim_customers["customer_id"].tolist()
    product_ids = dim_products["product_id"].tolist()
    store_ids = dim_stores["store_id"].tolist()

    rows = []

    # For anomalies: choose a few special days for spikes/drops
    spike_day = start_date + timedelta(days=300)
    drop_day = start_date + timedelta(days=200)

    for i in range(1, n_rows + 1):
        # Base random timestamp
        ts = random_datetime(start_date, now, faker)

        if with_anomalies:
            # 1) Time-based anomalies: more transactions on spike_day, fewer on drop_day
            r = random.random()
            if r < 0.02:  # 2% rows forced to spike day
                ts = spike_day.replace(
                    hour=random.randint(10, 20),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59),
                )
            elif r < 0.03:  # 1% rows forced to drop day
                ts = drop_day.replace(
                    hour=random.randint(0, 23),
                    minute=random.randint(0, 59),
                    second=random.randint(0, 59),
                )

        customer_id = random.choice(customer_ids)
        product_id = random.choice(product_ids)
        store_id = random.choice(store_ids)

        # Base realistic quantity & price
        quantity = np.random.choice([1, 2, 3, 4, 5, 10], p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01])
        unit_price = float(
            max(10.0, np.random.normal(loc=500.0, scale=200.0))
        )  # mostly around 500, min 10
        total_amount = round(quantity * unit_price, 2)

        payment_method = random.choice(PAYMENT_METHODS)
        channel = random.choice(CHANNELS)
        status = random.choices(STATUSES, weights=[0.9, 0.05, 0.05], k=1)[0]

        # 2) Value-based anomalies
        if with_anomalies:
            # ~0.3% of rows with extremely high amount
            r_val = random.random()
            if r_val < 0.003:
                quantity = random.randint(50, 200)
                unit_price = round(random.uniform(5000, 20000), 2)
                total_amount = round(quantity * unit_price, 2)
            # ~0.3% of rows with extremely low amount (suspicious discounts)
            elif r_val < 0.006:
                quantity = 1
                unit_price = round(random.uniform(1, 5), 2)
                total_amount = round(quantity * unit_price, 2)

        rows.append(
            {
                "sale_id": i,
                "customer_id": customer_id,
                "store_id": store_id,
                "product_id": product_id,
                "sale_timestamp": ts.isoformat(timespec="seconds"),
                "quantity": int(quantity),
                "unit_price": round(unit_price, 2),
                "total_amount": total_amount,
                "payment_method": payment_method,
                "channel": channel,
                "status": status,
            }
        )

    fact_sales = pd.DataFrame(rows)
    return fact_sales


def inject_dq_issues(
    dim_customers: pd.DataFrame,
    dim_products: pd.DataFrame,
    dim_stores: pd.DataFrame,
    fact_sales: pd.DataFrame,
    seed: int,
):
    """
    Add classic DQ issues:

    - Nulls in attributes
    - Out-of-domain categorical values
    - Negative quantities
    - Inconsistent total_amount
    - Duplicated sale_id
    - Orphan FKs (no matching dim rows)
    """
    np.random.seed(seed + 2)
    df_cust = dim_customers.copy()
    df_prod = dim_products.copy()
    df_store = dim_stores.copy()
    df_fact = fact_sales.copy()

    n = len(df_fact)
    if n == 0:
        return df_cust, df_prod, df_store, df_fact

    def sample_idx(df, fraction: float):
        k = max(1, int(len(df) * fraction))
        return np.random.choice(df.index, size=k, replace=False)

    # 1) Nulls in dimensions (attributes, not keys)
    for col in ["email", "segment", "country"]:
        idx = sample_idx(df_cust, 0.02)
        df_cust.loc[idx, col] = None

    for col in ["category"]:
        idx = sample_idx(df_prod, 0.02)
        df_prod.loc[idx, col] = None

    for col in ["region"]:
        idx = sample_idx(df_store, 0.02)
        df_store.loc[idx, col] = None

    # 2) Out-of-domain categorical values
    idx_bad_pay = sample_idx(df_fact, 0.01)
    df_fact.loc[idx_bad_pay, "payment_method"] = "CRYPTO"

    idx_bad_status = sample_idx(df_fact, 0.005)
    df_fact.loc[idx_bad_status, "status"] = "UNKNOWN"

    # 3) Negative quantities
    idx_neg_qty = sample_idx(df_fact, 0.01)
    df_fact.loc[idx_neg_qty, "quantity"] = -df_fact.loc[idx_neg_qty, "quantity"].abs()

    # 4) Inconsistent total_amount (break relationship)
    idx_bad_total = sample_idx(df_fact, 0.03)
    df_fact.loc[idx_bad_total, "total_amount"] = (
        df_fact.loc[idx_bad_total, "total_amount"] * np.random.uniform(0.5, 1.5, size=len(idx_bad_total))
    ).round(2)

    # 5) Duplicate sale_id rows
    idx_dupes = sample_idx(df_fact, 0.02)
    dup_rows = df_fact.loc[idx_dupes].copy()
    df_fact = pd.concat([df_fact, dup_rows], ignore_index=True)

    # 6) Orphan foreign keys in fact table
    #    Add some customer_id / product_id / store_id values not present in dims
    idx_fk_orphans = sample_idx(df_fact, 0.02)
    max_cust = df_cust["customer_id"].max()
    max_prod = df_prod["product_id"].max()
    max_store = df_store["store_id"].max()

    df_fact.loc[idx_fk_orphans, "customer_id"] = np.random.randint(max_cust + 1, max_cust + 100, size=len(idx_fk_orphans))
    df_fact.loc[idx_fk_orphans, "product_id"] = np.random.randint(max_prod + 1, max_prod + 50, size=len(idx_fk_orphans))
    df_fact.loc[idx_fk_orphans, "store_id"] = np.random.randint(max_store + 1, max_store + 20, size=len(idx_fk_orphans))

    return df_cust, df_prod, df_store, df_fact


def main():
    args = parse_args()
    ensure_dir(args.output_prefix)

    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Generating dimensions and {args.rows} fact rows...")
    # Heuristic: fewer dimension rows than facts
    n_customers = max(100, args.rows // 50)
    n_products = max(50, args.rows // 100)
    n_stores = max(10, args.rows // 500)

    dim_customers, dim_products, dim_stores = generate_dimensions(
        n_customers=n_customers,
        n_products=n_products,
        n_stores=n_stores,
        seed=args.seed,
    )

    fact_sales = generate_fact_sales(
        n_rows=args.rows,
        dim_customers=dim_customers,
        dim_products=dim_products,
        dim_stores=dim_stores,
        seed=args.seed,
        with_anomalies=args.with_anomalies,
    )

    if args.with_issues:
        print("[INFO] Injecting data quality issues into dimensions and fact_sales...")
        dim_customers, dim_products, dim_stores, fact_sales = inject_dq_issues(
            dim_customers, dim_products, dim_stores, fact_sales, args.seed
        )

    # Output file paths
    prefix = args.output_prefix
    if prefix.endswith("/"):
        base = prefix
    else:
        base = prefix + "_"

    path_dim_cust = os.path.join(base if base.endswith("/") else "", "dim_customers.csv")
    path_dim_prod = os.path.join(base if base.endswith("/") else "", "dim_products.csv")
    path_dim_store = os.path.join(base if base.endswith("/") else "", "dim_stores.csv")
    path_fact = os.path.join(base if base.endswith("/") else "", "fact_sales.csv")

    print(f"[INFO] Writing dimensions and fact table to: {base}")
    dim_customers.to_csv(path_dim_cust, index=False)
    dim_products.to_csv(path_dim_prod, index=False)
    dim_stores.to_csv(path_dim_store, index=False)
    fact_sales.to_csv(path_fact, index=False)

    print("[INFO] Done.")
    print(f"[INFO] Files generated:")
    print(f"  - {path_dim_cust}")
    print(f"  - {path_dim_prod}")
    print(f"  - {path_dim_store}")
    print(f"  - {path_fact}")


if __name__ == "__main__":
    main()
