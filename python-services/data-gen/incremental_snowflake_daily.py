import os
import random
import uuid
import argparse
from datetime import datetime, date, timedelta

from faker import Faker
from dotenv import load_dotenv
import snowflake.connector

fake = Faker()

# -----------------------------------
# 1) CLI Args: batch date
# -----------------------------------
parser = argparse.ArgumentParser(description="Incremental daily load into Snowflake.")
parser.add_argument(
    "--batch-date",
    help="Batch date in YYYY-MM-DD format (default: today)",
    required=False,
)
args = parser.parse_args()

if args.batch_date:
    BATCH_DATE = datetime.strptime(args.batch_date, "%Y-%m-%d").date()
else:
    BATCH_DATE = date.today()

print(f"🚚 Running incremental load for batch date: {BATCH_DATE}")

# -----------------------------------
# 2) Load Snowflake Credentials
# -----------------------------------
load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE", "SALES_DQ"),
    schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
)
cursor = conn.cursor()

# -----------------------------------
# 3) Config: daily increments + DQ
# -----------------------------------
DAILY_CONFIG = {
    "DIM_CUSTOMERS": {"new_rows": 500},   # new customers per day
    "DIM_PRODUCTS":  {"new_rows": 200},   # new products per day
    "DIM_STORES":    {"new_rows": 50},    # new stores per day
    "FACT_SALES":    {"new_rows": 50_000} # new sales per day
}

DQ_RULES = {
    "dims": {
        "null_pct": 0.02,
        "wrong_type_pct": 0.01,
        "outlier_pct": 0.02,
        "duplicate_pct": 0.02,
    },
    "fact": {
        "duplicate_pct": 0.02,
        "wrong_fk_pct": 0.05,
        "high_amount_outlier_pct": 0.03,
        "future_date_pct": 0.02,
    },
}

TABLES = {
    "DIM_CUSTOMERS": "SALES_DQ.PUBLIC.DIM_CUSTOMERS",
    "DIM_PRODUCTS": "SALES_DQ.PUBLIC.DIM_PRODUCTS",
    "DIM_STORES":   "SALES_DQ.PUBLIC.DIM_STORES",
    "FACT_SALES":   "SALES_DQ.PUBLIC.FACT_SALES"
}

BATCH_SIZE = 10_000


# -----------------------------------
# 4) Helpers
# -----------------------------------
def inject_dim_dq(value, rules):
    """Apply DQ issues for DIM columns (nullable ones)."""
    r = random.random()

    if r < rules["null_pct"]:
        return None

    if r < rules["wrong_type_pct"]:
        if isinstance(value, (int, float)):
            return fake.word()
        else:
            return 9999

    if r < rules["outlier_pct"]:
        if isinstance(value, int):
            return random.randint(100000, 999999)
        if isinstance(value, float):
            return value * 1000

    return value


def gen_customer():
    return {
        "CUSTOMER_ID": str(uuid.uuid4()),
        "CUSTOMER_NAME": fake.name(),
        "COUNTRY": fake.country(),
        "SEGMENT": random.choice(["Retail", "Corporate", "SMB"]),
    }


def gen_product():
    return {
        "PRODUCT_ID": str(uuid.uuid4()),
        "PRODUCT_NAME": fake.word().title(),
        "CATEGORY": random.choice(["Electronics", "Clothing", "Sports", "Home", "Toys"]),
        "SUBCATEGORY": fake.word().title(),
        "PRICE": round(random.uniform(10, 3000), 2),
    }


def gen_store():
    return {
        "STORE_ID": str(uuid.uuid4()),
        "STORE_NAME": fake.company(),
        "REGION": random.choice(["APAC", "EMEA", "AMER"]),
        "CITY": fake.city(),
    }


def random_time_in_day(d: date):
    start_dt = datetime.combine(d, datetime.min.time())
    return start_dt + timedelta(seconds=random.randint(0, 86399))


def gen_fact(customer_ids, product_ids, store_ids):
    if not customer_ids or not product_ids or not store_ids:
        # still generate, but these will be broken FKs – which is also a DQ pattern
        cust = str(uuid.uuid4())
        prod = str(uuid.uuid4())
        store = str(uuid.uuid4())
    else:
        cust = random.choice(customer_ids)
        prod = random.choice(product_ids)
        store = random.choice(store_ids)

    return {
        "SALE_ID": str(uuid.uuid4()),
        "CUSTOMER_ID": cust,
        "PRODUCT_ID": prod,
        "STORE_ID": store,
        "AMOUNT": round(random.uniform(20, 20000), 2),
        "SALE_TS": random_time_in_day(BATCH_DATE),
    }


def batch_insert(table_name, records):
    if not records:
        return
    cols = list(records[0].keys())
    placeholders = ",".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({placeholders})"
    cursor.executemany(sql, [tuple(r[c] for c in cols) for r in records])
    conn.commit()


def load_existing_ids():
    print("🔎 Loading existing dimension IDs from Snowflake...")
    cursor.execute("SELECT CUSTOMER_ID FROM SALES_DQ.PUBLIC.DIM_CUSTOMERS")
    customer_ids = [r[0] for r in cursor.fetchall()]

    cursor.execute("SELECT PRODUCT_ID FROM SALES_DQ.PUBLIC.DIM_PRODUCTS")
    product_ids = [r[0] for r in cursor.fetchall()]

    cursor.execute("SELECT STORE_ID FROM SALES_DQ.PUBLIC.DIM_STORES")
    store_ids = [r[0] for r in cursor.fetchall()]

    print(
        f"   Loaded {len(customer_ids)} customers, "
        f"{len(product_ids)} products, {len(store_ids)} stores."
    )
    return customer_ids, product_ids, store_ids


# -----------------------------------
# 5) Incremental Load
# -----------------------------------
customer_ids, product_ids, store_ids = load_existing_ids()

# ---- DIM_CUSTOMERS ----
dim_dq = DQ_RULES["dims"]

for dim_name in ["DIM_CUSTOMERS", "DIM_PRODUCTS", "DIM_STORES"]:
    cfg = DAILY_CONFIG[dim_name]
    target = cfg["new_rows"]
    table_fqn = TABLES[dim_name]

    print(f"\n📌 Inserting {target} new rows into {table_fqn} for {BATCH_DATE}...")

    records = []
    for _ in range(target):
        if dim_name == "DIM_CUSTOMERS":
            r = gen_customer()
            r["CUSTOMER_NAME"] = inject_dim_dq(r["CUSTOMER_NAME"], dim_dq)
            r["COUNTRY"] = inject_dim_dq(r["COUNTRY"], dim_dq)
            r["SEGMENT"] = inject_dim_dq(r["SEGMENT"], dim_dq)
            customer_ids.append(r["CUSTOMER_ID"])

        elif dim_name == "DIM_PRODUCTS":
            r = gen_product()
            r["PRODUCT_NAME"] = inject_dim_dq(r["PRODUCT_NAME"], dim_dq)
            r["CATEGORY"] = inject_dim_dq(r["CATEGORY"], dim_dq)
            r["SUBCATEGORY"] = inject_dim_dq(r["SUBCATEGORY"], dim_dq)
            r["PRICE"] = inject_dim_dq(r["PRICE"], dim_dq)
            product_ids.append(r["PRODUCT_ID"])

        elif dim_name == "DIM_STORES":
            r = gen_store()
            r["STORE_NAME"] = inject_dim_dq(r["STORE_NAME"], dim_dq)
            r["REGION"] = inject_dim_dq(r["REGION"], dim_dq)
            r["CITY"] = inject_dim_dq(r["CITY"], dim_dq)
            store_ids.append(r["STORE_ID"])

        records.append(r)

        if len(records) >= BATCH_SIZE:
            batch_insert(table_fqn, records)
            records = []

    batch_insert(table_fqn, records)
    print(f"   ✅ Done inserting into {table_fqn}")

# ---- FACT_SALES ----
fact_cfg = DAILY_CONFIG["FACT_SALES"]
fact_dq = DQ_RULES["fact"]
fact_rows = fact_cfg["new_rows"]
fact_table = TABLES["FACT_SALES"]

print(f"\n📌 Inserting {fact_rows} new FACT_SALES rows for {BATCH_DATE} into {fact_table}...")

records = []
for _ in range(fact_rows):
    r = gen_fact(customer_ids, product_ids, store_ids)

    # DQ: wrong foreign keys
    if random.random() < fact_dq["wrong_fk_pct"]:
        fk_to_break = random.choice(["CUSTOMER_ID", "PRODUCT_ID", "STORE_ID"])
        r[fk_to_break] = str(uuid.uuid4())

    # DQ: high amount outliers
    if random.random() < fact_dq["high_amount_outlier_pct"]:
        r["AMOUNT"] = round(random.uniform(100_000, 1_000_000), 2)

    # DQ: future sale dates
    if random.random() < fact_dq["future_date_pct"]:
        r["SALE_TS"] = datetime.now() + timedelta(days=random.randint(1, 365))

    records.append(r)

    if len(records) >= BATCH_SIZE:
        batch_insert(fact_table, records)
        records = []

batch_insert(fact_table, records)
print(f"   ✅ Done inserting into {fact_table}")

# -----------------------------------
# 6) Close
# -----------------------------------
cursor.close()
conn.close()
print("\n🎉 Incremental load completed.\n")