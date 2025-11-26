import os
import random
import uuid
from faker import Faker
from datetime import datetime, timedelta
from dotenv import load_dotenv
import snowflake.connector
from tqdm import tqdm  # optional progress bar

fake = Faker()

# -----------------------------------
# 1) Load Snowflake Credentials
# -----------------------------------
load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

cursor = conn.cursor()

cursor.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
cursor.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")
cursor.execute(f"USE SCHEMA {os.getenv('SNOWFLAKE_SCHEMA')}")

# -----------------------------------
# 2) Record Counts + DQ Rules
# -----------------------------------
CONFIG = {
    "DIM_CUSTOMERS": {"rows": 10_000},
    "DIM_PRODUCTS": {"rows": 10_000},
    "DIM_STORES": {"rows": 10_000},
    "FACT_SALES": {"rows": 1_000_000},
}

DQ_RULES = {
    "dims": {
        "null_pct": 0.03,
        "wrong_type_pct": 0.01,
        "outlier_pct": 0.02,
        "duplicate_pct": 0.02
    },
    "fact": {
        "duplicate_pct": 0.03,
        "wrong_fk_pct": 0.07,
        "high_amount_outlier_pct": 0.05,
        "future_date_pct": 0.03
    }
}

TABLES = {
    "DIM_CUSTOMERS": "SALES_DQ.PUBLIC.DIM_CUSTOMERS",
    "DIM_PRODUCTS": "SALES_DQ.PUBLIC.DIM_PRODUCTS",
    "DIM_STORES":   "SALES_DQ.PUBLIC.DIM_STORES",
    "FACT_SALES":   "SALES_DQ.PUBLIC.FACT_SALES"
}

BATCH_SIZE = 10_000  # insert in chunks


# -----------------------------------
# 3) DQ Helpers
# -----------------------------------
def inject_dim_dq(value, rules):
    r = random.random()

    if r < rules["null_pct"]:
        return None

    if r < rules["wrong_type_pct"]:
        return fake.word() if isinstance(value, (int, float)) else 9999

    if r < rules["outlier_pct"]:
        if isinstance(value, int):
            return random.randint(100000, 999999)
        if isinstance(value, float):
            return value * 1000

    return value


# -----------------------------------
# 4) Data Generators
# -----------------------------------
def gen_customer():
    return {
        "CUSTOMER_ID": str(uuid.uuid4()),
        "CUSTOMER_NAME": fake.name(),
        "COUNTRY": fake.country(),
        "SEGMENT": random.choice(["Retail", "Corporate", "SMB"])
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

def gen_fact(customer_ids, product_ids, store_ids):
    return {
        "SALE_ID": str(uuid.uuid4()),
        "CUSTOMER_ID": random.choice(customer_ids),
        "PRODUCT_ID": random.choice(product_ids),
        "STORE_ID": random.choice(store_ids),
        "AMOUNT": round(random.uniform(20, 20000), 2),
        "SALE_TS": fake.date_time_between(start_date="-2y", end_date="now"),
    }

# -----------------------------------
# 5) Insert Helper
# -----------------------------------
def batch_insert(table_name, records):
    if not records:
        return

    cols = list(records[0].keys())
    placeholder = ",".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({placeholder})"

    cursor.executemany(sql, [tuple(r[c] for c in cols) for r in records])
    conn.commit()


# -----------------------------------
# 6) LOAD DIM TABLES
# -----------------------------------
customer_ids = []
product_ids = []
store_ids = []

for table in ["DIM_CUSTOMERS", "DIM_PRODUCTS", "DIM_STORES"]:
    rows = CONFIG[table]["rows"]
    records = []
    dq = DQ_RULES["dims"]

    print(f"\n📌 Generating {rows:,} rows for {table}...")

    for _ in tqdm(range(rows)):
        if table == "DIM_CUSTOMERS":
            r = gen_customer()
            r["CUSTOMER_NAME"] = inject_dim_dq(r["CUSTOMER_NAME"], dq)
            r["COUNTRY"] = inject_dim_dq(r["COUNTRY"], dq)
            r["SEGMENT"] = inject_dim_dq(r["SEGMENT"], dq)
            customer_ids.append(r["CUSTOMER_ID"])

        elif table == "DIM_PRODUCTS":
            r = gen_product()
            r["PRODUCT_NAME"] = inject_dim_dq(r["PRODUCT_NAME"], dq)
            r["CATEGORY"] = inject_dim_dq(r["CATEGORY"], dq)
            r["SUBCATEGORY"] = inject_dim_dq(r["SUBCATEGORY"], dq)
            r["PRICE"] = inject_dim_dq(r["PRICE"], dq)
            product_ids.append(r["PRODUCT_ID"])

        elif table == "DIM_STORES":
            r = gen_store()
            r["STORE_NAME"] = inject_dim_dq(r["STORE_NAME"], dq)
            r["REGION"] = inject_dim_dq(r["REGION"], dq)
            r["CITY"] = inject_dim_dq(r["CITY"], dq)
            store_ids.append(r["STORE_ID"])

        records.append(r)

        if len(records) >= BATCH_SIZE:
            batch_insert(TABLES[table], records)
            records = []

    batch_insert(TABLES[table], records)


# -----------------------------------
# 7) LOAD FACT TABLE (1M rows)
# -----------------------------------
rows = CONFIG["FACT_SALES"]["rows"]
dq = DQ_RULES["fact"]
records = []

print(f"\n📌 Generating {rows:,} FACT_SALES rows...")

for _ in tqdm(range(rows)):
    r = gen_fact(customer_ids, product_ids, store_ids)

    # Wrong FK
    if random.random() < dq["wrong_fk_pct"]:
        fk = random.choice(["CUSTOMER_ID", "PRODUCT_ID", "STORE_ID"])
        r[fk] = str(uuid.uuid4())

    # Amount outlier
    if random.random() < dq["high_amount_outlier_pct"]:
        r["AMOUNT"] = random.randint(100_000, 1_000_000)

    # Future timestamps
    if random.random() < dq["future_date_pct"]:
        r["SALE_TS"] = datetime.now() + timedelta(days=random.randint(1, 365))

    records.append(r)

    if len(records) >= BATCH_SIZE:
        batch_insert(TABLES["FACT_SALES"], records)
        records = []

batch_insert(TABLES["FACT_SALES"], records)


# -----------------------------------
# 8) Done
# -----------------------------------
cursor.close()
conn.close()

print("\n🎉 COMPLETED: All records loaded successfully.\n")