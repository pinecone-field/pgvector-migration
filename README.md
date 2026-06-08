# Migrating from pgvector to Pinecone — a step-by-step manual

This manual walks you through moving a vector-search workload from **pgvector**
(the `vector` extension for PostgreSQL) to **Pinecone** (a managed vector
database). It is written to be followed top to bottom, even if you have never used
Pinecone before.

**The guiding principle is safety.** Your pgvector database stays the source of
truth and keeps serving traffic the entire time. Every step against Postgres is
**read-only** — we never modify, lock, or drop anything in your database. You only
cut traffic over to Pinecone after you have proven it returns the same results,
and you can roll back instantly by pointing reads back at pgvector.

> Want to see it run first? `migration_walkthrough.ipynb` in this repo does every
> step below against a throwaway Dockerized pgvector, so you can watch the whole
> flow end-to-end before touching your real database.

### What you will do

1. Set up a workstation and connect to your database **read-only**.
2. Discover which tables/columns hold vectors, and their dimension and metric.
3. Measure the workload (row counts, sizes, query rate).
4. Choose how to lay the data out in Pinecone.
5. Create a Pinecone index.
6. Copy the vectors in (bulk import, recommended — or streaming upsert).
7. Validate that Pinecone returns the same results as pgvector.
8. Cut traffic over gradually, with a rollback path.

---

## Background: how pgvector concepts map to Pinecone

If you are new to Pinecone, read this once. It makes the rest obvious.

| In pgvector (Postgres) | In Pinecone | Notes |
|---|---|---|
| A **table** with a `vector(N)` column | An **index** (and optionally **namespaces** inside it) | `N` is the *dimension* and must match exactly. |
| A **row** (`id`, vector, other columns) | A **record**: `id` (string), `values` (the vector), `metadata` (the other columns) | |
| Distance operator `<=>`, `<#>`, `<->` | The index **metric** (`cosine`, `dotproduct`, `euclidean`) | Must match how you currently query. |
| `ORDER BY embedding <=> q LIMIT k` | `index.query(vector=q, top_k=k)` | Pinecone returns the nearest `k`. |
| `WHERE category = 'x'` | `filter={"category": {"$eq": "x"}}` | Filtering happens on metadata. |
| The HNSW/IVFFlat index **you build and tune** | Managed automatically | Nothing to build or tune. |
| Read replicas / sharding **you operate** | **On-demand** autoscaling, or **dedicated read nodes** | You pick; no servers to run. |
| A separate table or schema **per tenant** | **One namespace per tenant** in a single index | Namespaces partition data inside one index. |

Two definitions you will see throughout:

- **Namespace** — a named partition *inside* one index. A query runs against one
  namespace, so namespaces keep tenants/datasets isolated and queries cheap.
- **Capacity mode** — how Pinecone serves reads:
  - **On-demand** (the default): shared infrastructure, pay per query. Best for
    low/spiky/unknown traffic.
  - **Dedicated read nodes**: hardware reserved just for your index, billed by the
    hour, for predictable low latency at sustained high query rates.

---

## Prerequisites

You need:

1. **A Pinecone account and API key.** Create one at
   [app.pinecone.io](https://app.pinecone.io) → *API Keys* → *Create API key*. It
   looks like `pcsk_...`. Treat it like a password.
2. **Network access to your Postgres** from wherever you run these steps (your
   laptop, a bastion, or a VM in the same network).
3. **Python 3.9+** installed.

Set up an isolated Python environment and install the libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # pinecone, psycopg[binary], pgvector, numpy, pyarrow
```

Provide your secrets as environment variables (never hard-code them in scripts):

```bash
export PINECONE_API_KEY="pcsk_xxxxxxxx..."
export PG_CONN="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
```

---

## Step 1 — Connect to your database, read-only

**Safety first.** Create (or ask your DBA for) a Postgres user that can only read.
This guarantees the migration cannot change your data:

```sql
-- Run once, as an admin, in your database:
CREATE USER pinecone_migration WITH PASSWORD 'choose-a-strong-password';
GRANT CONNECT ON DATABASE yourdb TO pinecone_migration;
GRANT USAGE ON SCHEMA public TO pinecone_migration;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO pinecone_migration;
```

Put that user in your `PG_CONN`. Now confirm you can connect from Python:

```python
import os, psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(os.environ["PG_CONN"])
register_vector(conn)        # makes psycopg return `vector` columns as lists/arrays
print(conn.execute("SELECT version();").fetchone()[0])
```

If that prints your Postgres version, you are connected. Keep this `conn` open;
later steps reuse it.

---

## Step 2 — Discover your vector data

You need three facts about **every** table you intend to migrate: the **vector
column**, its **dimension**, and the **metric** you query with.

**Find every vector column and its dimension** (the type prints as `vector(768)`):

```sql
SELECT c.relname AS table_name,
       a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS column_type   -- e.g. "vector(768)"
FROM   pg_attribute a
JOIN   pg_class     c ON c.oid = a.attrelid
JOIN   pg_type      t ON t.oid = a.atttypid
WHERE  t.typname = 'vector'
  AND  a.attnum > 0 AND NOT a.attisdropped
  AND  c.relkind = 'r'                                        -- ordinary tables
ORDER  BY table_name;
```

**Find the metric** by inspecting the vector index on each table. The operator
class in the index definition tells you the distance function:

```sql
SELECT indexname, indexdef
FROM   pg_indexes
WHERE  tablename = 'your_table';
```

| If `indexdef` contains… | Your metric is | Pinecone `metric=` |
|---|---|---|
| `vector_cosine_ops` (operator `<=>`) | cosine | `"cosine"` |
| `vector_ip_ops` (operator `<#>`) | inner / dot product | `"dotproduct"` |
| `vector_l2_ops` (operator `<->`) | Euclidean (L2) | `"euclidean"` |

If there is no vector index, use whichever operator your application's query uses.
**The metric must match** — picking the wrong one silently returns wrong results.

Write down, per table: `id` column, vector column, dimension, metric, and the
other columns you want to keep for filtering (these become **metadata**).

---

## Step 3 — Measure the workload

These numbers tell you which Pinecone layout to choose, and give you a target to
validate against later. All read-only.

**Row count and on-disk size, per table:**

```sql
SELECT count(*) FROM your_table;                                   -- number of vectors

SELECT pg_size_pretty(pg_total_relation_size('your_table'));       -- table size

SELECT pg_size_pretty(COALESCE(sum(pg_relation_size(indexrelid)), 0))
FROM   pg_index WHERE indrelid = 'your_table'::regclass;           -- index size
```

**Average query rate (QPS).** Two ways — use whichever you can:

- *From the application:* divide your queries-per-day by 86,400, and note your peak.
- *From Postgres* (needs the `pg_stat_statements` extension enabled): reset the
  counter, let normal traffic run for a known window (say 10 minutes), then read
  how many vector queries ran:

  ```sql
  SELECT pg_stat_statements_reset();
  -- ... wait a known number of seconds (your measurement window) ...
  SELECT sum(calls) AS total_queries
  FROM   pg_stat_statements
  WHERE  query ILIKE '%<=>%' OR query ILIKE '%<#>%' OR query ILIKE '%<->%';
  ```

  `QPS = total_queries / window_seconds`. Note both the **average** and the **peak**.

---

## Step 4 — Choose your Pinecone layout

Two decisions.

**(a) Capacity mode:**

| Your traffic | Choose |
|---|---|
| Low, spiky, or unknown; cost-sensitive | **On-demand** (start here — you can switch later) |
| Sustained high QPS needing predictable latency | **Dedicated read nodes** |

When unsure, **start on-demand.** You can move an index to dedicated read nodes
later without re-copying any data (Step 10).

**(b) Namespace layout** — how your tables map into indexes/namespaces:

| Your situation | Layout | Why |
|---|---|---|
| One table, or each table queried independently | **One namespace per table** in a single index | Queries scan only that namespace — cheapest and fastest. |
| You query across all the data at once, or want one endpoint | **One shared namespace**, add a `source_table` metadata field, filter at query time | Single place to search; filter narrows to a table. |
| Tables differ in dimension or metric | **One index per table** | Dimension and metric are per-index and must match. |

> Tables with **different dimensions or metrics cannot share an index** — give
> each its own index.

---

## Step 5 — Decide your record mapping

Every Postgres row becomes a Pinecone record with exactly three parts:

- **`id`** — a unique **string**. Prefix it with the table name so ids never
  collide and you can tell where a record came from: `f"{table}#{row_id}"`
  (e.g. `"documents#4021"`).
- **`values`** — the vector, as a list of floats.
- **`metadata`** — the other columns you want to filter on. Rules that matter:
  - Values must be **scalars only**: string, number, boolean, or list of strings.
    No nested objects/dicts.
  - Convert Postgres `numeric`/`Decimal` to `float`.
  - Skip `NULL`s (don't include the key).
  - Do **not** create a field literally named `metadata`.
  - If you merge tables into one namespace, add `"source_table": "<table>"`.

---

## Step 6 — Create the Pinecone index

Pick a region close to your application. Dimension and metric come from Step 2.

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "my-app"        # lowercase letters, numbers, hyphens
DIMENSION  = 768             # from Step 2 — must match your vectors exactly
METRIC     = "cosine"        # from Step 2

# On-demand (recommended starting point)
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
# create_index waits until the index is ready before returning.
```

To create a **dedicated read nodes** index instead (only if Step 4 said so):

```python
pc.create_index(
    name=INDEX_NAME, dimension=DIMENSION, metric=METRIC,
    spec=ServerlessSpec(cloud="aws", region="us-east-1",
        read_capacity={"mode": "Dedicated",
            "dedicated": {"node_type": "b1", "scaling": "Manual",
                          "manual": {"shards": 1, "replicas": 1}}}),
)
```

> **Note:** the SDK object is `pc.Index("name")` (capital **I**) to get a handle for
> reading/writing data, while index *management* calls (`create_index`,
> `has_index`, `configure_index`) are on `pc` directly.

---

## Step 7 — Copy the vectors into Pinecone

> **Do a pilot first.** Before copying everything, migrate one small table (or add
> `LIMIT 1000` to the query) and run Step 8 on it. Confirm the counts and a few
> queries match. Only then run the full copy. This catches a wrong dimension,
> metric, or mapping while it is cheap to fix.

There are two methods. **Bulk import (7a) is recommended for production.** Use
streaming upsert (7b) only for small datasets or quick tests.

First, a small config block describing your tables (fill in from Steps 2 & 5):

```python
# table -> which columns are the id, the vector, and the metadata to keep
TABLES = {
    "documents": {"id": "id", "vector": "embedding", "metadata": ["title", "category"]},
    "products":  {"id": "id", "vector": "embedding", "metadata": ["name", "price"]},
    # ...add each table...
}
```

And a helper that turns one row into the three Pinecone parts (used by both methods):

```python
import json
from decimal import Decimal

def build_record(table, row, cfg):
    """row is (id, *metadata_cols, vector) in the SELECT order below."""
    row_id      = row[0]
    vector      = row[-1]                          # numpy array via register_vector
    meta_values = row[1:-1]
    metadata = {"source_table": table}             # provenance; safe to keep always
    for name, value in zip(cfg["metadata"], meta_values):
        if value is None:
            continue
        if isinstance(value, Decimal):
            value = float(value)                   # JSON/metadata can't hold Decimal
        metadata[name] = value
    return (f"{table}#{row_id}", vector.tolist(), metadata)

def select_sql(table, cfg):
    cols = ", ".join([cfg["id"]] + cfg["metadata"] + [cfg["vector"]])
    return f"SELECT {cols} FROM {table};"
```

### 7a — Bulk import via Parquet (recommended)

Bulk import is asynchronous, runs server-side, costs far less per vector, and
scales to billions of records. You write **Parquet** files to cloud object storage
(Amazon S3, Google Cloud Storage, or Azure Blob), then tell Pinecone to import
them.

**(1) Export each table to a Parquet file.** The file must have exactly these
columns and types:

| column | type | content |
|---|---|---|
| `id` | `string` | the record id |
| `values` | `list<float>` | the vector |
| `metadata` | `string` | the metadata as a **JSON string** (or `null`) |

```python
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("values", pa.list_(pa.float32())),
    ("metadata", pa.string()),       # JSON-encoded string
])

def export_table_to_parquet(table, cfg, out_path):
    ids, values, metas = [], [], []
    # A server-side cursor streams rows without loading the whole table into RAM.
    # It must run inside a transaction:
    with conn.transaction():
        with conn.cursor(name=f"stream_{table}") as cur:
            cur.execute(select_sql(table, cfg))
            for row in cur:
                rid, vec, meta = build_record(table, row, cfg)
                ids.append(rid)
                values.append(vec)
                metas.append(json.dumps(meta))      # metadata MUST be a JSON string
    table_arrow = pa.table({"id": ids, "values": values, "metadata": metas},
                           schema=PARQUET_SCHEMA)
    pq.write_table(table_arrow, out_path)
    print(f"wrote {len(ids)} rows -> {out_path}")
```

**(2) Lay the files out by namespace.** Pinecone imports a *directory tree* where
**each subdirectory is one namespace**. The default namespace uses the literal
name `__default__`. The target namespaces **must not already exist** in the index.

```
s3://your-bucket/import-root/
  documents/0.parquet          # namespace "documents"
  products/0.parquet           # namespace "products"
```

For *one shared namespace* instead, put every table's file (named `0.parquet`,
`1.parquet`, …) under a single subdirectory, e.g. `all/`.

```python
# Example: one namespace per table, written locally then uploaded to your bucket.
import os
os.makedirs("export", exist_ok=True)
for table, cfg in TABLES.items():
    os.makedirs(f"export/{table}", exist_ok=True)
    export_table_to_parquet(table, cfg, f"export/{table}/0.parquet")
# Then upload the whole ./export/ tree to s3://your-bucket/import-root/
```

**(3) Start the import.** Point `uri` at the import **root** (the directory that
contains the namespace subdirectories), not at a single file. Private buckets need
a one-time [storage integration](https://docs.pinecone.io/guides/operations/integrations/manage-storage-integrations)
(create it in the Pinecone console); public buckets do not.

```python
index = pc.Index(INDEX_NAME)
op = index.start_import(
    uri="s3://your-bucket/import-root",
    integration_id="your-storage-integration-id",   # omit for public buckets
    error_mode="CONTINUE",                           # skip bad rows; or "ABORT"
)
print("import id:", op.id)

# Poll until it finishes (imports run in the background):
import time
while True:
    status = index.describe_import(op.id)
    print(status.status, getattr(status, "percent_complete", ""))
    if status.status in ("Completed", "Failed", "Cancelled"):
        break
    time.sleep(10)
```

**Import limits:** 10 GB per file, 100,000 files per import, 1 TB total
(on-demand). If one namespace would exceed 10 GB, split it into `0.parquet`,
`1.parquet`, … in the same subdirectory.

### 7b — Streaming upsert (small datasets / quick tests)

This writes directly from Python without object storage. Each `upsert` call is
limited by **two** things — at most **1000 vectors** *and* at most **2 MB** per
request. For typical dimensions the 2 MB size limit is what bites first (a 768-dim
float vector is ~3 KB, so ~1000 of them ≈ 3 MB, which is rejected). Use a batch of
about **200** and reduce it further if your dimension is larger.

```python
index = pc.Index(INDEX_NAME)
BATCH = 200

def upsert_table(table, cfg, namespace):
    total = 0
    with conn.transaction():                          # server-side cursor needs a txn
        with conn.cursor(name=f"stream_{table}") as cur:
            cur.execute(select_sql(table, cfg))
            while True:
                rows = cur.fetchmany(BATCH)
                if not rows:
                    break
                records = [build_record(table, r, cfg) for r in rows]
                index.upsert(vectors=records, namespace=namespace)
                total += len(records)
    print(f"upserted {total} vectors from {table} into namespace '{namespace}'")

# One namespace per table:
for table, cfg in TABLES.items():
    upsert_table(table, cfg, namespace=table)

# Or one shared namespace:
# for table, cfg in TABLES.items():
#     upsert_table(table, cfg, namespace="all")
```

---

## Step 8 — Validate (do not skip)

**(1) Counts match.** Pinecone's per-namespace count must equal your pgvector
`count(*)`. Allow a few seconds after writing — stats are eventually consistent.

```python
import time
time.sleep(5)
stats = index.describe_index_stats()
print("total in Pinecone:", stats.total_vector_count)
for ns, info in stats.namespaces.items():
    print(f"  namespace '{ns}': {info.vector_count}")
# Compare each against:  SELECT count(*) FROM <table>;
```

**(2) Results match.** Take a few real query vectors, search both systems, and
confirm the top results are the same ids in the same order.

```python
# pgvector (cosine example: smaller distance = closer)
def pg_topk(table, cfg, qvec, k=10):
    sql = f"SELECT {cfg['id']} FROM {table} ORDER BY {cfg['vector']} <=> %s LIMIT %s"
    with conn.cursor() as cur:
        cur.execute(sql, (qvec, k))
        return [f"{table}#{r[0]}" for r in cur.fetchall()]

# Pinecone
def pc_topk(qvec, k=10, namespace="documents"):
    res = index.query(vector=list(qvec), top_k=k, namespace=namespace)
    return [m.id for m in res.matches]

# For a handful of sample vectors, pg_topk(...) and pc_topk(...) should agree.
```

Small ordering differences on near-ties are normal (Pinecone uses approximate
search). Large disagreements usually mean a **wrong metric** (Step 2) or vectors
that were normalized differently — fix before cutting over.

---

## Step 9 — Cut over safely (keep pgvector running)

Your pgvector database stays in charge until Pinecone has earned the traffic. Do
this gradually:

1. **Keep both in sync.** The import in Step 7 is a one-time snapshot. From now on,
   have your application **write new and updated vectors to *both* pgvector and
   Pinecone** (dual-write), or re-run the import on a schedule. Otherwise Pinecone
   drifts out of date.
2. **Shadow reads.** Send a copy of some live read traffic to Pinecone *without*
   using its results yet. Compare results and latency to pgvector. Fix any gaps.
3. **Canary.** Route a small percentage of real reads (say 1%, then 10%) to
   Pinecone. Watch quality and latency. Increase only when satisfied.
4. **Full cutover.** Once a meaningful share runs on Pinecone with good results,
   make Pinecone the primary read path. **Keep pgvector running as a fallback.**
5. **Rollback is trivial.** If anything looks wrong at any stage, point reads back
   to pgvector — it never stopped serving and is still authoritative.

**Decommission pgvector only much later** — after a sustained period with Pinecone
as primary, no incidents, and no reason to roll back. There is no rush.

---

## Step 10 — (Optional) tune capacity later

Once you know your real query rate, you can move an on-demand index to **dedicated
read nodes** *in place* — no data is re-copied:

```python
pc.configure_index(INDEX_NAME, read_capacity={
    "mode": "Dedicated",
    "dedicated": {"node_type": "b1", "scaling": "Manual",
                  "manual": {"shards": 1, "replicas": 1}},
})
```

Throughput scales roughly linearly with replicas. From a load test, size them as
`replicas ≈ ceil(target_QPS / QPS_per_replica)`. Dedicated read nodes bill by the
hour for as long as they exist, so delete any index you created only for testing:
`pc.delete_index("test-index")`.

---

## Troubleshooting (common errors and fixes)

| Symptom | Cause | Fix |
|---|---|---|
| `Request size 3MB exceeds the maximum supported size of 2MB` | Upsert batch too large in bytes | Lower the batch (Step 7b uses 200); the limit is 2 MB *and* 1000 vectors. |
| `DECLARE CURSOR can only be used in transaction blocks` | Server-side (named) cursor used outside a transaction | Wrap the read in `with conn.transaction():` (shown in Step 7). |
| `'Pinecone' object has no attribute 'index'` | Wrong method name | Use `pc.Index("name")` — capital **I**. |
| Dimension mismatch error on upsert/import | Index dimension ≠ vector length | Recreate the index with the dimension from Step 2. |
| Query results clearly wrong vs. pgvector | Wrong metric | Match the metric to your pgvector operator class (Step 2). |
| `vector` column comes back as a string | `register_vector(conn)` not called | Call it right after connecting (Step 1). |
| Bulk import rejects the namespace | Namespace already exists in the index | Import only into **new** namespaces, or use streaming upsert. |
| Metadata error on upsert | Nested object, `Decimal`, or a field named `metadata` | Keep metadata flat and scalar; cast `Decimal`→`float`; rename the field. |

---

## Quick reference

| Task | Command |
|---|---|
| Find vector columns | `pg_attribute` query in Step 2 |
| Count vectors | `SELECT count(*) FROM t;` |
| Create index | `pc.create_index(name, dimension, metric, spec=ServerlessSpec(...))` |
| Get a data handle | `index = pc.Index(name)` |
| Bulk import | `index.start_import(uri=..., integration_id=..., error_mode="CONTINUE")` |
| Stream upsert | `index.upsert(vectors=[(id, values, metadata), ...], namespace=ns)` |
| Search | `index.query(vector=q, top_k=k, namespace=ns, filter={...})` |
| Check counts | `index.describe_index_stats()` |
| Go dedicated | `pc.configure_index(name, read_capacity={...})` |

For an end-to-end runnable example of all of the above, open
**`migration_walkthrough.ipynb`**.
