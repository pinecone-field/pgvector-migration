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
2. If you use **Row-Level Security**, classify each policy and decide how to enforce it in Pinecone.
3. Discover which tables/columns hold vectors, and their dimension and metric.
4. Measure the workload (row counts, sizes, query rate).
5. Choose how to lay the data out in Pinecone.
6. Create a Pinecone index.
7. Copy the vectors in (bulk import, recommended — or streaming upsert).
8. Validate that Pinecone returns the same results as pgvector.
9. Keep Pinecone in sync with ongoing inserts, updates, and deletes.
10. Cut traffic over gradually, with a rollback path.

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
| A **Row-Level Security policy** (`CREATE POLICY ... USING`) | A **namespace per principal**, or a **metadata filter injected server-side** | Pinecone has no RLS. The filter is *not* a security boundary — see [Access control](#access-control--map-row-level-security-before-you-choose-a-layout). |

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

> **If your tables use Row-Level Security (RLS), read the next section first.**
> The `pinecone_migration` role above reads rows the way a table *owner* does —
> RLS policies are **not** applied to it — so the migration copies **every** row,
> across every tenant, regardless of who could see it in your app. That is fine
> (you want a complete copy), but it means the access rules RLS enforced in
> Postgres must be re-created on the Pinecone side *before* you choose a layout.

---

## Access control — map Row-Level Security before you choose a layout

Skip this section if you do **not** use Postgres [Row-Level Security][rls]. If you
do, decide how each policy will be enforced in Pinecone now, because the answer
changes your layout (Step 5) and your record mapping (Step 6).

**Why it can't just carry over.** An RLS policy is arbitrary SQL evaluated by the
database on every query (`USING (tenant_id = current_setting('app.tenant_id'))`).
Pinecone has nothing equivalent: a query either targets a **namespace** or carries
a **metadata filter**, and a filter is a flat predicate over scalar fields stored
on each record. It supports only these operators:

| Operator | Meaning |
|---|---|
| `$eq` / `$ne` | equal / not equal |
| `$gt` / `$gte` / `$lt` / `$lte` | numeric comparison |
| `$in` / `$nin` | set membership / exclusion |
| `$and` / `$or` | combine conditions |

No joins, no subqueries, no `current_user`, no functions. So a policy maps cleanly
only when it is *"a predicate over columns I can copy onto the record and compare
against a value the server knows at query time."*

### Step A — list your policies and classify them

```sql
-- Every policy on the tables you intend to migrate:
SELECT schemaname, tablename, policyname, cmd, qual
FROM   pg_policies
WHERE  schemaname = 'public'
ORDER  BY tablename, policyname;

-- And which tables actually have RLS enabled:
SELECT relname, relrowsecurity, relforcerowsecurity
FROM   pg_class
WHERE  relrowsecurity = true;
```

Read the `qual` (the `USING` expression) of each policy and sort it into one of
four buckets:

| Policy shape (`USING ...`) | Maps to | How |
|---|---|---|
| `tenant_id = current_setting('app.tenant_id')` | **Namespace** (preferred) or `$eq` filter | One namespace per tenant, *or* `filter={"tenant_id": {"$eq": <tenant>}}` |
| `visibility IN ('public', current_role)` | `$in` filter | Copy `visibility` to metadata; `filter={"visibility": {"$in": ["public", <role>]}}` |
| `region = ... AND status = 'active'` | `$and` of `$eq` filters | Copy both columns to metadata; combine with `$and` |
| Row carries an ACL list (`allowed_groups text[]`) | list metadata + `$in` | Store `allowed_groups` as a list field; `filter={"allowed_groups": {"$in": [<caller's groups>]}}` |
| `EXISTS (SELECT 1 FROM memberships m WHERE ...)` — **joins another table** | **denormalize**, then filter | Flatten the membership onto each record at copy time, then filter on it (see the gotcha below) |
| References a function, or data not on the row | **does not map** | Keep the access decision in your application; gate which records you query/return |

### Step B — pick the enforcement mechanism

**For plain tenant isolation, prefer a namespace per tenant** over a metadata
filter. The data is physically partitioned, the query targets one namespace chosen
from the authenticated session, and a *forgotten* filter can't silently leak across
tenants — you would have to query the wrong namespace, which is a louder, more
obvious bug. Reach for a metadata filter instead when:

- you have too many / too-small tenants for a namespace each,
- the rule is **finer than a tenant** (per-document ACLs), or
- you legitimately need to query across tenants sometimes.

A common hybrid: **namespace = the hard tenant boundary, metadata filter = the
finer ACL within the tenant.**

### Step C — enforce the filter server-side (the part that actually matters)

RLS is enforced *by the database engine* — a buggy or compromised query still
cannot see another tenant's rows. **A Pinecone metadata filter is the opposite
default: omit it and you get everything.** It recreates the *effect* of RLS, not
its *enforcement*. It is safe only if:

- the tenant/principal value is derived **server-side from the authenticated
  session** — never accepted from the client, where it could be spoofed, and
- the filter is applied **unconditionally** on every query.

```python
# GOOD — the boundary is chosen from the verified session, not the request body.
def search(session, query_vector, k=10):
    tenant = session.tenant_id                       # from your auth layer, trusted
    return index.query(
        vector=query_vector,
        top_k=k,
        namespace=tenant,                            # hard boundary, or:
        filter={"tenant_id": {"$eq": tenant}},       # ...soft boundary — still server-set
    )

# BAD — caller controls the boundary; this is not isolation.
#   filter = request.json["filter"]                  # spoofable
#   index.query(vector=qvec, top_k=k, filter=filter)
```

Wrap this in a single query function that *all* reads go through, so no call site
can forget the filter or the namespace.

> **Gotcha — permission changes can go stale.** If you enforce access with
> *denormalized* metadata (an `allowed_groups` list, a membership flag copied onto
> the record), then a `GRANT`/`REVOKE` happens in a **different table** than the
> vector rows. The sync in [Step 9](#step-9--keep-pinecone-in-sync-with-pgvector-so-it-stays-switch-ready)
> is driven by changes to the *data* tables (its `updated_at` watermark and its
> per-table triggers), so a membership change won't trigger a re-upsert and
> Pinecone keeps serving the **old** ACL. If you go this route, add triggers on the
> permission tables that enqueue the affected record ids, or re-run `reconcile`
> after permission changes.

[rls]: https://www.postgresql.org/docs/current/ddl-rowsecurity.html

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
later without re-copying any data (Step 11).

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

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
    source_tag="pinecone_field:pgvector_migration",
)

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
    values = vector.tolist() if hasattr(vector, "tolist") else vector.to_list()
    return (f"{table}#{row_id}", values, metadata)

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

## Step 9 — Keep Pinecone in sync with pgvector (so it stays switch-ready)

The load in Step 7 is a **snapshot taken at one instant**. From that moment until
you cut over, pgvector keeps changing — new rows, updated vectors, deleted rows.
You must propagate those changes to Pinecone so it remains a faithful copy that is
ready to switch to at any time.

Two facts shape every approach:

- **Upserts are idempotent.** Re-sending a record with the same `id` overwrites it.
  So **inserts and updates are the same operation** — just `upsert`.
- **Deletes are the hard part.** A "what changed since last time?" query *cannot
  see a row that was deleted* — the row is simply gone. So you need either an
  explicit record of deletions, or a periodic full reconciliation.

> **Important — ongoing sync uses the data API, not bulk import.** Bulk import only
> *adds* data and only into **new** namespaces; it cannot update a live namespace
> and cannot delete. So trickle updates use `index.upsert` / `index.delete`
> (below). Reserve bulk import (Step 7a) for the one-time backfill and for periodic
> **full rebuilds** (import a fresh snapshot into a *new* namespace, validate, then
> point your app at it and delete the old one — a "namespace swap").

Pick one of the two strategies below. They map directly onto the two ideas of
"track state in Postgres and anti-join" vs. "keep a change log and apply it."

### Runnable implementation: `sync.py`

This repo includes **`sync.py`**, which implements *both* strategies below and is
tested end-to-end (initial load → insert/update/delete → verify in Pinecone). Edit
the `TABLES` config at the top of the file, set `PG_CONN`, `PINECONE_API_KEY`, and
`PINECONE_INDEX`, then:

```bash
# Strategy B — change-log + triggers (captures inserts, updates, AND deletes)
python sync.py init     --strategy changelog   # one-time: create the outbox + triggers
python sync.py backfill                         # one-time: load existing rows
python sync.py sync     --strategy changelog    # run on a schedule (e.g. cron)

# Strategy A — watermark / anti-join
python sync.py init     --strategy watermark
python sync.py sync     --strategy watermark    # the first run also backfills

python sync.py reconcile                        # id-diff safety net (either strategy)
```

Create the triggers (`init`) **before** the `backfill` so any change made during
the backfill is captured and re-applied (upserts are idempotent). Unlike the
read-only migration, `sync.py` needs **write** access for its bookkeeping (the
ledger/state tables for A, or the change-log table + triggers for B).

The two sections below explain what each strategy does under the hood.

### Strategy A — Watermark / anti-join in Postgres (simplest)

Best when rows are mostly **inserted** (rarely updated/deleted) and you have a
monotonic change signal — a serial `id` and/or an `updated_at timestamptz`.

**Pull new and updated rows** since the last run using a high-water mark you store
somewhere small (a one-row `pinecone_sync_state` table, a file, etc.):

```sql
-- changed rows since the last sync watermark
SELECT id, title, category, embedding
FROM   documents
WHERE  updated_at > :last_synced_at      -- requires an updated_at on writes
ORDER  BY updated_at;
```

Upsert those into Pinecone, then advance the watermark to the max `updated_at` you
just processed.

**Insert-only variant (your `NOT EXISTS` idea):** if there is no `updated_at`, keep
a ledger of ids you have already migrated and pull only the ones that are new:

```sql
-- a tiny ledger table you maintain (one row per migrated record)
CREATE TABLE pinecone_ledger (table_name text, id text, PRIMARY KEY (table_name, id));

-- rows not yet in Pinecone
SELECT s.id, s.title, s.category, s.embedding
FROM   documents s
WHERE  NOT EXISTS (
         SELECT 1 FROM pinecone_ledger l
         WHERE  l.table_name = 'documents' AND l.id = s.id::text);
```

After upserting them, insert their ids into `pinecone_ledger`.

**Deletes need a reconciliation pass** (a watermark/`NOT EXISTS` can't see them).
Anti-join the *other* direction — ids you have migrated that no longer exist in the
source — and delete those from Pinecone:

```sql
-- ids in the ledger whose source row is gone
SELECT l.id
FROM   pinecone_ledger l
WHERE  l.table_name = 'documents'
  AND  NOT EXISTS (SELECT 1 FROM documents s WHERE s.id::text = l.id);
```

```python
stale = [r[0] for r in conn.execute(
    """SELECT l.id FROM pinecone_ledger l
       WHERE l.table_name = 'documents'
         AND NOT EXISTS (SELECT 1 FROM documents s WHERE s.id::text = l.id)""").fetchall()]
for i in range(0, len(stale), 1000):
    index.delete(ids=[f"documents#{rid}" for rid in stale[i:i+1000]], namespace="documents")
# then remove those ids from pinecone_ledger
```

*Trade-off:* no triggers or prod schema changes (beyond the optional ledger), but
updates require an `updated_at`, and deletes require the periodic reconcile.

### Strategy B — Change-log (outbox) table + triggers (captures everything)

Best when **updates and deletes matter**. A trigger records every change to an
outbox table; a worker drains it and applies upserts/deletes to Pinecone. This is
the classic, dependency-free change-data-capture (CDC) pattern.

**1) Create the change-log and triggers** (a one-time DDL on your database — small
per-write overhead; coordinate with your DBA):

```sql
CREATE TABLE pinecone_changelog (
    seq        bigserial PRIMARY KEY,
    table_name text        NOT NULL,
    row_id     text        NOT NULL,
    op         text        NOT NULL CHECK (op IN ('upsert', 'delete')),
    changed_at timestamptz NOT NULL DEFAULT now(),
    processed  boolean     NOT NULL DEFAULT false
);

CREATE OR REPLACE FUNCTION log_pinecone_change() RETURNS trigger AS $$
BEGIN
  IF (TG_OP = 'DELETE') THEN
    INSERT INTO pinecone_changelog(table_name, row_id, op)
      VALUES (TG_TABLE_NAME, OLD.id::text, 'delete');
    RETURN OLD;
  ELSE
    INSERT INTO pinecone_changelog(table_name, row_id, op)
      VALUES (TG_TABLE_NAME, NEW.id::text, 'upsert');
    RETURN NEW;
  END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_pinecone_documents
  AFTER INSERT OR UPDATE OR DELETE ON documents
  FOR EACH ROW EXECUTE FUNCTION log_pinecone_change();
-- repeat the trigger for each migrated table
```

**2) Drain the change-log on a schedule.** Collapse to the latest op per row (so an
insert-then-delete nets to a delete, repeated updates collapse to one upsert),
apply deletes and upserts, then mark the rows processed:

```python
from collections import defaultdict

# Snapshot the frontier first so rows arriving mid-run aren't marked done.
max_seq = conn.execute(
    "SELECT COALESCE(max(seq), 0) FROM pinecone_changelog WHERE NOT processed"
).fetchone()[0]

# Latest op per (table, row) up to the frontier.
changes = conn.execute("""
    SELECT DISTINCT ON (table_name, row_id) table_name, row_id, op
    FROM   pinecone_changelog
    WHERE  NOT processed AND seq <= %s
    ORDER  BY table_name, row_id, seq DESC
""", (max_seq,)).fetchall()

upserts, deletes = defaultdict(list), defaultdict(list)
for table, row_id, op in changes:
    (upserts if op == "upsert" else deletes)[table].append(row_id)

# Deletes: just send the ids.
for table, ids in deletes.items():
    for i in range(0, len(ids), 1000):
        index.delete(ids=[f"{table}#{r}" for r in ids[i:i+1000]], namespace=table)

# Upserts: re-read current row data from the source, then upsert (idempotent).
for table, ids in upserts.items():
    cfg = TABLES[table]
    cols = ", ".join([cfg["id"]] + cfg["metadata"] + [cfg["vector"]])
    fresh = conn.execute(
        f"SELECT {cols} FROM {table} WHERE {cfg['id']}::text = ANY(%s)", (ids,)
    ).fetchall()
    records = [build_record(table, r, cfg) for r in fresh]
    for i in range(0, len(records), 200):
        index.upsert(vectors=records[i:i+200], namespace=table)

conn.execute(
    "UPDATE pinecone_changelog SET processed = true WHERE NOT processed AND seq <= %s",
    (max_seq,))
```

*The Parquet variant of this:* for a large batch of upserts you can write the
changed rows to Parquet and **rebuild** into a new namespace (then swap), but
deletes always go through `index.delete` — they can't be expressed in an import
file. For high write volumes, replace the triggers with Postgres **logical
replication / Debezium** to stream changes from the WAL instead.

### Reconciliation safety net (recommended for either strategy)

Periodically prove the two stores agree by diffing ids. `index.list` pages through
every id in a namespace:

```python
pg_ids = {f"documents#{r[0]}" for r in conn.execute("SELECT id FROM documents")}
pc_ids = set()
for id_batch in index.list(namespace="documents"):   # yields pages of ListItem objects
    pc_ids.update(i.id for i in id_batch)             # i.id is the record id string

missing_in_pinecone = pg_ids - pc_ids   # -> upsert these
stale_in_pinecone   = pc_ids - pg_ids   # -> index.delete these
print(len(missing_in_pinecone), "to add;", len(stale_in_pinecone), "to delete")
```

Run this nightly (or before cutover). In steady state both sets should be empty;
anything else flags a sync gap to fix before you switch.

## Step 10 — Cut over safely (keep pgvector running)

1. **Keep both in sync.** Make sure the continuous sync from **Step 9** is running,
   so Pinecone reflects every insert, update, and delete from pgvector. Otherwise it
   drifts out of date and the cutover is unsafe.
2. **Shadow reads.** Send a copy of some live read traffic to Pinecone *without*
   using its results yet. Compare results and latency to pgvector. Fix any gaps.
3. **Canary.** Route a small percentage of real reads (say 1%, then 10%) to
   Pinecone. Watch quality and latency. Increase only when satisfied.
4. **Full cutover.** Once a meaningful share runs on Pinecone with good results,
   make Pinecone the primary read path. **Keep pgvector running as a fallback.**
5. **Rollback is trivial.** If anything looks wrong at any stage, point reads back
   to pgvector — it never stopped serving and is still authoritative.

---

## Step 11 — (Optional) tune capacity later

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
| Audit RLS policies | `SELECT * FROM pg_policies WHERE schemaname='public';` (Access control) |
| Find vector columns | `pg_attribute` query in Step 2 |
| Count vectors | `SELECT count(*) FROM t;` |
| Create index | `pc.create_index(name, dimension, metric, spec=ServerlessSpec(...))` |
| Get a data handle | `index = pc.Index(name)` |
| Bulk import | `index.start_import(uri=..., integration_id=..., error_mode="CONTINUE")` |
| Stream upsert | `index.upsert(vectors=[(id, values, metadata), ...], namespace=ns)` |
| Delete records | `index.delete(ids=[...], namespace=ns)` |
| List ids (reconcile) | `index.list(namespace=ns)` |
| Search | `index.query(vector=q, top_k=k, namespace=ns, filter={...})` |
| Check counts | `index.describe_index_stats()` |
| Go dedicated | `pc.configure_index(name, read_capacity={...})` |

For an end-to-end runnable example of all of the above, open
**`migration_walkthrough.ipynb`**.
