# pgvector → Pinecone Migration Manual

A concise, repeatable recipe for moving a vector workload from **pgvector**
(Postgres) to **Pinecone serverless**. The companion notebook
`migration_walkthrough.ipynb` is a fully worked example of every step below.

---

## 0. Prerequisites

```bash
pip install -r requirements.txt          # pinecone, psycopg[binary], pgvector, numpy
docker compose up -d                     # local pgvector on localhost:5432 (for the demo)
export PINECONE_API_KEY=...              # your Pinecone API key
```

For a real migration, point `PG_CONN` at your existing Postgres instead of the
Docker one.

---

## 1. Inventory the source

Record, for **each table** that holds vectors:

| What | How |
|------|-----|
| Vector **dimension** & metric | from the column type `vector(N)` and the operator you query with (`<=>` = cosine, `<#>` = dot product, `<->` = L2) |
| **# vectors** | `SELECT count(*) FROM <table>;` |
| **Table size** | `SELECT pg_size_pretty(pg_total_relation_size('<table>'));` |
| **Index size** | `SELECT pg_size_pretty(sum(pg_relation_size(indexrelid))) FROM pg_index WHERE indrelid = '<table>'::regclass;` |
| Metadata columns | every non-vector column you'll want to filter on later |

The dimension and metric **must match** on the Pinecone side.

---

## 2. Extract metrics → pick a target shape

Measure your query rate, then choose capacity mode and namespace layout.

**Average QPS** — two ways:
- **Client-side:** time a representative batch of queries → `queries / elapsed_seconds`.
- **Server-side:** `SELECT pg_stat_statements_reset();`, run your workload for a known
  window, then `SELECT sum(calls) FROM pg_stat_statements WHERE query ILIKE '%<=>%';`
  and divide by the window length. (Requires `pg_stat_statements` in
  `shared_preload_libraries`.)

**Choose capacity mode:**

| Your situation | Pinecone capacity |
|----------------|-------------------|
| Spiky / low / unknown QPS, cost-sensitive | **On-demand** (default) — pay per read unit |
| Sustained high QPS needing predictable low latency | **Dedicated read nodes** — fixed hourly cost, all data kept warm |

**Choose namespace layout:**

| Data pattern | Layout |
|--------------|--------|
| Each table queried on its own | One index, **one namespace per table** (cheapest/fastest — scans only that namespace) |
| Tables queried together, or you want one endpoint | One index, **one shared namespace** + a `source_table` metadata field; filter at query time |
| Truly independent products/scale | **One index per table** |

---

## 3. Define the mapping

Decide how a Postgres row becomes a Pinecone record `(id, values, metadata)`:

- **ID:** prefix with the table to avoid collisions and keep provenance —
  `f"{table}#{primary_key}"`.
- **values:** the `vector` column as a `list[float]`.
- **metadata:** the non-vector columns, **scalars only** (str / number / bool).
  Add `source_table` if you're merging tables. Never nest objects; never name a
  field `metadata`.

---

## 4. Create the target index

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# On-demand (default capacity)
pc.create_index(name="my-index", dimension=N, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))

# Dedicated read nodes (provisioned capacity)
pc.create_index(name="my-index", dimension=N, metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1",
        read_capacity={"mode": "Dedicated",
            "dedicated": {"node_type": "b1", "scaling": "Manual",
                          "manual": {"shards": 1, "replicas": 1}}}))
```

`create_index` polls until the index is ready by default.

---

## 5. Load the vectors into Pinecone

### 5a. Bulk import via Parquet — *recommended*

For a production migration, **export to Parquet and bulk-import** rather than
streaming `upsert`s: it is asynchronous, runs server-side, costs far less per
vector, and scales to millions of records.

**Export** — write one Parquet file per namespace with this exact schema, then
upload the tree to object storage (S3 / GCS / Azure Blob):

| column | type | notes |
|--------|------|-------|
| `id` | `string` | record id (e.g. `f"{table}#{pk}"`) |
| `values` | `list<float>` | the dense vector |
| `metadata` | `string` | optional, **JSON-encoded** (`json.dumps(...)`) or `null` |

Lay it out with **one subdirectory per namespace** under the import root; the
default namespace uses the literal name `__default__`, and target namespaces
**must not already exist** in the index:

```
s3://my-bucket/import-root/
  __default__/0.parquet          # default namespace
  documents/0.parquet            # or one directory per namespace
  products/0.parquet
```

```python
import json, pyarrow as pa, pyarrow.parquet as pq

schema = pa.schema([("id", pa.string()),
                    ("values", pa.list_(pa.float32())),
                    ("metadata", pa.string())])

def export_table(table, ns_dir, file_idx=0):
    ids, values, metas = [], [], []
    with conn.transaction():                                 # server-side cursor needs a txn
        with conn.cursor(name=f"stream_{table}") as scur:
            scur.execute(f"SELECT id, {', '.join(META_COLS[table])}, embedding FROM {table};")
            for r in scur:
                ids.append(f"{table}#{r[0]}")
                values.append(r[-1].tolist())
                metas.append(json.dumps({"source_table": table,
                                         **dict(zip(META_COLS[table], r[1:-1]))}, default=float))
    pq.write_table(pa.table({"id": ids, "values": values, "metadata": metas}, schema=schema),
                   f"{ns_dir}/{file_idx}.parquet")
```

**Import** — point `start_import` at the import *root* (not a file). Private
buckets need a [storage integration](https://docs.pinecone.io/guides/operations/integrations/manage-storage-integrations); public buckets don't.

```python
index = pc.Index("my-index")
op = index.start_import(uri="s3://my-bucket/import-root",
                        integration_id="<id>",      # omit for public buckets
                        error_mode="CONTINUE")
index.describe_import(op.id)                          # poll until Completed
```

**Limits:** 10 GB per file, 100k files per import, 1 TB total (on-demand). Shard a
large namespace into `0.parquet`, `1.parquet`, …

### 5b. Streaming upsert — for small/medium datasets

Stream from Postgres with a **server-side cursor** and upsert in **size-aware
batches**. Each `upsert()` is bounded by **two** limits — ≤1000 vectors **and**
≤2 MB per request. A 768-dim float vector is ~3 KB, so the 2 MB limit binds first:
use ~200 per batch and scale inversely with your dimension.

```python
index = pc.Index("my-index")

def migrate_table(table, namespace=""):
    cols = ", ".join(["id"] + META_COLS[table] + ["embedding"])
    with conn.transaction():                                 # server-side cursor needs a txn
        with conn.cursor(name=f"stream_{table}") as scur:
            scur.execute(f"SELECT {cols} FROM {table};")
            while True:
                rows = scur.fetchmany(200)                   # size-aware batch
                if not rows:
                    break
                records = [(f"{table}#{r[0]}", r[-1].tolist(),
                            {"source_table": table, **dict(zip(META_COLS[table], r[1:-1]))})
                           for r in rows]
                index.upsert(vectors=records, namespace=namespace)
```

- **Per-table namespaces:** call `migrate_table(t, namespace=t)` for each table.
- **Combined namespace:** call `migrate_table(t, namespace="all")` for each table.

---

## 6. Validate

```python
stats = index.describe_index_stats()
# total / per-namespace counts must equal the pgvector count(*)
print(stats.total_vector_count, stats.namespaces)
```

Then spot-check parity: run the same query vector against pgvector
(`ORDER BY embedding <=> %s LIMIT k`) and Pinecone (`index.query(...)`) and confirm
the top results line up. For combined namespaces, verify the metadata filter:
`filter={"source_table": {"$eq": "products"}}`.

---

## 7. Cut over gradually (keep pgvector running)

Your pgvector instance is production — **keep it as the source of truth and serving
backend until Pinecone is proven.** Don't tear it down as part of the migration.

1. **Dual-write** new/updated vectors to *both* pgvector and Pinecone so the index
   stays current while you evaluate. (The bulk import in §5 is the one-time
   backfill; ongoing changes need dual-write or periodic re-import.)
2. **Shadow / canary reads:** run a fraction of read traffic against Pinecone and
   compare results and latency to pgvector before shifting more.
3. **Shift reads** to Pinecone gradually once parity and latency meet your bar.
   Keep pgvector serving as a fallback.
4. **Decommission pgvector only later** — after a sustained confidence window with
   Pinecone as primary and no need to roll back. There is no rush to stop it.
5. If you started on-demand and now know your QPS, migrate to **dedicated read
   nodes** in place — no re-import needed:

   ```python
   pc.configure_index("my-index", read_capacity={
       "mode": "Dedicated",
       "dedicated": {"node_type": "b1", "scaling": "Manual",
                     "manual": {"shards": 1, "replicas": 1}}})
   ```

   Size replicas from your load test: `replicas ≈ ceil(target_QPS / QPS_per_replica)`.

> **Cost note:** dedicated-read-node indexes bill hourly — delete any you created
> only for testing. (`docker compose down -v` applies to the *demo* Postgres in this
> repo, **not** your production pgvector.)

---

## pgvector → Pinecone cheat sheet

| pgvector / Postgres | Pinecone |
|---------------------|----------|
| Table of `vector(N)` rows | Index (`dimension=N`) — optionally split by namespace |
| Distance operator `<=>` / `<#>` / `<->` | `metric="cosine"` / `"dotproduct"` / `"euclidean"` |
| `ORDER BY embedding <=> q LIMIT k` | `index.query(vector=q, top_k=k)` |
| Non-vector columns | Record `metadata` (scalars) |
| `WHERE category = 'x'` | `filter={"category": {"$eq": "x"}}` |
| HNSW/IVFFlat index you build & tune | Managed ANN — nothing to build |
| Manual sharding / read replicas | On-demand autoscaling **or** dedicated read nodes |
| Per-tenant table/schema | One namespace per tenant |
