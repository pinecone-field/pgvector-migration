#!/usr/bin/env python3
"""sync.py — keep a Pinecone index in sync with pgvector after the initial load.

The initial migration (see migration_walkthrough.ipynb / README Step 7) is a
one-time snapshot. This script keeps Pinecone current with ongoing inserts,
updates, and deletes in pgvector, so the index stays ready to switch to.

Two strategies — pick with --strategy:

  watermark  (A)  Track migrated ids in a ledger and the latest synced
                  `updated_at` per table. Each run upserts rows that are new
                  (NOT EXISTS in the ledger) or changed (updated_at > watermark),
                  and deletes rows whose source row is gone (ledger anti-join).
                  Lightest touch on the source DB; updates require an updated_at
                  column that your writes maintain.

  changelog  (B)  An outbox table + AFTER INSERT/UPDATE/DELETE triggers record
                  every change. Each run drains the log and applies upserts and
                  deletes. Captures inserts, updates, AND deletes exactly, with no
                  updated_at column required. Needs DDL (a table + triggers).

Commands:
  init       create the bookkeeping (ledger+state for A; changelog+triggers for B)
  backfill   one-time full load of all current rows into Pinecone
  sync       apply incremental changes since the last run
  reconcile  id-diff safety net: make Pinecone exactly match pgvector

Config: edit TABLES below. Set env PG_CONN, PINECONE_API_KEY, PINECONE_INDEX.
Each table maps to a namespace of the same name (one namespace per table).

NOTE: ongoing sync uses the upsert/delete data API on purpose. Bulk import only
adds data and only into NEW namespaces, so it cannot serve trickle updates or
deletes — reserve it for the initial backfill and periodic full rebuilds.
"""
import argparse
import os
from collections import defaultdict
from decimal import Decimal

import psycopg
from pgvector.psycopg import register_vector
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Configuration — EDIT THIS to match your schema.
#
# The entries below are the DEMO shape: they match the tables created by
# migration_walkthrough.ipynb (documents/products/images), so sync.py runs
# against the demo out of the box. For a real migration, replace them with your
# own tables. Each entry:
#   id:         primary-key column (used to build the Pinecone record id)
#   vector:     the vector(N) column
#   metadata:   non-vector columns to carry into Pinecone metadata
#   updated_at: timestamp column bumped on every write (strategy A only; or None)
#
# The demo tables have no updated_at column, so it is None here: the watermark
# strategy then syncs new rows via the ledger and removals via the anti-join, but
# not in-place updates. Add an updated_at column (and use changelog for the demo,
# which captures updates regardless) if you need update detection.
# ---------------------------------------------------------------------------
TABLES = {
    "documents": {"id": "id", "vector": "embedding",
                  "metadata": ["title", "category"], "updated_at": None},
    "products":  {"id": "id", "vector": "embedding",
                  "metadata": ["name", "price"], "updated_at": None},
    "images":    {"id": "id", "vector": "embedding",
                  "metadata": ["caption", "source"], "updated_at": None},
}

UPSERT_BATCH = 200    # vectors per upsert (2 MB request limit binds for 768-dim)
DELETE_BATCH = 1000   # ids per delete


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def connect():
    conn = psycopg.connect(os.environ["PG_CONN"], autocommit=True)
    register_vector(conn)
    return conn


def get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ.get("PINECONE_INDEX", "pg-sync"))


def select_cols(cfg):
    return ", ".join([cfg["id"]] + cfg["metadata"] + [cfg["vector"]])


def build_record(table, row, cfg):
    """row is (id, *metadata_cols, vector) -> (pinecone_id, values, metadata)."""
    row_id = row[0]
    vector = row[-1]                                   # numpy array via register_vector
    metadata = {"source_table": table}
    for name, value in zip(cfg["metadata"], row[1:-1]):
        if value is None:
            continue
        if isinstance(value, Decimal):
            value = float(value)                       # metadata can't hold Decimal
        metadata[name] = value
    return (f"{table}#{row_id}", vector.tolist(), metadata)


def upsert_records(index, table, records):
    for i in range(0, len(records), UPSERT_BATCH):
        index.upsert(vectors=records[i:i + UPSERT_BATCH], namespace=table)


def delete_ids(index, namespace, ids):
    """Delete a list of full Pinecone ids, in batches."""
    for i in range(0, len(ids), DELETE_BATCH):
        index.delete(ids=ids[i:i + DELETE_BATCH], namespace=namespace)


def delete_row_ids(index, table, row_ids):
    """Delete by raw source ids (prefixed with the table name)."""
    delete_ids(index, table, [f"{table}#{r}" for r in row_ids])


def fetch_records(conn, table, cfg, row_ids):
    """Read current rows for the given source ids and build Pinecone records."""
    rows = conn.execute(
        f"SELECT {select_cols(cfg)} FROM {table} WHERE {cfg['id']}::text = ANY(%s)",
        ([str(r) for r in row_ids],),
    ).fetchall()
    return [build_record(table, r, cfg) for r in rows]


# ---------------------------------------------------------------------------
# backfill — one-time full load (use this OR the notebook migration)
# ---------------------------------------------------------------------------
def backfill(conn, index):
    for table, cfg in TABLES.items():
        total, batch = 0, []
        with conn.transaction():
            with conn.cursor(name=f"bf_{table}") as cur:
                cur.execute(f"SELECT {select_cols(cfg)} FROM {table}")
                for row in cur:
                    batch.append(build_record(table, row, cfg))
                    if len(batch) >= UPSERT_BATCH:
                        index.upsert(vectors=batch, namespace=table)
                        total += len(batch); batch = []
            if batch:
                index.upsert(vectors=batch, namespace=table)
                total += len(batch)
        print(f"backfill {table}: upserted {total} vectors")


# ---------------------------------------------------------------------------
# Strategy A — watermark / anti-join
# ---------------------------------------------------------------------------
def init_watermark(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pinecone_ledger (
            table_name text, id text, PRIMARY KEY (table_name, id));""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pinecone_sync_state (
            table_name      text PRIMARY KEY,
            last_updated_at timestamptz NOT NULL DEFAULT 'epoch');""")
    print("watermark: created pinecone_ledger and pinecone_sync_state")


def sync_watermark(conn, index):
    for table, cfg in TABLES.items():
        idcol, ua = cfg["id"], cfg.get("updated_at")
        conn.execute("INSERT INTO pinecone_sync_state(table_name) VALUES (%s) "
                     "ON CONFLICT DO NOTHING", (table,))
        watermark = conn.execute(
            "SELECT last_updated_at FROM pinecone_sync_state WHERE table_name=%s",
            (table,)).fetchone()[0]

        # --- upserts: rows not yet migrated, or changed since the watermark
        not_migrated = (f"NOT EXISTS (SELECT 1 FROM pinecone_ledger l "
                        f"WHERE l.table_name=%s AND l.id={idcol}::text)")
        if ua:
            sql = (f"SELECT {select_cols(cfg)}, {ua} FROM {table} "
                   f"WHERE {not_migrated} OR {ua} > %s")
            rows = conn.execute(sql, (table, watermark)).fetchall()
        else:
            sql = f"SELECT {select_cols(cfg)} FROM {table} WHERE {not_migrated}"
            rows = conn.execute(sql, (table,)).fetchall()

        records, new_ids, max_ua = [], [], watermark
        for r in rows:
            if ua:
                *core, ts = r
                if ts is not None and (max_ua is None or ts > max_ua):
                    max_ua = ts
            else:
                core = list(r)
            records.append(build_record(table, core, cfg))
            new_ids.append(str(core[0]))
        upsert_records(index, table, records)

        with conn.cursor() as cur:
            cur.executemany("INSERT INTO pinecone_ledger(table_name, id) VALUES (%s, %s) "
                            "ON CONFLICT DO NOTHING", [(table, i) for i in new_ids])
        if ua and max_ua is not None:
            conn.execute("UPDATE pinecone_sync_state SET last_updated_at=%s "
                         "WHERE table_name=%s", (max_ua, table))

        # --- deletes: ledger ids whose source row no longer exists
        stale = [row[0] for row in conn.execute(
            f"SELECT l.id FROM pinecone_ledger l WHERE l.table_name=%s "
            f"AND NOT EXISTS (SELECT 1 FROM {table} s WHERE s.{idcol}::text = l.id)",
            (table,)).fetchall()]
        delete_row_ids(index, table, stale)
        with conn.cursor() as cur:
            cur.executemany("DELETE FROM pinecone_ledger WHERE table_name=%s AND id=%s",
                            [(table, i) for i in stale])

        print(f"watermark sync {table}: upserted {len(records)}, deleted {len(stale)}")


# ---------------------------------------------------------------------------
# Strategy B — change-log (outbox) + triggers
# ---------------------------------------------------------------------------
def init_changelog(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pinecone_changelog (
            seq        bigserial PRIMARY KEY,
            table_name text        NOT NULL,
            row_id     text        NOT NULL,
            op         text        NOT NULL CHECK (op IN ('upsert', 'delete')),
            changed_at timestamptz NOT NULL DEFAULT now(),
            processed  boolean     NOT NULL DEFAULT false);""")
    for table, cfg in TABLES.items():
        idcol, fn = cfg["id"], f"log_pinecone_change_{table}"
        conn.execute(f"""
            CREATE OR REPLACE FUNCTION {fn}() RETURNS trigger AS $fn$
            BEGIN
              IF (TG_OP = 'DELETE') THEN
                INSERT INTO pinecone_changelog(table_name, row_id, op)
                  VALUES (TG_TABLE_NAME, OLD.{idcol}::text, 'delete');
                RETURN OLD;
              ELSE
                INSERT INTO pinecone_changelog(table_name, row_id, op)
                  VALUES (TG_TABLE_NAME, NEW.{idcol}::text, 'upsert');
                RETURN NEW;
              END IF;
            END; $fn$ LANGUAGE plpgsql;""")
        conn.execute(f"DROP TRIGGER IF EXISTS trg_pinecone_{table} ON {table};")
        conn.execute(f"CREATE TRIGGER trg_pinecone_{table} "
                     f"AFTER INSERT OR UPDATE OR DELETE ON {table} "
                     f"FOR EACH ROW EXECUTE FUNCTION {fn}();")
    print("changelog: created pinecone_changelog and per-table triggers")


def sync_changelog(conn, index):
    max_seq = conn.execute(
        "SELECT COALESCE(max(seq), 0) FROM pinecone_changelog WHERE NOT processed"
    ).fetchone()[0]
    if max_seq == 0:
        print("changelog sync: nothing to do")
        return

    # Latest op per row up to the frontier (insert+delete collapses to delete).
    rows = conn.execute("""
        SELECT DISTINCT ON (table_name, row_id) table_name, row_id, op
        FROM   pinecone_changelog
        WHERE  NOT processed AND seq <= %s
        ORDER  BY table_name, row_id, seq DESC""", (max_seq,)).fetchall()

    upserts, deletes = defaultdict(list), defaultdict(list)
    for table, row_id, op in rows:
        (upserts if op == "upsert" else deletes)[table].append(row_id)

    for table, ids in deletes.items():
        delete_row_ids(index, table, ids)
    for table, ids in upserts.items():
        upsert_records(index, table, fetch_records(conn, table, TABLES[table], ids))

    conn.execute("UPDATE pinecone_changelog SET processed=true "
                 "WHERE NOT processed AND seq <= %s", (max_seq,))
    n_up = sum(len(v) for v in upserts.values())
    n_del = sum(len(v) for v in deletes.values())
    print(f"changelog sync: {n_up} upserts, {n_del} deletes (through seq {max_seq})")


# ---------------------------------------------------------------------------
# reconcile — id diff safety net (works for either strategy)
# ---------------------------------------------------------------------------
def reconcile(conn, index):
    for table, cfg in TABLES.items():
        pg_ids = {f"{table}#{r[0]}" for r in
                  conn.execute(f"SELECT {cfg['id']} FROM {table}").fetchall()}
        pc_ids = set()
        for id_batch in index.list(namespace=table):     # pages of ids
            # index.list yields ListItem objects (.id) in pinecone>=9; older
            # versions yielded plain id strings. getattr handles both.
            pc_ids.update(getattr(i, "id", i) for i in id_batch)

        missing = pg_ids - pc_ids        # in pgvector, not Pinecone -> upsert
        stale = pc_ids - pg_ids          # in Pinecone, not pgvector -> delete
        if missing:
            raw_ids = [m.split("#", 1)[1] for m in missing]
            upsert_records(index, table, fetch_records(conn, table, cfg, raw_ids))
        if stale:
            delete_ids(index, table, list(stale))
        print(f"reconcile {table}: +{len(missing)} upserted, -{len(stale)} deleted")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Keep Pinecone in sync with pgvector.")
    ap.add_argument("command", choices=["init", "backfill", "sync", "reconcile"])
    ap.add_argument("--strategy", choices=["watermark", "changelog"], default="changelog")
    args = ap.parse_args()

    conn = connect()
    index = get_index()
    if args.command == "init":
        init_watermark(conn) if args.strategy == "watermark" else init_changelog(conn)
    elif args.command == "backfill":
        backfill(conn, index)
    elif args.command == "sync":
        sync_watermark(conn, index) if args.strategy == "watermark" else sync_changelog(conn, index)
    elif args.command == "reconcile":
        reconcile(conn, index)
    conn.close()


if __name__ == "__main__":
    main()
