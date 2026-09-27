"""A small synthetic relational schema that exercises every feature family.

    users     (entity, static)      user_id, age, country
    products  (static parent)       product_id, price, category
    orders    (timed child)         order_id, user_id -> users, product_id -> products, ts,
                                    amount, channel
    reviews   (timed grandchild)    review_id, order_id -> orders, ts (after the order),
                                    helpful, rating

1-hop windows (orders), 2-hop track records (users <- orders -> products), grandchild
aggregates (users <- orders <- reviews), parent context (orders -> products) and day-level
recency (orders) all have rows here.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("RELDFS_TMP", os.path.join(tempfile.gettempdir(), "reldfs-test-duckdb"))


def make_db(seed: int = 0):
    from relbench.base import Database, Table

    rs = np.random.RandomState(seed)
    n_users, n_products, n_orders = 60, 15, 1500
    t0 = pd.Timestamp("2020-01-01")
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "age": rs.randint(18, 80, n_users).astype("float64"),
        "country": rs.choice(["a", "b", "c"], n_users),
    })
    products = pd.DataFrame({
        "product_id": np.arange(n_products),
        "price": rs.gamma(2.0, 10.0, n_products).round(2),
        "category": rs.choice(["x", "y", "z", "w"], n_products),
    })
    ots = t0 + pd.to_timedelta(rs.randint(0, 730 * 24, n_orders), unit="h")
    orders = pd.DataFrame({
        "order_id": np.arange(n_orders),
        "user_id": rs.randint(0, n_users, n_orders),
        "product_id": rs.randint(0, n_products, n_orders),
        "ts": ots,
        "amount": rs.gamma(2.0, 20.0, n_orders).round(2),
        "channel": rs.choice(["web", "app", "shop"], n_orders),
    }).sort_values("ts", kind="mergesort").reset_index(drop=True)
    has = rs.rand(n_orders) < 0.7
    rv = orders[has]
    reviews = pd.DataFrame({
        "review_id": np.arange(len(rv)),
        "order_id": rv["order_id"].to_numpy(),
        "ts": rv["ts"].to_numpy() + pd.to_timedelta(rs.randint(1, 60 * 24, len(rv)), unit="h"),
        "helpful": rs.gamma(1.5, 3.0, len(rv)).round(3),
        "rating": rs.randint(1, 6, len(rv)),
    })
    db = Database({
        "users": Table(df=users, fkey_col_to_pkey_table={}, pkey_col="user_id", time_col=None),
        "products": Table(df=products, fkey_col_to_pkey_table={}, pkey_col="product_id", time_col=None),
        "orders": Table(df=orders, fkey_col_to_pkey_table={"user_id": "users", "product_id": "products"},
                        pkey_col="order_id", time_col="ts"),
        "reviews": Table(df=reviews, fkey_col_to_pkey_table={"order_id": "orders"},
                         pkey_col="review_id", time_col="ts"),
    })
    return db


def make_task():
    return SimpleNamespace(entity_table="users", entity_col="user_id", time_col="timestamp",
                           target_col="label", timedelta=pd.Timedelta(days=30))


def make_labels(seed: int = 0) -> pd.DataFrame:
    rs = np.random.RandomState(seed + 1)
    cuts = pd.date_range("2020-03-01", periods=18, freq="30D")   # all at midnight
    rows = [(u, c) for c in cuts for u in range(60) if rs.rand() < 0.6]
    lab = pd.DataFrame(rows, columns=["user_id", "timestamp"])
    lab["label"] = rs.randint(0, 2, len(lab))
    return lab


@pytest.fixture(scope="module")
def toy():
    import reldfs.features as F

    db, task, lab = make_db(), make_task(), make_labels()
    plan = F.make_plan(db, task)
    return SimpleNamespace(db=db, task=task, labels=lab, plan=plan)
