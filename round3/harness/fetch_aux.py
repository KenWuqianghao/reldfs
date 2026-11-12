"""Fetch small numeric summaries of static string columns that the slim mirror drops (v4).

    python harness/fetch_aux.py

Writes data/aux/<name>.parquet, one row per entity, numbers only (no text is kept):
  amazon_customer   customer_id, name_len, name_freq (customers with the same lower-cased name),
                    name_default (Amazon Customer / Kindle Customer), name_space
  amazon_product    product_id, brand_freq (products of the brand), brand_null, title_len, desc_len
  hm_customer       customer_id, postal_freq (customers with the same postal code)
These are attributes of the entity row as of the dump date, like the mirror's other static columns.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()   # local mirrors and caches
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

BASE = "datasets/stanford-star/relbench-v1"


def read(fs, path: str, cols: list) -> pd.DataFrame:
    return pq.ParquetFile(fs.open(f"{BASE}/{path}", "rb", block_size=8 << 20)).read(columns=cols).to_pandas()


def main() -> int:
    fs = HfFileSystem()
    out = DATA / "aux"
    out.mkdir(parents=True, exist_ok=True)

    c = read(fs, "rel-amazon/db/customer.parquet", ["customer_id", "customer_name"])
    nm = c["customer_name"].fillna("").astype(str).str.strip().str.lower()
    pd.DataFrame({"customer_id": c["customer_id"], "name_len": nm.str.len().astype("float32"),
                  "name_freq": nm.map(nm.value_counts()).astype("float32"),
                  "name_default": nm.isin(["amazon customer", "kindle customer"]).astype("float32"),
                  "name_space": nm.str.contains(" ").astype("float32")}).to_parquet(out / "amazon_customer.parquet")
    print(f"[aux] amazon_customer {len(c)} rows; default names {float(nm.isin(['amazon customer', 'kindle customer']).mean()):.3f}", flush=True)

    p = read(fs, "rel-amazon/db/product.parquet", ["product_id", "brand", "title", "description"])
    br = p["brand"].fillna("").astype(str).str.strip().str.lower()
    pd.DataFrame({"product_id": p["product_id"], "brand_freq": br.map(br.value_counts()).astype("float32"),
                  "brand_null": (br == "").astype("float32"),
                  "title_len": p["title"].fillna("").astype(str).str.len().astype("float32"),
                  "desc_len": p["description"].fillna("").astype(str).str.len().astype("float32")}
                 ).to_parquet(out / "amazon_product.parquet")
    print(f"[aux] amazon_product {len(p)} rows", flush=True)

    h = read(fs, "rel-hm/db/customer.parquet", ["customer_id", "postal_code"])
    pc = h["postal_code"].fillna("").astype(str)
    pd.DataFrame({"customer_id": h["customer_id"], "postal_freq": pc.map(pc.value_counts()).astype("float32")}
                 ).to_parquet(out / "hm_customer.parquet")
    print(f"[aux] hm_customer {len(h)} rows; distinct postal codes {pc.nunique()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
