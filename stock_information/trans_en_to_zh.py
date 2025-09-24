import argparse
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from transformers import MarianMTModel, MarianTokenizer
from db.MySQL_db_connection import MySQLConn
from opencc import OpenCC
import torch  # ← 新增

DB = "stock_market_data_lake"
TABLE = f"{DB}.us_stock_company_info_clean"

EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"

_tokenizer = None
_model = None
_opencc_s2twp = OpenCC("s2twp")

def load_model_once():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"[init] loading model: {EN_ZH_MODEL}", flush=True)
        _tokenizer = MarianTokenizer.from_pretrained(EN_ZH_MODEL)
        _model = MarianMTModel.from_pretrained(EN_ZH_MODEL)
        print("[init] model loaded.", flush=True)
    return _tokenizer, _model

def batch_translate_en_to_zh_cn(texts: List[str], max_new_tokens: int = 256, debug: bool = False) -> List[str]:
    tok, mdl = load_model_once()
    if not texts:
        return []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    if debug:
        print(f"[debug] translate batch size={len(texts)}, device={device}, max_new_tokens={max_new_tokens}", flush=True)
    with torch.no_grad():
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False
        )
    return [tok.decode(o, skip_special_tokens=True) for o in outputs]

def to_zh_tw_from_zh_cn(texts: List[str]) -> List[str]:
    return [_opencc_s2twp.convert(t) if t else t for t in texts]


# ---------------- SQL（描述/名稱各自處理）----------------

SQL_COUNT_DESC = f"""
SELECT COUNT(1) AS cnt
FROM {TABLE}
WHERE
  JSON_UNQUOTE(JSON_EXTRACT(description, '$.en')) IS NOT NULL
  AND JSON_UNQUOTE(JSON_EXTRACT(description, '$.en')) <> ''
  AND (
    JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) = ''
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) = JSON_UNQUOTE(JSON_EXTRACT(description, '$.en'))
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_cn')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_cn')) = ''
  );
"""

SQL_FETCH_DESC = f"""
SELECT stock_id,
       JSON_UNQUOTE(JSON_EXTRACT(description, '$.en')) AS desc_en,
       JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) AS desc_tw,
       JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_cn')) AS desc_cn
FROM {TABLE}
WHERE
  JSON_UNQUOTE(JSON_EXTRACT(description, '$.en')) IS NOT NULL
  AND JSON_UNQUOTE(JSON_EXTRACT(description, '$.en')) <> ''
  AND (
    JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) = ''
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_tw')) = JSON_UNQUOTE(JSON_EXTRACT(description, '$.en'))
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_cn')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(description, '$.zh_cn')) = ''
  )
ORDER BY stock_id
LIMIT %s OFFSET %s;
"""

SQL_UPDATE_DESC = f"""
UPDATE {TABLE}
SET description = JSON_SET(
      description,
      '$.zh_tw', %s,
      '$.zh_cn', %s
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE stock_id = %s;
"""

# ---- stock_name ----
SQL_COUNT_NAME = f"""
SELECT COUNT(1) AS cnt
FROM {TABLE}
WHERE
  JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en')) IS NOT NULL
  AND JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en')) <> ''
  AND (
    JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) = ''
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) = JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en'))
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_cn')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_cn')) = ''
  );
"""

SQL_FETCH_NAME = f"""
SELECT stock_id,
       JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en')) AS name_en,
       JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) AS name_zh_tw,
       JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_cn')) AS name_zh_cn
FROM {TABLE}
WHERE
  JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en')) IS NOT NULL
  AND JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en')) <> ''
  AND (
    JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) = ''
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_tw')) = JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.en'))
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_cn')) IS NULL
    OR JSON_UNQUOTE(JSON_EXTRACT(stock_name, '$.zh_cn')) = ''
  )
ORDER BY stock_id
LIMIT %s OFFSET %s;
"""

SQL_UPDATE_NAME = f"""
UPDATE {TABLE}
SET stock_name = JSON_SET(
      stock_name,
      '$.zh_tw', %s,
      '$.zh_cn', %s
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE stock_id = %s;
"""


# ---------------- DB 操作 ----------------

def count_candidates(only: str) -> Tuple[int, int]:
    """回傳 (name_cnt, desc_cnt)；若 only 指定就只計其中一個。"""
    with MySQLConn(DB) as conn, conn.cursor() as cur:
        name_cnt = desc_cnt = 0
        if only in ("", "both", "name"):
            cur.execute(SQL_COUNT_NAME); name_cnt = cur.fetchone()["cnt"]
        if only in ("", "both", "description"):
            cur.execute(SQL_COUNT_DESC); desc_cnt = cur.fetchone()["cnt"]
    return name_cnt, desc_cnt


def fetch_batch(only: str, limit: int, offset: int) -> Dict[str, List[Dict[str, Any]]]:
    out = {"name": [], "description": []}
    with MySQLConn(DB) as conn, conn.cursor() as cur:
        if only in ("", "both", "name"):
            cur.execute(SQL_FETCH_NAME, (limit, offset))
            out["name"] = cur.fetchall()
        if only in ("", "both", "description"):
            cur.execute(SQL_FETCH_DESC, (limit, offset))
            out["description"] = cur.fetchall()
    return out


# ---------------- 主程式 ----------------

def main():
    ap = argparse.ArgumentParser(description="EN -> ZH(zh_cn -> zh_tw) for name/description")
    ap.add_argument("--limit", type=int, default=10, help="Rows per batch for each type")  # ← 改成預設10，比較快驗證
    ap.add_argument("--offset", type=int, default=0, help="Offset per type")
    ap.add_argument("--only", choices=["name", "description", "both"], default="both",
                    help="Translate only 'name', only 'description', or 'both'")
    ap.add_argument("--dry-run", action="store_true", help="Print only, do not write DB")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")                # ← 新增
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Limit generation length")  # ← 新增
    args = ap.parse_args()

    name_cnt, desc_cnt = count_candidates(args.only)
    print(f"[info] candidates -> name: {name_cnt}, description: {desc_cnt}", flush=True)

    # --- names ---
    if args.only in ("both", "name") and name_cnt > 0:
        while True:
            rows = fetch_batch("name", args.limit, 0)["name"]  # ← 固定用 OFFSET 0
            if not rows:
                break

            sids = [r["stock_id"] for r in rows]
            texts_en = [r["name_en"] or "" for r in rows]

            zh_cn_list = batch_translate_en_to_zh_cn(texts_en)
            zh_tw_list = to_zh_tw_from_zh_cn(zh_cn_list)

            updates = list(zip(zh_tw_list, zh_cn_list, sids))  # (tw, cn, id)

            if not args.dry_run and updates:
                with MySQLConn(DB) as conn, conn.cursor() as cur:
                    cur.executemany(SQL_UPDATE_NAME, updates)
                    conn.commit()
                print(f"[name] updated {len(updates)} rows.")
            else:
                print("[stage] dry-run=True or no updates; skip DB write.")

    # --- descriptions ---
    if args.only in ("both", "description") and desc_cnt > 0:
        while True:
            rows = fetch_batch("description", args.limit, 0)["description"]  # ← 固定用 OFFSET 0
            if not rows:
                break

            sids = [r["stock_id"] for r in rows]
            texts_en = [r["desc_en"] or "" for r in rows]

            zh_cn_list = batch_translate_en_to_zh_cn(texts_en)
            zh_tw_list = to_zh_tw_from_zh_cn(zh_cn_list)

            updates = list(zip(zh_tw_list, zh_cn_list, sids))  # (tw, cn, id)

            if not args.dry_run and updates:
                with MySQLConn(DB) as conn, conn.cursor() as cur:
                    cur.executemany(SQL_UPDATE_DESC, updates)
                    conn.commit()
                print(f"[desc] updated {len(updates)} rows.")
            else:
                print("[stage] dry-run=True or no updates; skip DB write.")

    print("[done] all tasks finished.", flush=True)

if __name__ == "__main__":
    main()