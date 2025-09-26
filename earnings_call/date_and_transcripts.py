import os
import time
import json
import requests
from datetime import date, datetime
from dotenv import load_dotenv
from db.MySQL_db_connection import MySQLConn
from opencc import OpenCC
import torch
from transformers import MarianMTModel, MarianTokenizer

load_dotenv()
AV_API = os.getenv("AV_API")

DB_NAME = "stock_market_data_lake"
SYMBOL_TABLE = "us_company_overview_raw"
SYMBOL_COL = "symbol"

START_DATE = date(date.today().year, 1, 1)
END_DATE = date.today()

REQ_SLEEP_SEC = 0.8
USE_REPORTED_DATE_FALLBACK = True

# --------------- 翻譯模型初始化 ---------------
EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"
_tokenizer = None
_model = None
_opencc_s2twp = OpenCC("s2twp")  # s2twp: Simplified Chinese to Traditional Chinese (Taiwan)

def load_translation_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("[init] Loading translation model...", flush=True)
        _tokenizer = MarianTokenizer.from_pretrained(EN_ZH_MODEL)
        _model = MarianMTModel.from_pretrained(EN_ZH_MODEL)
        print("[init] Model loaded.", flush=True)
    return _tokenizer, _model

def batch_translate_en_to_zh_cn(texts, max_new_tokens=256):
    tok, mdl = load_translation_model()
    if not texts:
        return []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    with torch.no_grad():
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False
        )
    return [tok.decode(o, skip_special_tokens=True) for o in outputs]

def convert_zh_cn_to_zh_tw(texts):
    return [_opencc_s2twp.convert(t) if t else t for t in texts]

# --------------- Alpha Vantage 抓取邏輯 ---------------
def safe_parse_date(s: str):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    return None

def compute_quarter_code(fiscal_date_str: str):
    if not fiscal_date_str:
        return None
    try:
        y, m, _ = fiscal_date_str.split("-")
        q = (int(m)-1)//3 + 1
        return f"{y}Q{q}"
    except:
        return None

def normalize_quarter(q: str, fallback: str):
    if not q:
        return fallback
    q = q.strip().upper().replace(" ", "").replace("-", "")
    if "FY" in q:
        q = q.replace("FY", "")
    if q.startswith("Q") and len(q) >= 6 and q[2:].isdigit():
        return q[2:] + q[:2]
    if len(q) == 6 and q[4] == "Q" and q[:4].isdigit() and q[5] in "1234":
        return q
    return fallback

def fetch_json(session, url, max_retries=3):
    backoff = 10
    for attempt in range(1, max_retries + 1):
        r = session.get(url, timeout=30)
        try:
            data = r.json()
        except:
            data = {}
        if isinstance(data, dict) and ("Note" in data or "Information" in data):
            if attempt == max_retries:
                raise RuntimeError(data.get("Note") or data.get("Information") or "Rate limited")
            time.sleep(backoff)
            backoff *= 2
            continue
        return data
    return {}

def fetch_earnings(session, symbol):
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={AV_API}"
    return fetch_json(session, url)

def fetch_transcript(session, symbol, quarter_code):
    url = (f"https://www.alphavantage.co/query?"
           f"function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={quarter_code}&apikey={AV_API}")
    d = fetch_json(session, url)
    if not isinstance(d, dict):
        return None, None, None
    # 解析 call_date
    candidates = [
        d.get("callDate"), d.get("eventDate"), d.get("date"),
        (d.get("metadata") or {}).get("eventDate") if isinstance(d.get("metadata"), dict) else None
    ]
    call_date = None
    for c in candidates:
        dt = safe_parse_date(c)
        if dt:
            call_date = dt
            break
    quarter_from_api = normalize_quarter(d.get("quarter") or d.get("Quarter") or quarter_code,
                                         quarter_code)
    transcript_only = d.get("transcript")
    return call_date, quarter_from_api, transcript_only

def load_symbols():
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT {SYMBOL_COL} AS symbol
                FROM {SYMBOL_TABLE}
                WHERE {SYMBOL_COL} IS NOT NULL AND {SYMBOL_COL} <> ''
            """)
            rows = cur.fetchall()
    return sorted({(r["symbol"] or "").strip().upper() for r in rows if r.get("symbol")})

def upsert_call_date(symbol, call_date, quarter, fiscal_date, report_date):
    sql = """
    INSERT INTO us_earnings_call_date
    (symbol, call_date, quarter, fiscal_date, report_date)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      quarter=VALUES(quarter),
      fiscal_date=VALUES(fiscal_date),
      report_date=VALUES(report_date)
    """
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, call_date, quarter, fiscal_date, report_date))
            conn.commit()
            call_id = cur.lastrowid
            if call_id == 0:
                cur.execute(
                    "SELECT id FROM us_earnings_call_date WHERE symbol=%s AND call_date=%s",
                    (symbol, call_date),
                )
                row = cur.fetchone()
                call_id = row["id"] if row else None
            return call_id

def upsert_transcript(call_id: int, lang: str, transcript_list: list):
    sql = """
    INSERT INTO us_earnings_call_transcripts
    (call_id, lang, transcript)
    VALUES (%s, %s, CAST(%s AS JSON))
    ON DUPLICATE KEY UPDATE
      transcript = VALUES(transcript)
    """
    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (call_id, lang, json.dumps(transcript_list, ensure_ascii=False)))
        conn.commit()

# ---------------- 主程式 ----------------
def main():
    session = requests.Session()
    symbols = load_symbols()
    print(f"Loaded {len(symbols)} symbols.")

    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        earnings = fetch_earnings(session, sym)
        q_earnings = (earnings or {}).get("quarterlyEarnings", []) or []

        for q in q_earnings:
            fiscal_str = q.get("fiscalDateEnding")
            report_str = q.get("reportedDate")
            fiscal_dt  = safe_parse_date(fiscal_str)
            report_dt  = safe_parse_date(report_str)
            if not (fiscal_dt and report_dt):
                continue
            if not (START_DATE <= report_dt <= END_DATE):
                continue

            quarter_code = compute_quarter_code(fiscal_str)
            if not quarter_code:
                continue

            call_date, quarter_from_api, transcript_en = fetch_transcript(session, sym, quarter_code)
            if not call_date and USE_REPORTED_DATE_FALLBACK:
                call_date = report_dt
            if not call_date:
                continue
            if not transcript_en:
                continue

            call_id = upsert_call_date(sym, call_date, quarter_from_api or quarter_code,
                                       fiscal_dt, report_dt)

            # 1️⃣ 英文逐字稿
            upsert_transcript(call_id, 'en', transcript_en)

            # 2️⃣ 翻譯為簡體中文
            en_texts = [seg.get("content", "") for seg in transcript_en]
            zh_cn_contents = batch_translate_en_to_zh_cn(en_texts)
            transcript_cn = []
            for seg, cn_text in zip(transcript_en, zh_cn_contents):
                transcript_cn.append({"title": seg.get("title", ""), "content": cn_text})
            upsert_transcript(call_id, 'zh-cn', transcript_cn)

            # 3️⃣ 將簡體轉為繁體
            zh_tw_contents = convert_zh_cn_to_zh_tw(zh_cn_contents)
            transcript_tw = []
            for seg, tw_text in zip(transcript_en, zh_tw_contents):
                transcript_tw.append({"title": seg.get("title", ""), "content": tw_text})
            upsert_transcript(call_id, 'zh-tw', transcript_tw)

            print(f"    Saved transcripts (en, zh-cn, zh-tw) for {sym} {quarter_from_api}")
            time.sleep(REQ_SLEEP_SEC)

if __name__ == "__main__":
    main()