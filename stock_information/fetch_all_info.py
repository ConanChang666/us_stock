import os, time, json, random
from datetime import datetime
from time import monotonic
import requests
from db.MySQL_db_connection import MySQLConn
from stock_information.ticker import tickers

API_KEY = (os.getenv("AV_API") or "").strip()
if not API_KEY:
    raise RuntimeError("請在環境變數 AV_API 設定你的 Alpha Vantage API key")

BASE_URL = "https://www.alphavantage.co/query"
DB_NAME  = "stock_market_data_lake"
RATE_RPM = 75.0
JITTER_PCT = 0.15
BASE_INTERVAL = 60.0 / RATE_RPM

UPSERT_SQL = """
INSERT INTO us_company_overview_raw (symbol, payload, fetched_at)
VALUES (%s, CAST(%s AS JSON), %s)
ON DUPLICATE KEY UPDATE
  payload = VALUES(payload),
  fetched_at = VALUES(fetched_at);
"""

def _pace(next_at: list[float]):
    now = monotonic()
    sleep_for = next_at[0] - now
    if sleep_for > 0:
        time.sleep(sleep_for)
    jitter = random.uniform(0, BASE_INTERVAL * JITTER_PCT)
    next_at[0] = max(monotonic(), next_at[0]) + BASE_INTERVAL + jitter

def fetch_overview_once(symbol: str, pacer_state: list[float]) -> dict | None:
    """只呼叫一次，如果抓不到資料直接略過（回傳 None）。"""
    _pace(pacer_state)
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": API_KEY}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"網路錯誤 {symbol}: {e}，略過")
        return None

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} {symbol}，略過")
        return None

    try:
        data = resp.json()
    except Exception:
        print(f"非 JSON 回應 {symbol}，略過")
        return None

    # 沒有關鍵欄位也略過
    if not isinstance(data, dict) or (("Symbol" not in data) and ("Name" not in data)):
        print(f"沒有有效資料 {symbol}，略過")
        return None

    data.setdefault("Symbol", symbol)
    return data

def main():
    symbols = list(tickers.keys())
    print(f"準備抓取 {len(symbols)} 檔 symbol（固定 ~0.8s/req，抓不到直接略過）")

    fetched_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    next_at = [monotonic()]  

    with MySQLConn(DB_NAME) as conn:
        with conn.cursor() as cur:
            count_success = 0
            for i, sym in enumerate(symbols, 1):
                print(f"[{i}/{len(symbols)}] {sym} ...")
                data = fetch_overview_once(sym, next_at)
                if data is None:   # 抓不到,直接跳過
                    continue
                cur.execute(UPSERT_SQL, (sym, json.dumps(data, ensure_ascii=False), fetched_at))
                count_success += 1
                if count_success % 50 == 0:
                    conn.commit()
            conn.commit()

    print("抓取完成")

if __name__ == "__main__":
    main()