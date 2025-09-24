# -*- coding: utf-8 -*-
"""
ETL: stock_market_data_lake.us_company_overview_raw (payload JSON)
 -> stock_market_data_lake.us_stock_company_info_clean

- 以 payload.Symbol 對照 identifier.ticker_mapping.symbol（active=1）
- exchange -> market
- industry -> industry_id
- 其餘來自 raw payload
- 名稱/描述先以英文佔位（翻譯另行補齊）

環境變數（由 db/MySQL_db_connection.py 讀取）：
- MYSQL_DB_HOST
- MYSQL_DB_USER
- MYSQL_DB_PWD
- （可選）MYSQL_POOL_SIZE, MYSQL_POOL_TIMEOUT, MYSQL_POOL_PING
"""

from db.MySQL_db_connection import MySQLConn

SRC_DB = "stock_market_data_lake"
DST_DB = "stock_market_data_lake"
ID_DB  = "identifier"

RAW_TABLE   = f"{SRC_DB}.us_company_overview_raw"
CLEAN_TABLE = f"{DST_DB}.us_stock_company_info_clean"
MAP_TABLE   = f"{ID_DB}.ticker_mapping"


UPSERT_SQL = f"""
INSERT INTO {CLEAN_TABLE} (
  stock_id, stock_name, industry_id, market, country, currency,
  office_website, address, description
)
SELECT
  -- stock_id
  JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Symbol')) AS stock_id,

  -- stock_name（先以英文佔位，後續翻譯補繁/簡）
  JSON_OBJECT(
    'zh_tw', JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Name')),
    'zh_cn', JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Name')),
    'en',    JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Name'))
  ) AS stock_name,

  -- industry_id 從 mapping.industry（整數）
  tm.industry AS industry_id,

  -- market 從 mapping.exchange（NYSE / NASDAQ / AMEX / CBOE ...）
  tm.exchange AS market,

  -- country 固定 US（若想以 payload.Country 轉 US/USA 可在此處正規化）
  'US' AS country,

  -- currency 來自 payload.Currency，預設 USD
  COALESCE(JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Currency')), 'USD') AS currency,

  -- 官網
  JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.OfficialSite')) AS office_website,

  -- address 只存英文；保 255 長度
  SUBSTRING(JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Address')), 1, 255) AS address,

  -- description（先以英文佔位，後續翻譯補繁/簡）
  JSON_OBJECT(
    'zh_tw', JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Description')),
    'zh_cn', JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Description')),
    'en',    JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Description'))
  ) AS description

FROM {RAW_TABLE} r
JOIN {MAP_TABLE} tm
  ON tm.symbol = JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Symbol'))
WHERE
  tm.active = 1
  AND JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Symbol')) IS NOT NULL
  AND JSON_UNQUOTE(JSON_EXTRACT(r.payload, '$.Name'))   IS NOT NULL

ON DUPLICATE KEY UPDATE
  stock_name     = VALUES(stock_name),
  industry_id    = VALUES(industry_id),
  market         = VALUES(market),
  country        = VALUES(country),
  currency       = VALUES(currency),
  office_website = VALUES(office_website),
  address        = VALUES(address),
  description    = VALUES(description),
  updated_at     = CURRENT_TIMESTAMP;
"""


def run():
    # 使用連線池連到任一 DB 名稱即可；pool key 取決於 host/user/db
    with MySQLConn(DST_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(UPSERT_SQL)
            affected = cur.rowcount  # 受影響筆數（對 INSERT…SELECT 代表插入/更新的總筆數）
        conn.commit()
        print(f"[us_overview_raw_to_clean] upsert affected rows: {affected}")


if __name__ == "__main__":
    run()