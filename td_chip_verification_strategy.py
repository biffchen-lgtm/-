"""
台灣股票 TD 九轉空頭籌碼對帳策略系統
偵測空頭破位訊號並跨表對帳融資、外資動態
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# 日誌配置
# ============================================================================
def setup_logging(log_level=logging.INFO):
    """設置日誌系統"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler('td_strategy.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# 數據清洗與轉換
# ============================================================================
def clean_numeric_value(val, allow_nan=True) -> Optional[float]:
    """
    通用數值清洗函式
    
    Args:
        val: 任意輸入值
        allow_nan: 是否允許 NaN 返回
        
    Returns:
        float: 清洗後的數值
    """
    if pd.isna(val):
        return np.nan if allow_nan else 0.0
    
    try:
        val_str = str(val).strip().replace(',', '').replace('"', '').replace(' ', '')
        
        # 快速過濾空值與特殊標記
        if not val_str or val_str in ('', '--', 'NaN', 'nan', 'N/A', 'n/a'):
            return np.nan if allow_nan else 0.0
        
        return float(val_str)
    except (ValueError, TypeError):
        return np.nan if allow_nan else 0.0

def validate_stock_id(stock_id: str) -> bool:
    """驗證股票代號"""
    stock_id = str(stock_id).strip()
    return len(stock_id) == 4 and stock_id.isdigit()

def extract_date_from_filename(filename: str) -> Optional[str]:
    """從檔案名稱提取日期"""
    try:
        date_str = filename.split('_')[-1].replace('.csv', '').strip()
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
    except (IndexError, AttributeError):
        pass
    return None

# ============================================================================
# 技術指標計算
# ============================================================================
def calculate_td_sell_setup(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """
    計算純空頭九轉破位序列
    核心邏輯：今日收盤價 < 前第 4 天收盤價 ➔ 空頭慣性啟動，計數累加 1，否則歸零
    
    Args:
        df: 包含價格數據的 DataFrame
        close_col: 收盤價欄位名稱
        
    Returns:
        pd.DataFrame: 增加 TD_Count 欄位的 DataFrame
    """
    if close_col not in df.columns:
        logger.warning(f"⚠️  未找到欄位 {close_col}，TD_Count 設為 0")
        df['TD_Count'] = 0
        return df
    
    closes = df[close_col].values
    td_counts = np.zeros(len(closes), dtype=int)
    
    for i in range(4, len(closes)):
        if pd.notna(closes[i]) and pd.notna(closes[i-4]):
            if closes[i] < closes[i-4]:
                td_counts[i] = td_counts[i-1] + 1
            else:
                td_counts[i] = 0
    
    df['TD_Count'] = td_counts
    return df

# ============================================================================
# 文件讀取與清洗
# ============================================================================
def read_price_csv(filepath: str) -> Optional[pd.DataFrame]:
    """安全讀取價格 CSV 檔案"""
    try:
        df = pd.read_csv(filepath, dtype=str, encoding='utf-8')
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, dtype=str, encoding='big5')
            return df
        except Exception as e:
            logger.warning(f"❌ 讀取價格檔案失敗 {filepath}: {e}")
            return None
    except Exception as e:
        logger.warning(f"❌ 讀取價格檔案出錯 {filepath}: {e}")
        return None

def read_margin_csv(filepath: str) -> Optional[pd.DataFrame]:
    """安全讀取融資 CSV 檔案"""
    try:
        df = pd.read_csv(filepath, dtype=str, encoding='utf-8')
        df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, dtype=str, encoding='big5')
            df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
            return df
        except Exception as e:
            logger.debug(f"⚠️  讀取融資檔案失敗: {e}")
            return None
    except Exception as e:
        logger.debug(f"⚠️  讀取融資檔案出錯: {e}")
        return None

def read_investor_csv(filepath: str) -> Optional[pd.DataFrame]:
    """安全讀取法人 CSV 檔案"""
    try:
        df = pd.read_csv(filepath, dtype=str, encoding='utf-8')
        df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, dtype=str, encoding='big5')
            df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
            return df
        except Exception as e:
            logger.debug(f"⚠️  讀取法人檔案失敗: {e}")
            return None
    except Exception as e:
        logger.debug(f"⚠️  讀取法人檔案出錯: {e}")
        return None

# ============================================================================
# 籌碼對帳引擎
# ============================================================================
def extract_margin_value(df_margin: Optional[pd.DataFrame], stock_id: str) -> float:
    """提取融資數值"""
    if df_margin is None or df_margin.empty:
        return np.nan
    
    try:
        # 尋找代號欄位
        id_cols = [c for c in df_margin.columns if '代號' in c or '股票' in c]
        if not id_cols:
            return np.nan
        
        id_col = id_cols[0]
        
        # 尋找融資餘額欄位
        margin_cols = [c for c in df_margin.columns 
                      if ('融資' in c or '餘額' in c) and '券' not in c]
        if not margin_cols:
            return np.nan
        
        margin_col = margin_cols[0]
        
        # 查詢該股
        match_rows = df_margin[df_margin[id_col].astype(str).str.strip().str.replace('"', '') == stock_id]
        if match_rows.empty:
            return np.nan
        
        return clean_numeric_value(match_rows.iloc[0][margin_col])
    except Exception as e:
        logger.debug(f"⚠️  提取融資數值出錯: {e}")
        return np.nan

def extract_foreign_buy_value(df_investor: Optional[pd.DataFrame], stock_id: str) -> float:
    """提取外資買賣超值"""
    if df_investor is None or df_investor.empty:
        return np.nan
    
    try:
        # 尋找代號欄位
        id_cols = [c for c in df_investor.columns if '代號' in c or '股票' in c]
        if not id_cols:
            return np.nan
        
        id_col = id_cols[0]
        
        # 尋找外資買賣超欄位
        foreign_cols = [c for c in df_investor.columns 
                       if ('外資' in c or '外陸資' in c) and '買賣超' in c]
        if not foreign_cols:
            return np.nan
        
        foreign_col = foreign_cols[0]
        
        # 查詢該股
        match_rows = df_investor[df_investor[id_col].astype(str).str.strip().str.replace('"', '') == stock_id]
        if match_rows.empty:
            return np.nan
        
        # 轉換為張數 (股數 / 1000)
        value = clean_numeric_value(match_rows.iloc[0][foreign_col])
        return value / 1000 if not np.isnan(value) else np.nan
    except Exception as e:
        logger.debug(f"⚠️  提取外資數值出錯: {e}")
        return np.nan

def generate_foreign_status(foreign_buy: float) -> str:
    """生成外資籌碼狀態說明"""
    if np.isnan(foreign_buy):
        return "無籌碼資料"
    
    if foreign_buy < 0:
        return f"🔴 外資實質大倒貨 [ 賣超 {abs(foreign_buy):.1f} 張 ] (危險，賣壓沉重)"
    else:
        return f"🟢 外資逆勢吸籌 [ 買超 {foreign_buy:.1f} 張 ] (注意，可能為假跌破)"

def generate_margin_status(margin_now: float, margin_prev: float) -> str:
    """生成融資籌碼狀態說明"""
    if np.isnan(margin_now) or np.isnan(margin_prev):
        return "無籌碼資料"
    
    margin_diff = margin_now - margin_prev
    
    if margin_diff > 0:
        return f"🔴 散戶進房接刀 [ 融資暴增 {int(margin_diff)} 張 ] (浮額過剩，主力出貨給散戶)"
    elif margin_diff < 0:
        return f"🟢 散戶認賠殺出 [ 融資同步減少 {int(abs(margin_diff))} 張 ] (籌碼正進行清洗)"
    else:
        return "⚪ 融資餘額無變動"

# ============================================================================
# 主程式
# ============================================================================
def run_pure_sell_chip_check_strategy(data_dir: str = "./tw_stock_data") -> bool:
    """
    抓出週五純九轉空頭 1K 破位股，並自動融合當日融資與外資賣超數據進行終極對帳
    
    Args:
        data_dir: 數據目錄路徑
        
    Returns:
        bool: 執行是否成功
    """
    try:
        # 確保目錄存在並解析路徑
        data_path = Path(data_dir).expanduser().resolve()
        
        if not data_path.exists():
            logger.error(f"❌ 嚴重錯誤：目錄不存在 '{data_path}'")
            logger.info(f"💡 請建立目錄: {data_path}")
            return False
        
        # 1. 搜尋所有價格檔案
        price_files = sorted(glob.glob(str(data_path / "ALL_BUT_0999_*.csv")))
        
        if not price_files:
            logger.error(f"❌ 錯誤：在 '{data_path}' 路徑下找不到任何 CSV 資料。請確認已下載最新數據。")
            return False
        
        # 提取最新交易日
        last_file = price_files[-1]
        last_market_date = extract_date_from_filename(os.path.basename(last_file))
        
        if not last_market_date:
            logger.error("❌ 無法從檔案名稱解析日期")
            return False
        
        formatted_date = f"{last_market_date[0:4]}-{last_market_date[4:6]}-{last_market_date[6:8]}"
        
        logger.info(f"📡 籌碼對帳引擎啟動 ➔ 定位最後收盤日：【 {formatted_date} 】")
        logger.info(f"🔍 正在篩選【純九轉空頭 1K】個股，並跨表格抽查外資法人與融資動態...")
        logger.info("-" * 95)
        
        all_stock_data = {}
        successful_files = 0
        total_records = 0
        
        # 2. 橫向串接本地端歷史 K 線與每日籌碼
        for i, file in enumerate(price_files, 1):
            date_str = extract_date_from_filename(os.path.basename(file))
            if not date_str:
                continue
            
            try:
                # 讀取價格數據
                df_day = read_price_csv(file)
                if df_day is None or df_day.empty:
                    continue
                
                # 讀取對應的融資與法人報表
                margin_file = data_path / f"MARGIN_{date_str}.csv"
                investor_file = data_path / f"INVESTORS_{date_str}.csv"
                
                df_margin = read_margin_csv(str(margin_file)) if margin_file.exists() else None
                df_investor = read_investor_csv(str(investor_file)) if investor_file.exists() else None
                
                # 價格欄位清洗
                price_cols = ['開盤價', '最高價', '最低價', '收盤價']
                for col in price_cols:
                    if col in df_day.columns:
                        df_day[col] = df_day[col].astype(str).str.replace(',', '')
                        df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
                
                volume_col = '成交股數'
                if volume_col in df_day.columns:
                    df_day[volume_col] = pd.to_numeric(
                        df_day[volume_col].astype(str).str.replace(',', ''),
                        errors='coerce'
                    )
                
                # 遍歷每檔個股
                records_in_file = 0
                for _, row in df_day.iterrows():
                    try:
                        stock_id = str(row['證券代號']).strip().replace('"', '')
                        
                        if not validate_stock_id(stock_id):
                            continue
                        
                        stock_name = str(row['證券名稱']).strip().replace('"', '')
                        
                        # 提取籌碼數據
                        margin_val = extract_margin_value(df_margin, stock_id)
                        foreign_buy_shares = extract_foreign_buy_value(df_investor, stock_id)
                        
                        # 構建記錄
                        record = {
                            'date': date_str,
                            'open': clean_numeric_value(row.get('開盤價')),
                            'high': clean_numeric_value(row.get('最高價')),
                            'low': clean_numeric_value(row.get('最低價')),
                            'close': clean_numeric_value(row.get('收盤價')),
                            'volume': clean_numeric_value(row.get(volume_col)) / 1000 if not np.isnan(clean_numeric_value(row.get(volume_col))) else np.nan,
                            'margin': margin_val,
                            'foreign_buy': foreign_buy_shares
                        }
                        
                        if stock_id not in all_stock_data:
                            all_stock_data[stock_id] = {
                                'name': stock_name,
                                'history': []
                            }
                        
                        all_stock_data[stock_id]['history'].append(record)
                        records_in_file += 1
                        total_records += 1
                    except Exception as e:
                        logger.debug(f"⚠️  處理行記錄出錯: {e}")
                        continue
                
                if records_in_file > 0:
                    successful_files += 1
                
                if i % 10 == 0 or i == len(price_files):
                    logger.debug(f"   已處理 {i}/{len(price_files)} 個檔案...")
                    
            except Exception as e:
                logger.debug(f"⚠️  處理檔案 {file} 出錯: {e}")
                continue
        
        if not all_stock_data or total_records == 0:
            logger.error("❌ 無法讀取任何有效的股票數據")
            return False
        
        logger.info(f"✅ 成功讀取 {successful_files}/{len(price_files)} 個檔案，共 {total_records} 筆記錄")
        
        match_count = 0
        excel_rows = []
        
        # 3. 進行指標比對與籌碼實質對帳
        for stock_id, info in all_stock_data.items():
            try:
                hist_df = pd.DataFrame(info['history'])
                
                if len(hist_df) < 10:
                    continue
                
                hist_df = hist_df.sort_values('date').reset_index(drop=True)
                
                if hist_df.iloc[-1]['date'] != last_market_date:
                    continue
                
                # 計算純九轉
                hist_df = calculate_td_sell_setup(hist_df, close_col='close')
                
                last_idx = hist_df.index[-1]
                prev_idx = hist_df.index[-2]
                row_now = hist_df.loc[last_idx]
                row_prev = hist_df.loc[prev_idx]
                
                # 核心技術條件：週五剛好為空頭破位第一天 (1K)
                if row_now['TD_Count'] == 1:
                    match_count += 1
                    
                    # --- 籌碼面實質增減數據計算 ---
                    foreign_status = generate_foreign_status(row_now['foreign_buy'])
                    margin_status = generate_margin_status(row_now['margin'], row_prev['margin'])
                    
                    # 終端機列印
                    logger.info(f"🔥 [1K空頭破位達標股] ➔ {stock_id} {info['name']}")
                    logger.info(f"   ➔ 週五收盤價：{row_now['close']} 元 | 當日成交量：{int(row_now['volume']) if not np.isnan(row_now['volume']) else 'N/A'} 張")
                    logger.info(f"   ➔ 實戰對帳：{foreign_status}")
                    logger.info(f"   ➔ 散戶動態：{margin_status}")
                    logger.info("-" * 95)
                    
                    # 收集 Excel 行
                    excel_rows.append({
                        "股票代號": stock_id,
                        "股票名稱": info['name'],
                        "週五收盤價(元)": row_now['close'],
                        "當日成交量(張)": int(row_now['volume']) if not np.isnan(row_now['volume']) else "N/A",
                        "外資三大法人對帳狀態": foreign_status,
                        "散戶融資券籌碼動態": margin_status
                    })
                    
            except Exception as e:
                logger.debug(f"⚠️  分析股票 {stock_id} 出錯: {e}")
                continue
        
        # 4. 生成 Excel 報告
        output_filename = f"台股明日放空避險實戰策略報表_{last_market_date}.xlsx"
        
        try:
            if excel_rows:
                output_df = pd.DataFrame(excel_rows)
                output_df.to_excel(output_filename, index=False, engine='openpyxl')
                logger.info(f"\n🎉 報表全自動建立成功 ➔ 【 {output_filename} 】")
                logger.info(f"📊 共計抓出 {match_count} 檔空頭破位股。請優先鎖定雙紅燈（外資大賣+融資大增）的標的作為明日開盤首選。")
            else:
                # 強制備用防線：生成空範本
                template_df = pd.DataFrame(
                    columns=["股票代號", "股票名稱", "週五收盤價(元)", "當日成交量(張)", "外資三大法人對帳狀態", "散戶融資券籌碼動態"]
                )
                template_df.to_excel(output_filename, index=False, engine='openpyxl')
                logger.info(f"\nℹ️  當日無符合個股，已為您生成空的備用報表範本 ➔ 【 {output_filename} 】")
            
            return True
        except Exception as e:
            logger.error(f"❌ 無法生成 Excel 檔案: {e}")
            logger.info("💡 請確認已安裝 openpyxl: pip install openpyxl")
            
            # 備用 CSV 方案
            csv_filename = f"台股明日放空避險實戰策略報表_{last_market_date}.csv"
            try:
                if excel_rows:
                    output_df = pd.DataFrame(excel_rows)
                    output_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                else:
                    pd.DataFrame(
                        columns=["股票代號", "股票名稱", "週五收盤價(元)", "當日成交量(張)", "外資三大法人對帳狀態", "散戶融資券籌碼動態"]
                    ).to_csv(csv_filename, index=False, encoding='utf-8-sig')
                
                logger.info(f"✅ 已改為 CSV 格式儲存 ➔ 【 {csv_filename} 】")
                return True
            except Exception as e2:
                logger.error(f"❌ CSV 保存也失敗: {e2}")
                return False
                
    except KeyboardInterrupt:
        logger.warning("⚠️  程式被使用者中斷")
        return False
    except Exception as e:
        logger.error(f"❌ 發生未預期的錯誤: {e}")
        logger.exception("詳細錯誤堆疊:")
        return False

# ============================================================================
# 進入點
# ============================================================================
if __name__ == "__main__":
    # 自動偵測數據目錄
    default_data_dir = Path.home() / "tw_stock_data"
    
    # 優先使用命令列參數
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    elif default_data_dir.exists():
        data_dir = str(default_data_dir)
    else:
        data_dir = "./tw_stock_data"
    
    logger.info(f"🚀 程式啟動，數據目錄: {data_dir}")
    success = run_pure_sell_chip_check_strategy(data_dir)
    
    # 返回適當的退出碼
    sys.exit(0 if success else 1)
