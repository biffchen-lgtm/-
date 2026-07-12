"""
台灣股票多頭九轉突破籌碼對帳策略系統
偵測多頭突破訊號並跨表對帳融資、外資動態，自動導出 Excel 報表
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
            logging.FileHandler('buy_strategy.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# 數據清洗與轉換
# ============================================================================
def clean_numeric_value(val, allow_nan: bool = True) -> Optional[float]:
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

def format_date_display(date_str: str) -> str:
    """格式化日期為顯示格式"""
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str

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

def generate_foreign_text(foreign_qty: float) -> str:
    """生成外資籌碼文字描述"""
    if np.isnan(foreign_qty):
        return "無即時法人資料"
    
    if foreign_qty > 0:
        return f"買超 {foreign_qty:.1f} 張"
    elif foreign_qty < 0:
        return f"賣超 {abs(foreign_qty):.1f} 張"
    else:
        return "無變動"

def generate_margin_text(margin_now: float, margin_prev: float) -> str:
    """生成融資籌碼文字描述"""
    if np.isnan(margin_now) or np.isnan(margin_prev):
        return "本地融資欄位未對齊"
    
    margin_diff = margin_now - margin_prev
    
    if margin_diff > 0:
        return f"增加 {int(margin_diff)} 張"
    elif margin_diff < 0:
        return f"減少 {int(abs(margin_diff))} 張"
    else:
        return "無變動"

def calculate_chip_tag(foreign_qty: float, margin_now: float, margin_prev: float) -> str:
    """計算智慧多方籌碼共振標籤"""
    has_foreign_data = not np.isnan(foreign_qty)
    has_margin_data = not (np.isnan(margin_now) or np.isnan(margin_prev))
    
    # 預設標籤
    chip_tag = "🔵 1K 多頭突破股 (籌碼中性)"
    
    if has_foreign_data and foreign_qty > 0:
        if has_margin_data:
            margin_diff = margin_now - margin_prev
            if margin_diff < 0:
                chip_tag = "🔥 黃金共振（外資大買＋融資大減，下週飆股首選！）"
            else:
                chip_tag = "🟢 法人認同股（外資實質放量買超，多方主控）"
        else:
            chip_tag = "🟢 法人認同股（外資實質放量買超，多方主控）"
    elif has_foreign_data and foreign_qty < 0:
        chip_tag = "⚠️ 散戶追高陷阱（外資逆勢大賣超＋融資大增）"
    
    return chip_tag

# ============================================================================
# 主程式
# ============================================================================
def run_pure_buy_excel_strategy(data_dir: str = "./tw_stock_data") -> bool:
    """
    抓出週五多頭九轉 1K 突破股 (收盤價 > 前第 4 天收盤價)，
    智慧融合融資與外資數據，並 100% 強制導出為實體 Excel 報表。
    
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
            logger.error(f"❌ 嚴重錯誤：在 '{data_path}' 路徑下找不到任何量價 K 線檔案！請檢查路徑。")
            return False
        
        # 提取最新交易日
        last_file = price_files[-1]
        last_market_date = extract_date_from_filename(os.path.basename(last_file))
        
        if not last_market_date:
            logger.error("❌ 無法從檔案名稱解析日期")
            return False
        
        formatted_date = format_date_display(last_market_date)
        
        logger.info(f"📡 多頭籌碼對帳引擎啟動 ➔ 鎖定最後收盤日：【 {formatted_date} 】")
        logger.info(f"🔍 正在橫向合併量價、融資與外資法人報表，全力追蹤全台股 Buy 突破標的...")
        logger.info("-" * 95)
        
        all_stock_data = {}
        successful_files = 0
        total_records = 0
        
        # 2. 跨表格大數據橫向整合 (Merge)
        for i, file in enumerate(price_files, 1):
            date_str = extract_date_from_filename(os.path.basename(file))
            if not date_str:
                continue
            
            try:
                # 讀取價格數據
                df_day = read_price_csv(file)
                if df_day is None or df_day.empty:
                    continue
                
                # 讀取信用交易與法人報表
                margin_file = data_path / f"MARGIN_{date_str}.csv"
                investor_file = data_path / f"INVESTORS_{date_str}.csv"
                
                df_margin = read_margin_csv(str(margin_file)) if margin_file.exists() else None
                df_investor = read_investor_csv(str(investor_file)) if investor_file.exists() else None
                
                # 價格資料型態清洗
                if '證券代號' in df_day.columns:
                    df_day['證券代號'] = df_day['證券代號'].astype(str).str.strip().str.replace('"', '')
                
                # 清洗價格欄位
                price_cols = ['開盤價', '最高價', '最低價', '收盤價']
                for col in price_cols:
                    if col in df_day.columns:
                        df_day[col] = df_day[col].astype(str).str.replace(',', '')
                        df_day[col] = pd.to_numeric(df_day[col], errors='coerce')
                
                # 清洗成交股數
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
                        stock_id = str(row.get('證券代號', '')).strip().replace('"', '')
                        
                        if not validate_stock_id(stock_id):
                            continue
                        
                        stock_name = str(row.get('證券名稱', '')).strip().replace('"', '')
                        
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
        
        report_rows = []
        
        # 3. 【多頭 1K 核心邏輯】比對最後一天收盤價 > 前第 4 天收盤價
        for stock_id, info in all_stock_data.items():
            try:
                hist_df = pd.DataFrame(info['history'])
                
                if len(hist_df) < 5:
                    continue
                
                hist_df = hist_df.sort_values('date').reset_index(drop=True)
                
                if hist_df.iloc[-1]['date'] != last_market_date:
                    continue
                
                row_now = hist_df.iloc[-1]
                row_prev = hist_df.iloc[-2]
                row_compare = hist_df.iloc[-5]  # 前第 4 天的 K 線
                
                # 排除數值缺失之個股
                if np.isnan(row_now['close']) or np.isnan(row_compare['close']):
                    continue
                
                # 【Buy 突破條件】今日收盤價 > 前第 4 天收盤價
                if row_now['close'] > row_compare['close']:
                    foreign_qty = row_now['foreign_buy']
                    margin_now = row_now['margin']
                    margin_prev = row_prev['margin']
                    
                    # 格式化籌碼文字描述
                    f_text = generate_foreign_text(foreign_qty)
                    m_text = generate_margin_text(margin_now, margin_prev)
                    chip_tag = calculate_chip_tag(foreign_qty, margin_now, margin_prev)
                    
                    report_rows.append({
                        "股票代號": stock_id,
                        "股票名稱": info['name'],
                        "週五收盤價": row_now['close'],
                        "週五成交量(張)": int(row_now['volume']) if not np.isnan(row_now['volume']) else 0,
                        "外資當日買賣超(張)": f_text,
                        "融資當日變動(張)": m_text,
                        "終極籌碼共振標籤": chip_tag,
                        "技術面價格行為描述": f"最新收盤價 ({row_now['close']}元) 實質突破前第 4 天收盤價 ({row_compare['close']}元)。"
                    })
            except Exception as e:
                logger.debug(f"⚠️  分析股票 {stock_id} 出錯: {e}")
                continue
        
        output_filename = f"台股明日多頭買進實戰策略報表_{last_market_date}.xlsx"
        
        # 4. 【強制降維建表防線】保證 100% 產出實體 Excel 報表
        try:
            if report_rows:
                result_df = pd.DataFrame(report_rows)
                # 依照多方價值排序，把具有「黃金共振」的超級飆股排在最上面
                result_df = result_df.sort_values(by="終極籌碼共振標籤", ascending=False)
                result_df.to_excel(output_filename, index=False, engine='openpyxl')
                logger.info(f"\n🎉 報表全自動建立成功 ➔ 【 {output_filename} 】")
                logger.info(f"🔥 實戰公告：成功精準將全台股共 [ {len(report_rows)} ] 檔多頭起漲突破大軍完全寫入報表！")
                return True
            else:
                logger.warning("⚠️  觸發安全防線：啟動【價格行為強制硬寫入程序】...")
                
                # 備用防線：即使無法完全對帳，也要寫入純 1K 多頭突破股
                backup_list = []
                for stock_id, info in all_stock_data.items():
                    try:
                        hist_df = pd.DataFrame(info['history']).sort_values('date').reset_index(drop=True)
                        
                        if len(hist_df) >= 5 and hist_df.iloc[-1]['date'] == last_market_date:
                            if hist_df.iloc[-1]['close'] > hist_df.iloc[-5]['close']:
                                backup_list.append({
                                    "股票代號": stock_id,
                                    "股票名稱": info['name'],
                                    "週五收盤價": hist_df.iloc[-1]['close'],
                                    "週五成交量(張)": int(hist_df.iloc[-1]['volume']) if not np.isnan(hist_df.iloc[-1]['volume']) else 0,
                                    "外資當日買賣超(張)": "未對齊",
                                    "融資當日變動(張)": "未對齊",
                                    "終極籌碼共振標籤": "純 1K 多頭突破大軍",
                                    "技術面價格行為描述": "最新收盤價突破前第 4 天收盤價。"
                                })
                    except Exception as e:
                        logger.debug(f"⚠️  備用防線處理出錯: {e}")
                        continue
                
                if backup_list:
                    pd.DataFrame(backup_list).to_excel(output_filename, index=False, engine='openpyxl')
                    logger.info(f"🎉 [熔斷機制補救成功] 多方實戰報表已 100% 強制生成 ➔ 【 {output_filename} 】")
                    return True
                else:
                    logger.error("❌ 檢查完畢：本地硬碟內之 CSV 行情資料庫不齊全，請確認 ./tw_stock_data 內存有歷史數據。")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 無法生成 Excel 檔案: {e}")
            logger.info("💡 請確認已安裝 openpyxl: pip install openpyxl")
            
            # 備用 CSV 方案
            csv_filename = f"台股明日多頭買進實戰策略報表_{last_market_date}.csv"
            try:
                if report_rows:
                    output_df = pd.DataFrame(report_rows)
                    output_df = output_df.sort_values(by="終極籌碼共振標籤", ascending=False)
                    output_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                else:
                    # 生成備用清單
                    backup_list = []
                    for stock_id, info in all_stock_data.items():
                        hist_df = pd.DataFrame(info['history']).sort_values('date').reset_index(drop=True)
                        if len(hist_df) >= 5 and hist_df.iloc[-1]['date'] == last_market_date:
                            if hist_df.iloc[-1]['close'] > hist_df.iloc[-5]['close']:
                                backup_list.append({
                                    "股票代號": stock_id,
                                    "股票名稱": info['name'],
                                    "週五收盤價": hist_df.iloc[-1]['close'],
                                    "週五成交量(張)": int(hist_df.iloc[-1]['volume']) if not np.isnan(hist_df.iloc[-1]['volume']) else 0,
                                    "外資當日買賣超(張)": "未對齊",
                                    "融資當日變動(張)": "未對齊",
                                    "終極籌碼共振標籤": "純 1K 多頭突破大軍",
                                    "技術面價格行為描述": "最新收盤價突破前第 4 天收盤價。"
                                })
                    if backup_list:
                        pd.DataFrame(backup_list).to_csv(csv_filename, index=False, encoding='utf-8-sig')
                
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
    success = run_pure_buy_excel_strategy(data_dir)
    
    # 返回適當的退出碼
    sys.exit(0 if success else 1)
