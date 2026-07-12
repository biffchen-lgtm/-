"""
台灣股票融資券交易數據分析系統
自動化讀取、清洗、分析融資融券變動數據
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
            logging.FileHandler('margin_analysis.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# 數據清洗與轉換
# ============================================================================
def clean_and_convert_numeric(val) -> Optional[float]:
    """
    強力清洗欄位雜質，將引號、逗號、空格及 "--" 全面消滅，強制轉為純數值
    
    Args:
        val: 任意輸入值
        
    Returns:
        float: 清洗後的數值，或 NaN
    """
    if pd.isna(val):
        return np.nan
    
    try:
        val_str = str(val).strip().replace(',', '').replace('"', '').replace(' ', '')
        
        # 快速過濾空值與特殊標記
        if not val_str or val_str in ('', '--', 'NaN', 'nan', 'N/A', 'n/a'):
            return np.nan
        
        return float(val_str)
    except (ValueError, TypeError):
        return np.nan

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    從檔案名稱提取日期 (格式: YYYYMMDD)
    
    Args:
        filename: 檔案名稱 (e.g., "MARGIN_20231215.csv")
        
    Returns:
        str: 格式化的日期字串 (e.g., "2023-12-15") 或 None
    """
    try:
        date_str = filename.split('_')[-1].replace('.csv', '').strip()
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except (IndexError, AttributeError):
        pass
    return None

def validate_stock_id(stock_id: str) -> bool:
    """
    驗證股票代號是否為標準台股 4 碼格式
    
    Args:
        stock_id: 股票代號
        
    Returns:
        bool: 是否為有效的台股代號
    """
    return len(stock_id) == 4 and stock_id.isdigit()

# ============================================================================
# CSV 檔案讀取與欄位偵測
# ============================================================================
def detect_column_name(df: pd.DataFrame, keywords: List[str], exclude_keywords: List[str] = None) -> Optional[str]:
    """
    智慧巡迴偵測欄位名稱
    
    Args:
        df: DataFrame
        keywords: 尋找的關鍵字清單
        exclude_keywords: 排除的關鍵字清單
        
    Returns:
        str: 找到的欄位名稱，或 None
    """
    exclude_keywords = exclude_keywords or []
    
    for col in df.columns:
        col_stripped = col.strip()
        # 檢查是否包含任何關鍵字
        if any(kw in col_stripped for kw in keywords):
            # 檢查是否不包含排除關鍵字
            if not any(kw in col_stripped for kw in exclude_keywords):
                return col_stripped
    
    return None

def read_margin_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    安全讀取融資 CSV 檔案
    
    Args:
        filepath: 檔案路徑
        
    Returns:
        pd.DataFrame: 清洗後的 DataFrame，或 None
    """
    try:
        df = pd.read_csv(filepath, dtype=str, encoding='utf-8')
        # 清洗欄位名稱
        df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
        return df
    except UnicodeDecodeError:
        # 嘗試其他編碼
        try:
            df = pd.read_csv(filepath, dtype=str, encoding='big5')
            df.columns = df.columns.str.strip().str.replace(r'\s+', '', regex=True)
            return df
        except Exception as e:
            logger.warning(f"❌ 讀取檔案失敗 {filepath}: {e}")
            return None
    except Exception as e:
        logger.warning(f"❌ 讀取檔案出錯 {filepath}: {e}")
        return None

# ============================================================================
# 核心分析引擎
# ============================================================================
def process_margin_file(filepath: str, last_market_date: str) -> List[Dict]:
    """
    處理單一融資 CSV 檔案
    
    Args:
        filepath: 檔案路徑
        last_market_date: 最新交易日 (YYYYMMDD)
        
    Returns:
        list: 該日的融資記錄清單
    """
    df_margin = read_margin_csv(filepath)
    if df_margin is None or df_margin.empty:
        return []
    
    # 偵測股票代號欄位
    id_col = detect_column_name(df_margin, ['代號', '股票'])
    if id_col is None:
        logger.debug(f"⚠️  無法在 {filepath} 中找到代號欄位")
        return []
    
    # 偵測融資餘額欄位
    target_margin_col = detect_column_name(
        df_margin,
        ['今日餘額', '融資餘額', '前日餘額'],
        exclude_keywords=['券', '限額']
    )
    if target_margin_col is None:
        logger.debug(f"⚠️  無法在 {filepath} 中找到融資餘額欄位")
        return []
    
    # 偵測股票名稱欄位
    name_col = detect_column_name(df_margin, ['名稱', '股票名'], exclude_keywords=['代號'])
    
    records = []
    date_str = os.path.basename(filepath).split('_')[-1].replace('.csv', '')
    
    for _, row in df_margin.iterrows():
        try:
            stock_id = str(row[id_col]).strip().replace('"', '')
            
            # 驗證股票代號
            if not validate_stock_id(stock_id):
                continue
            
            # 清洗融資數值
            margin_numeric = clean_and_convert_numeric(row[target_margin_col])
            if np.isnan(margin_numeric):
                continue
            
            # 取得股票名稱
            stock_name = "台股個股"
            if name_col:
                stock_name = str(row[name_col]).strip().replace('"', '')
            
            records.append({
                'stock_id': stock_id,
                'stock_name': stock_name,
                'date': date_str,
                'margin_bal': margin_numeric
            })
        except Exception as e:
            logger.debug(f"⚠️  處理行記錄出錯: {e}")
            continue
    
    return records

def calculate_margin_changes(all_records: List[Dict], last_market_date: str) -> List[Dict]:
    """
    計算融資變動並生成分析報告
    
    Args:
        all_records: 所有融資記錄
        last_market_date: 最新交易日
        
    Returns:
        list: 分析報告行數
    """
    # 將記錄轉換為 DataFrame 並按股票分組
    df_all = pd.DataFrame(all_records)
    if df_all.empty:
        return []
    
    report_rows = []
    
    for stock_id, group_df in df_all.groupby('stock_id'):
        try:
            # 排序日期
            group_df = group_df.sort_values('date').reset_index(drop=True)
            
            # 檢查是否至少有 2 筆記錄
            if len(group_df) < 2:
                continue
            
            # 確保最新記錄是最後交易日
            if group_df.iloc[-1]['date'] != last_market_date:
                continue
            
            row_now = group_df.iloc[-1]
            row_prev = group_df.iloc[-2]
            
            margin_now = row_now['margin_bal']
            margin_prev = row_prev['margin_bal']
            
            # 計算增減張數
            margin_diff = margin_now - margin_prev
            
            # 判定籌碼狀態
            if margin_diff > 0:
                status_text = f"🔴 融資暴增 [ ＋{int(margin_diff)} 張 ] (散戶接刀進場)"
            elif margin_diff < 0:
                status_text = f"🟢 融資洗盤 [ 減 {int(abs(margin_diff))} 張 ] (籌碼浮額沉澱)"
            else:
                status_text = "⚪ 融資餘額無變動"
            
            report_rows.append({
                "股票代號": stock_id,
                "股票名稱": row_now['stock_name'],
                "最新融資餘額(張)": int(margin_now),
                "前一日融資餘額(張)": int(margin_prev),
                "融資當日增減變動(張)": int(margin_diff),
                "信用交易籌碼判定標籤": status_text
            })
        except Exception as e:
            logger.debug(f"⚠️  計算股票 {stock_id} 變動時出錯: {e}")
            continue
    
    return report_rows

# ============================================================================
# 主程式
# ============================================================================
def run_margin_database_reader(data_dir: str = "./tw_stock_data") -> bool:
    """
    自動橫向讀取並解體本地端所有交易日的融資券 CSV 檔案，清洗不對齊的官方欄位
    
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
        
        # 1. 搜尋所有融資檔案
        margin_files = sorted(glob.glob(str(data_path / "MARGIN_*.csv")))
        
        if not margin_files:
            logger.error(f"❌ 嚴重錯誤：在 '{data_path}' 路徑下找不到任何 MARGIN_*.csv 融資檔案！")
            logger.info("💡 請先確認 download_all.py 是否有成功將信用交易資料離線打包下載。")
            return False
        
        # 提取最新交易日
        last_file = margin_files[-1]
        last_market_date = os.path.basename(last_file).split('_')[-1].replace('.csv', '')
        formatted_date = f"{last_market_date[0:4]}-{last_market_date[4:6]}-{last_market_date[6:8]}"
        
        logger.info(f"📡 信用交易核心解碼引擎啟動 ➔ 偵測到硬碟共 [ {len(margin_files)} ] 個交易日籌碼。")
        logger.info(f"🔍 正在強力解析最新收盤日 【 {formatted_date} 】 的全台股融資增減明細...")
        logger.info("-" * 95)
        
        # 2. 批量讀取所有檔案
        all_records = []
        successful_files = 0
        
        for i, file in enumerate(margin_files, 1):
            records = process_margin_file(file, last_market_date)
            if records:
                all_records.extend(records)
                successful_files += 1
            
            # 進度指示
            if i % 10 == 0 or i == len(margin_files):
                logger.debug(f"   已處理 {i}/{len(margin_files)} 個檔案...")
        
        if not all_records:
            logger.error("❌ 結束：未偵測到符合的融資數據，請確認歷史 MARGIN_*.csv 內部是否有資料。")
            return False
        
        logger.info(f"✅ 成功讀取 {successful_files}/{len(margin_files)} 個檔案，共 {len(all_records)} 筆記錄")
        
        # 3. 計算融資變動
        report_rows = calculate_margin_changes(all_records, last_market_date)
        
        if not report_rows:
            logger.error("❌ 無法計算融資變動，請檢查數據完整性")
            return False
        
        logger.info(f"✅ 成功計算 {len(report_rows)} 支個股的融資變動")
        
        # 4. 列印核心籌碼警示
        logger.info("\n" + "=" * 95)
        logger.info("🔔 核心籌碼異常警示 🔔")
        logger.info("=" * 95)
        
        for row in report_rows:
            if row["股票代號"] == "9958" or abs(row["融資當日增減變動(張)"]) > 500:
                logger.info(f"📊 [融資查核成功] ➔ {row['股票代號']} {row['股票名稱']}")
                logger.info(f"   ➔ 最新融資總餘額：{row['最新融資餘額(張)']} 張")
                logger.info(f"   ➔ 信用籌碼增減：{row['信用交易籌碼判定標籤']}")
                logger.info("-" * 95)
        
        # 5. 生成 Excel 報告
        output_filename = f"本地端融資資料查核表_{last_market_date}.xlsx"
        result_df = pd.DataFrame(report_rows)
        result_df = result_df.sort_values(by="融資當日增減變動(張)", ascending=False)
        
        try:
            result_df.to_excel(output_filename, index=False, engine='openpyxl')
            logger.info(f"\n🎉 融資大數據庫解析完畢！已成功自動生成查帳 Excel 活頁簿 ➔ 【 {output_filename} 】")
            logger.info(f"📈 分析報告包含 {len(result_df)} 支個股的融資變動數據")
            return True
        except Exception as e:
            logger.error(f"❌ 無法生成 Excel 檔案: {e}")
            logger.info("💡 請確認已安裝 openpyxl: pip install openpyxl")
            
            # 備用方案：生成 CSV
            csv_filename = f"本地端融資資料查核表_{last_market_date}.csv"
            result_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 已改為 CSV 格式儲存 ➔ 【 {csv_filename} 】")
            return True
            
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
    default_data_dir = Path.home() / "tw_stock_data"  # Windows: C:\Users\biff\tw_stock_data
    
    # 優先使用命令列參數
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    elif default_data_dir.exists():
        data_dir = str(default_data_dir)
    else:
        data_dir = "./tw_stock_data"
    
    logger.info(f"🚀 程式啟動，數據目錄: {data_dir}")
    success = run_margin_database_reader(data_dir)
    
    # 返回適當的退出碼
    sys.exit(0 if success else 1)
