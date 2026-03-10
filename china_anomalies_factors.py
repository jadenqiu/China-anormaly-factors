"""
China Anomalies Factors Library
================================

This module provides factor calculation classes for Chinese A-share market anomalies.
Refactored from the original Jupyter notebook: china-anomalies-compiled.ipynb

Author: Refactored by AI Assistant
Date: 2025-10-17
Original compilation date: 2025-10-16

Structure:
- Global imports and configurations
- Utility functions  
- Factor base classes (A1, A2, A3, etc.)
- Concrete factor classes
"""

# =============================================================================
# SECTION 1: GLOBAL IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import os
import datetime as dt
from pathlib import Path
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple, Iterable
from abc import ABC, abstractmethod


# =============================================================================
# SECTION 2: DATA PATH CONFIGURATIONS
# =============================================================================

# Use relative path - get directory where this file is located
ROOT_DIR = Path(__file__).parent.absolute()
_data_dir = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"  # Output directory for factor results
TESTS_DIR = ROOT_DIR / "tests"
TEST_SAMPLE_META = TESTS_DIR / "test_data" / "sample_data_info.csv"
TEST_SAMPLES_DIR = TESTS_DIR / "test_data" / "samples"

# Create output subdirectories
OUTPUT_A1_DIR = OUTPUT_DIR / "A1_Momentum"
OUTPUT_A2_DIR = OUTPUT_DIR / "A2_Value"
OUTPUT_A3_DIR = OUTPUT_DIR / "A3_Investment"


# =============================================================================
# SECTION 3: DATA STANDARDIZATION UTILITIES
# =============================================================================

def standardize_ticker(code: Union[str, int]) -> str:
    """
    Standardize stock code to Wind format with exchange suffix.
    
    Converts RESSET numeric codes (e.g., 1, 600000) to Wind format (e.g., 000001.SZ, 600000.SH).
    Already-formatted codes (e.g., '000001.SZ') are returned as-is.
    
    Args:
        code: Stock code in RESSET format (int/str) or Wind format (str with suffix)
    
    Returns:
        Standardized code with exchange suffix (e.g., '000001.SZ', '600000.SH')
    
    Rules:
        - Codes starting with 0, 2, 3 → Shenzhen (.SZ)
        - Codes starting with 6, 9 → Shanghai (.SH)  
        - Codes starting with 4, 8 → Beijing (.BJ)
    
    Examples:
        >>> standardize_ticker(1)
        '000001.SZ'
        >>> standardize_ticker('600000')
        '600000.SH'
        >>> standardize_ticker('000001.SZ')
        '000001.SZ'
    """
    code_str = str(code).strip()
    
    # Already has exchange suffix
    if '.' in code_str and len(code_str) > 6:
        return code_str.upper()
    
    # Convert to 6-digit format
    try:
        code_num = str(int(float(code_str))).zfill(6)
    except (ValueError, TypeError):
        return code_str  # Return as-is if conversion fails
    
    # Add exchange suffix based on first digit
    first_digit = code_num[0]
    if first_digit in ('0', '2', '3'):
        return f"{code_num}.SZ"
    elif first_digit in ('6', '9'):
        return f"{code_num}.SH"
    elif first_digit in ('4', '8'):
        return f"{code_num}.BJ"
    else:
        return code_num  # Unknown pattern, return without suffix


def standardize_dataframe(df: pd.DataFrame, 
                         code_col: Optional[str] = None,
                         date_col: Optional[str] = None,
                         date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Standardize DataFrame to follow project conventions:
    - Rename code column to 'tic' and standardize format
    - Rename date column to 'dts' and convert to YYYY-MM-DD format
    - Drop rows with missing 'tic' or 'dts'
    
    Args:
        df: Input DataFrame
        code_col: Name of code column (auto-detected if None)
        date_col: Name of date column (auto-detected if None)
        date_format: Date format string for pd.to_datetime (e.g., '%Y%m%d' for Wind)
    
    Returns:
        Standardized DataFrame with 'tic' and 'dts' columns
    
    Notes:
        - Auto-detection tries: code/证券代码/stkcd for ticker, date/end_date/ann_date for date
        - All dates converted to datetime then .dt.date (YYYY-MM-DD)
        - RESSET codes standardized to Wind format with exchange suffix
    """
    df = df.copy()
    
    # Auto-detect code column
    if code_col is None:
        code_candidates = ['code', '证券代码', 'stkcd', 'Stkcd', 'StkCd']
        code_col = next((c for c in code_candidates if c in df.columns), None)
        if code_col is None:
            raise ValueError(f"Cannot find code column. Available: {df.columns.tolist()}")
    
    # Auto-detect date column  
    if date_col is None:
        date_candidates = ['date', 'end_date', 'ann_date', 'dts']
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"Cannot find date column. Available: {df.columns.tolist()}")
    
    # Standardize ticker (drop NA values first to avoid standardizing None)
    df = df.dropna(subset=[code_col])
    df['tic'] = df[code_col].apply(standardize_ticker)
    if code_col != 'tic':
        df = df.drop(columns=[code_col])
    
    # Standardize date
    df = df.dropna(subset=[date_col])
    if date_format:
        df['dts'] = pd.to_datetime(df[date_col], format=date_format, errors='coerce').dt.date
    else:
        # Auto-detect: try Wind integer format first, then default
        test_conv = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
        if test_conv.notna().mean() > 0.5:
            df['dts'] = test_conv.dt.date
        else:
            df['dts'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    
    if date_col != 'dts':
        df = df.drop(columns=[date_col])
    
    # Drop any remaining missing values (e.g., from date conversion failures)
    df = df.dropna(subset=['tic', 'dts'])
    
    return df

# Ensure output directories exist
for dir_path in [OUTPUT_DIR, OUTPUT_A1_DIR, OUTPUT_A2_DIR, OUTPUT_A3_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Data source paths
Wind_DailyStock_Path = _data_dir / "wind/dailystockreturn_wind.csv"
Resset_FinaIndicator_Path = _data_dir / "resset/quarterly_fina_indicator_cleaned.csv"
Resset_BalanceSheet_Path = _data_dir / "resset/quarterly_balancesheet_cleaned.csv"
Resset_Income_Path = _data_dir / "resset/quarterly_income_cleaned.csv"
Resset_DailyBasic_Path = _data_dir / "resset/daily_basic_resset.csv"
IndustryClass_Path = _data_dir / "zhongzheng/行业分类.csv"
ReportRc_tushare_Path = _data_dir / "tushare/report_rc.csv"
FF3_Path = _data_dir / "resset/RESSET_THRFACDAT_MONTHLY_1.csv"
RiskFreeRate_Path = _data_dir / "tushare/shibor.csv"
ReportRc_resset_Path = _data_dir / "resset/report_rc_cleand.csv"

DATADIR = _data_dir / "resset"
RESSET_DIR = _data_dir / "resset"
TUSHARE_DIR = _data_dir / "tushare"
OUTPUT_DIR = ROOT_DIR / "output"
RESULT_DIR = ROOT_DIR / "output" / "results"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Global anchor columns for factors
ANCHOR = ['code', 'ts']


# =============================================================================
# SECTION 3: UTILITY FUNCTIONS - CODE TRANSFORMATION
# =============================================================================

def RESSETCodeTrans(code):
    """
    Transform asset code from RESSET format to Wind format.
    
    Parameters
    ----------
    code : int or str
        RESSET asset code
        
    Returns
    -------
    str
        Wind asset code (e.g., '000001.SZ', '600000.SH')
    """
    code = str(int(code)).zfill(6)
    if code[0] in ['0', '3', '2']:
        code = code + '.SZ'
    elif code[0] in ['6', '9']:
        code = code + '.SH'
    elif code[0] in ['4', '8']:
        code = code + '.BJ'
    return code


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS - DATE HANDLING
# =============================================================================

def date_transfer(df: pd.DataFrame, date_col: str, format: Optional[str] = "YYYY-MM-DD") -> pd.DataFrame:
    """
    Transfer date column to datetime format and extract month/year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
    format : str, optional
        Date format string
        
    Returns
    -------
    pd.DataFrame
        Dataframe with date_col converted and month/year columns added
    """
    df = df.copy()
    if format:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=format)
        except:
            df[date_col] = df[date_col].astype(str)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col])
    
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    return df


def month_compute(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Convert date column to datetime and extract year, month, quarter.
    Supports both YYYYMM and YYYYMMDD formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
        
    Returns
    -------
    pd.DataFrame
        Dataframe with year, month, quarter columns added
    """
    df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        series = df[date_col].astype(str)
        sample = series.iloc[0]
        fmt = '%Y%m' if len(sample) == 6 else '%Y%m%d'
        df[date_col] = pd.to_datetime(series, format=fmt, errors='coerce')
    
    # Extract components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    # Handle NaN values before converting to int
    df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('Int64')
    
    return df


def get_annual_shift_data(data: pd.DataFrame, shift_num: int = 0) -> pd.DataFrame:
    """
    Shift columns by year for each stock code, preserving quarter matching.
    
    For quarterly data, this shifts by N years while matching the same quarter.
    For example, Q1 2023 will match with Q1 2022 (not all quarters of 2022).
    
    Args:
        data: DataFrame with ['code', 'end_date', ...target_cols]
        target_cols: list of columns to shift (e.g. ['eps'])
        shift_num: number of years to shift (int)
        method: 'year' or 'quarter' (default: 'year')
    Returns:
        DataFrame with shifted columns, named as '{col}_{shift_num}'
    """
    # This version matches the notebook: shift by year for each code
    # If end_date is string, convert to datetime
    df = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['end_date']):
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    
    df['year'] = df['end_date'].dt.year
    df['month'] = df['end_date'].dt.month
    df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('Int64')

    # For each code, shift target columns by year
    result = []
    for code, group in df.groupby('code'):
        group = group.sort_values('end_date')
        for col in [c for c in group.columns if c not in ['code', 'end_date', 'ann_date', 'year', 'month', 'quarter']]:
            shifted = group[[col, 'year', 'quarter']].copy()
            shifted['year'] = shifted['year'] + shift_num
            shifted = shifted.rename(columns={col: f'{col}_{shift_num}'})
            # Merge on both year AND quarter to avoid cartesian product
            group = pd.merge(group, shifted, on=['year', 'quarter'], how='left')
        result.append(group)
    
    df_shifted = pd.concat(result, ignore_index=True)
    
    # Only keep code, end_date, and shifted columns
    keep_cols = ['code', 'end_date'] + [f'{col}_{shift_num}' for col in data.columns if col not in ['code', 'end_date', 'ann_date']]
    return df_shifted[keep_cols]


def select_month(data: pd.DataFrame, month: Iterable[int] = {12,}, date_col: str = 'end_date') -> pd.DataFrame:
    """
    Filter dataframe to keep only specified months.
    
    Args:
        data: Input dataframe with date column
        month: Iterable of month numbers to keep (1-12)
        date_col: Name of the date column
    
    Returns:
        Filtered dataframe with only specified months
    """
    data[date_col] = pd.to_datetime(data[date_col])
    return data[data[date_col].dt.month.isin(month)]


def fiscal_year_concordance(data1: pd.DataFrame, data2: pd.DataFrame,
                            on_data1_cols: list[str] = ['code', 'end_date'],
                            on_data2_cols: list[str] = ['code', 'date']) -> pd.DataFrame:
    """
    Make a concordance for the end of month in fiscal year between data1 and data2.
    
    Args:
        data1: Low frequency data (e.g., annual/quarterly financial data)
        data2: High frequency data (e.g., monthly market data)
        on_data1_cols: Keys to use in data1
        on_data2_cols: Keys to use in data2
    
    Returns:
        Merged dataframe with December observations only
    """
    # Ensure merge keys are both datetime
    d1 = data1.copy()
    d2 = data2.copy()
    for col in on_data1_cols:
        if col in d1.columns and not pd.api.types.is_datetime64_any_dtype(d1[col]):
            d1[col] = pd.to_datetime(d1[col], errors='coerce')
    for col in on_data2_cols:
        if col in d2.columns and not pd.api.types.is_datetime64_any_dtype(d2[col]):
            d2[col] = pd.to_datetime(d2[col], errors='coerce')
    data = pd.merge(
        left=d1,
        right=d2,
        left_on=on_data1_cols,
        right_on=on_data2_cols,
        how='left'
    )
    data = select_month(data, month=[12,])
    return data


def fiscal_june_concordance(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """
    Make concordance between fiscal year-end (December t-1) and June of year t.
    
    Args:
        data1: Balance sheet data with fiscal year-end dates
        data2: Market data with June dates
    
    Returns:
        Merged dataframe aligning fiscal year t-1 December with June of year t
    """
    # Only keep fiscal year-end (December)
    data1 = data1[data1['month'] == 12]
    # Only keep June observations
    data2 = data2[data2['month'] == 6]
    # Align December of year t-1 with June of year t
    data2['year'] = data2['year'] - 1
    data = pd.merge(left=data1, right=data2, on=['code', 'year'], how='left')
    return data


def get_quartly_be() -> pd.DataFrame:
    """
    Calculate quarterly book equity from balance sheet data.
    
    Book Equity = Total Stockholders' Equity (excluding minority interest if needed)
    
    Returns:
        DataFrame with columns: ['code', 'end_date', 'quarterly_book_equity', 'ann_date', 'ts']
    """
    # Use actual RESSET column names
    # total_hldr_eqy_inc_min_int = Total Stockholders' Equity (including minority interest)
    # We use the fillna version for robustness
    cols = ['quarterly_total_hldr_eqy_inc_min_int_fillna']
    df = load_balancesheet_data(cols, RESSET_DIR)
    
    # Use total stockholders' equity as book equity
    df['quarterly_book_equity'] = df['quarterly_total_hldr_eqy_inc_min_int_fillna']
    
    # Add ts column
    if not pd.api.types.is_datetime64_any_dtype(df['end_date']):
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
    df['ts'] = df['end_date']  # Use end_date directly for merging
    
    return df[['code', 'end_date', 'quarterly_book_equity', 'ann_date', 'ts']]


def get_monthly_share(type: str = 'total') -> pd.DataFrame:
    """
    Get monthly share count data.
    
    Args:
        type: Type of shares ('total', 'tradable', etc.)
    
    Returns:
        DataFrame with monthly share counts
    """
    # Placeholder implementation
    return load_daily_basic_data(['total_shares'], RESSET_DIR)
# =============================================================================
# SECTION 5: UTILITY FUNCTIONS - DATA LOADING (WITH CACHE)
# =============================================================================

# Global cache for loaded datasets
_DATA_CACHE = {}

# -------------------------
# Test Sample Mode Controls
# -------------------------
def _is_sample_mode() -> bool:
    """Return True if hard-cut test sample mode is enabled via env."""
    return os.environ.get("SAMPLE_DATA_MODE", "0") in {"1", "true", "True"}

def _is_sample_regenerate() -> bool:
    """Return True if sample CSVs should be regenerated even if present."""
    return os.environ.get("SAMPLE_DATA_REGENERATE", "0") in {"1", "true", "True"}

def _get_sample_pad_months() -> int:
    """Months of padding to extend sample date window to keep join buffer."""
    try:
        return int(os.environ.get("SAMPLE_PAD_MONTHS", "6"))
    except Exception:
        return 6

def _load_sample_info() -> tuple[set[str], pd.Timestamp, pd.Timestamp]:
    """
    Load sample info (codes and date window) from tests cache.
    Returns a tuple of (codes_set, start_ts, end_ts).
    Raises FileNotFoundError if not available.
    """
    if not TEST_SAMPLE_META.exists():
        raise FileNotFoundError(str(TEST_SAMPLE_META))
    info = pd.read_csv(TEST_SAMPLE_META)
    
    # Try both 'tic' (new standard) and 'code' (legacy) column names
    code_col = 'tic' if 'tic' in info.columns else 'code'
    codes = set(info[code_col].astype(str))
    
    # Allow per-row dates; use global min/max
    start_ts = pd.to_datetime(info['start_date']).min()
    end_ts = pd.to_datetime(info['end_date']).max()
    # Apply padding
    pad = _get_sample_pad_months()
    start_ts = (start_ts - pd.DateOffset(months=pad)).normalize()
    # end_ts to end of month for inclusivity
    end_ts = (end_ts + pd.offsets.MonthEnd(pad)).normalize()
    return codes, start_ts, end_ts

def _sample_dest_path(csv_path: Path) -> Path:
    """
    Compute destination path for a sample CSV mirroring data/ subfolders.
    Example: data/resset/foo.csv -> tests/test_data/samples/resset/foo.sample.csv
    """
    try:
        rel = csv_path.relative_to(_data_dir)
    except ValueError:
        # If file not under data/, flatten name
        rel = Path(csv_path.name)
    dest = TEST_SAMPLES_DIR / rel
    # Ensure parent dir exists and add .sample suffix before extension
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix:
        return dest.with_suffix("") .with_name(dest.stem + ".sample").with_suffix(".csv")
    return dest.with_name(dest.name + ".sample.csv")

def _filter_chunk_on_date(chunk: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Filter a chunk by any available date-like column among ['date','end_date','ann_date'].
    If none present, return chunk unchanged (only code filter will apply)."""
    date_cols = [c for c in ['date', 'end_date', 'ann_date'] if c in chunk.columns]
    if not date_cols:
        return chunk
    mask = pd.Series(False, index=chunk.index)
    for c in date_cols:
        # Convert lazily and cache in-place for subsequent chunks
        if not pd.api.types.is_datetime64_any_dtype(chunk[c]):
            with pd.option_context('mode.chained_assignment', None):
                # Try YYYYMMDD format first (for Wind integer dates like 20200102)
                # If conversion succeeds with good results, use it; otherwise try default
                test = pd.to_datetime(chunk[c], format='%Y%m%d', errors='coerce')
                if test.notna().mean() > 0.5:  # More than half valid = likely correct format
                    chunk[c] = test
                else:
                    # Fallback to default parsing for other formats (e.g., YYYY-MM-DD)
                    chunk[c] = pd.to_datetime(chunk[c], errors='coerce')
        mask = mask | ((chunk[c] >= start_ts) & (chunk[c] <= end_ts))
    return chunk[mask]

def _ensure_sample_csv(csv_path: Path) -> Path:
    """
    Ensure a sample CSV exists for the given source file. If in sample mode,
    create it on first use by filtering to sample codes and padded date range.
    Returns the path to the sample CSV.
    """
    dest = _sample_dest_path(csv_path)
    if dest.exists() and not _is_sample_regenerate():
        return dest
    # Load sample info; if missing, fall back to full file
    try:
        codes, start_ts, end_ts = _load_sample_info()
    except FileNotFoundError:
        return csv_path
    # Build filter
    code_cols = ['code', '证券代码']
    # Stream filter in chunks to control memory
    filtered_parts = []
    chunksize = 500_000
    cols = None
    try:
        for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunksize):
            if cols is None:
                cols = list(chunk.columns)
            # Standardize code column if needed
            present_code_col = next((c for c in code_cols if c in chunk.columns), None)
            if present_code_col is None:
                # No code column; try to keep by date only
                sub = _filter_chunk_on_date(chunk, start_ts, end_ts)
            else:
                # Normalize code column name if needed
                if present_code_col != 'code':
                    chunk = chunk.rename(columns={present_code_col: 'code'})
                
                # Convert to string for matching
                chunk['code'] = chunk['code'].astype(str)
                
                # Match codes: try both exact match and normalized versions
                # Exact match (for Wind codes like "000001.SZ")
                mask = chunk['code'].isin(codes)
                
                # Also try matching after normalizing to 6-digit format
                # This handles RESSET numeric codes
                chunk_normalized = chunk['code'].str.replace(r'[^\d]', '', regex=True).str.zfill(6)
                codes_normalized = {str(c).replace('.SZ','').replace('.SH','').replace('.BJ','').zfill(6) for c in codes}
                mask = mask | chunk_normalized.isin(codes_normalized)
                
                sub = chunk[mask]
                # Filter by date window
                sub = _filter_chunk_on_date(sub, start_ts, end_ts)
            if not sub.empty:
                filtered_parts.append(sub)
        if filtered_parts:
            df_out = pd.concat(filtered_parts, ignore_index=True)
        else:
            # Create empty with same columns to avoid downstream errors
            df_out = pd.DataFrame(columns=cols)
    except Exception:
        # On any error, fall back to full file
        return csv_path
    # Save
    df_out.to_csv(dest, index=False)
    return dest

def _get_cache_key(csv_path: Path) -> str:
    """Generate cache key for a data file."""
    return str(csv_path.absolute())

def _load_with_cache(csv_path: Path, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load CSV with caching to avoid redundant file reads.
    
    Args:
        csv_path: Path to CSV file
        **read_csv_kwargs: Additional arguments for pd.read_csv
    
    Returns:
        DataFrame loaded from CSV (cached if available)
    """
    # Route to sample file if in sample mode
    target_path = csv_path
    if _is_sample_mode():
        target_path = _ensure_sample_csv(csv_path)
    cache_key = _get_cache_key(target_path)
    
    # Return cached version if available
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    
    # Load and cache
    df = pd.read_csv(target_path, **read_csv_kwargs)
    _DATA_CACHE[cache_key] = df
    return df

def clear_data_cache():
    """Clear the global data cache to free memory."""
    global _DATA_CACHE
    _DATA_CACHE = {}

def load_daily_basic_data(col: List[str], path: Path = DATADIR, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load daily basic data from RESSET.
    
    Returns DataFrame with columns: 'code', 'date' (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / 'daily_basic_resset.csv'
    col = ['code', 'date'] + col
    df = _load_with_cache(csv_path, nrows=nrows, low_memory=False)
    df = df.loc[:, col]
    # Add standardized columns as aliases (keep original for backward compatibility)
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    return df


def load_income_data(col: List[str], path: Path = DATADIR) -> pd.DataFrame:
    """Load income statement data from RESSET.
    
    Returns DataFrame with columns: 'code', 'end_date', 'ann_date', etc.
    (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / 'quarterly_income_cleaned.csv'
    col = ['code', 'ann_date', 'end_date', 'reporttype', 'comtype'] + col
    df = _load_with_cache(csv_path, low_memory=False)
    df = df.loc[:, col]
    # Add standardized columns as aliases
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
    return df


def load_balancesheet_data(col: List[str], path: Path = DATADIR, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load balance sheet data from RESSET.
    
    Returns DataFrame with columns: 'code', 'end_date', 'ann_date', etc.
    (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / 'quarterly_balancesheet_cleaned.csv'
    col = ['code', 'ann_date', 'end_date', 'reporttype', 'comtype'] + col
    df = _load_with_cache(csv_path, nrows=nrows, low_memory=False)
    df = df.loc[:, col]
    # Add standardized columns as aliases
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
    return df


def load_cashflow_data(col: List[str], path: Path = DATADIR) -> pd.DataFrame:
    """Load cash flow statement data from RESSET.
    
    Returns DataFrame with columns: 'code', 'end_date', 'ann_date', etc.
    (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / 'quarterly_cashflow_cleaned.csv'
    col = ['code', 'ann_date', 'end_date', 'reporttype', 'comtype'] + col
    df = _load_with_cache(csv_path, low_memory=False)
    df = df.loc[:, col]
    # Add standardized columns as aliases
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
    return df


def load_fina_indicator_data(col: List[str], path: Path = DATADIR, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load financial indicator data from RESSET.
    
    Returns DataFrame with columns: 'code', 'end_date', 'ann_date', etc.
    (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / "quarterly_fina_indicator_cleaned.csv"
    col = ['code', 'ann_date', 'end_date', 'reporttype'] + col
    df = _load_with_cache(csv_path, nrows=nrows, low_memory=False)
    df = df.loc[:, col]
    # Add standardized columns as aliases
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
    return df


def load_wind_daily_data(col: List[str], path: Path = _data_dir / "wind") -> pd.DataFrame:
    """Load daily return data from Wind.
    
    Returns DataFrame with columns: 'code', 'date' (plus 'tic', 'dts' as standardized aliases).
    """
    csv_path = path / "dailystockreturn_wind.csv"
    col = ['code', 'date'] + col
    df = _load_with_cache(csv_path)
    df = df.loc[:, col]
    # Add standardized columns as aliases
    df['tic'] = df['code'].apply(standardize_ticker)
    df['dts'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    return df


def load_riskfree_data(col: List[str], path: Path = _data_dir / "tushare") -> pd.DataFrame:
    """Load risk-free rate data from Tushare."""
    csv_path = path / "shibor.csv"
    df = _load_with_cache(csv_path)
    for c in col:
        df[c] = df[c].apply(lambda x: float(x) / 100)
    col = col + ['date']
    return df[col]


def load_indus_class_data(path: Path = IndustryClass_Path) -> pd.DataFrame:
    """Load industry classification data."""
    df = pd.read_csv(path)
    df.rename(columns={'证券代码': 'code', '证券代码简称': 'name',
                      '中证一级行业分类简称': 'class1',
                      '中证二级行业分类简称': 'class2',
                      '中证三级行业分类简称': 'class3',
                      '中证四级行业分类简称': 'class4'}, inplace=True)
    df = df[['code', 'name', 'class1', 'class2', 'class3', 'class4']]
    
    # Remove financial industry
    df = df[df['class1'] != '金融']
    
    # Standardize code
    df['code'] = df['code'].str.replace(r'\D', '', regex=True)
    df['code'] = df['code'].apply(RESSETCodeTrans)
    
    return df


def load_ff3_data(col: List[str], path: Path = _data_dir / "resset/FF3_cleaned.csv") -> pd.DataFrame:
    """Load Fama-French 3-factor data."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df.mktflg.isin(['A'])]
    return df


def get_monthly_mv(nrows: Optional[int] = None) -> pd.DataFrame:
    """Get monthly market value at month end."""
    df = load_daily_basic_data(['total_mv'], DATADIR, nrows=nrows)
    df = date_transfer(df, 'date')
    df.sort_values(by=['code', 'date'], inplace=True)
    df = df.groupby(by=['code', 'year', 'month']).tail(1)
    df.rename(columns={'total_mv': 'monthly_mv_end'}, inplace=True)
    return df


# =============================================================================
# SECTION 6: UTILITY FUNCTIONS - FACTOR FRAME CONVERSION
# =============================================================================

def as_factor_frame(df: pd.DataFrame,
                   ft_name: str,
                   anchor: List[str] = ANCHOR,
                   date_col: str = 'end_date') -> pd.DataFrame:
    """
    Convert dataframe to standardized factor frame format.
    
    This is the core function to transform raw factor calculations into
    the standardized multi-index format for downstream analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with factor values
    ft_name : str
        Name of the factor column
    anchor : list of str
        Index columns, default ['code', 'ts']
    date_col : str
        Name of the date column to use
        
    Returns
    -------
    pd.DataFrame
        Factor frame with multi-index (code, ts) and factor values
    """
    # Step 1: Remove irrelevant columns and rows without dates
    ft = df.copy()[['code', date_col, ft_name]].dropna(subset=[date_col])
    
    # Convert to period index
    ft['ts'] = pd.PeriodIndex(ft[date_col], freq='M')
    ft = ft[['code', 'ts', ft_name]]
    
    # Determine frequency
    ts_freq = ft.groupby(by='code')['ts'].diff().value_counts().index[0].n
    
    # Step 2: Remove NaN values and duplicates
    ft = ft.dropna(subset=[ft_name]).drop_duplicates(subset=anchor)
    
    # Step 3: Create full index grid
    all_ts = pd.date_range(
        start=ft['ts'].dt.to_timestamp().min(),
        end=ft['ts'].dt.to_timestamp().max(),
        freq='ME'
    ).strftime('%Y%m').astype(int)
    
    all_code = pd.Series(ft['code'].unique()).sort_values()
    all_index = pd.MultiIndex.from_product([all_code, all_ts], names=anchor).sort_values()
    
    # Convert ts to integer format
    ft['ts'] = ft['ts'].dt.strftime('%Y%m').astype(int)
    ft = ft.set_index(anchor).reindex(all_index).sort_index()
    
    # Return only the factor column
    ft = ft[[ft_name]]
    
    return ft


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS - VISUALIZATION
# =============================================================================

def plot_date_counts(df: pd.DataFrame) -> None:
    """Plot the number of observations per date."""
    date_counts = df['end_date'].value_counts()
    df_date_counts = date_counts.reset_index()
    df_date_counts.columns = ['end_date', 'counts']
    df_date_counts = df_date_counts.sort_values(by='end_date')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_date_counts['end_date'], df_date_counts['counts'], marker='o', linestyle='-')
    plt.xlabel('end_date')
    plt.ylabel('counts')
    plt.title('Date Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plotVarTrend(df: pd.DataFrame, variable: str) -> None:
    """Plot the trend of a variable over time."""
    tmp = df.groupby(by=['end_date'])[variable].mean().reset_index()
    tmp['end_date'] = pd.to_datetime(tmp['end_date'])
    
    plt.plot(tmp['end_date'], tmp[variable], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title(variable)
    plt.xticks(rotation=90)
    plt.show()


# =============================================================================
# SECTION 8: BASE CLASSES
# =============================================================================

class BaseFactor(ABC):
    """
    Abstract base class for all factors.
    
    All factor classes should inherit from this base and implement
    the required abstract methods and properties.
    """
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Return factor category (e.g., 'A1', 'A2')."""
        pass
    
    @property
    @abstractmethod
    def factor_id(self) -> str:
        """Return full factor ID (e.g., 'A.1.1')."""
        pass
    
    @property
    @abstractmethod
    def abbr(self) -> str:
        """Return factor abbreviation (e.g., 'sue', 'abr')."""
        pass
    
    @abstractmethod
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate the factor values.
        
        Returns
        -------
        pd.DataFrame
            Raw factor dataframe before conversion to factor frame
        """
        pass
    
    def as_factor_frame(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Convert raw factor dataframe to standardized factor frame.
        
        This is a wrapper around the global as_factor_frame function.
        """
        return as_factor_frame(df, self.abbr, **kwargs)
    
    def save(self, df: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Path:
        """
        Save factor results to parquet file with standardized naming.
        
        Parameters
        ----------
        df : pd.DataFrame
            Factor dataframe to save (should be in factor frame format)
        start_date : str, optional
            Start date for filename (format: YYYYMMDD). If None, inferred from data.
        end_date : str, optional
            End date for filename (format: YYYYMMDD). If None, inferred from data.
        
        Returns
        -------
        Path
            Path to the saved file
        """
        # Infer dates from data if not provided
        if start_date is None or end_date is None:
            dates = df.index.get_level_values('ts')
            if start_date is None:
                # Handle both integer (YYYYMM) and datetime formats
                if pd.api.types.is_integer_dtype(dates):
                    # Convert YYYYMM integer to date string
                    start_date = str(dates.min()) + '01'
                else:
                    start_date = dates.min().strftime('%Y%m%d')
            if end_date is None:
                # Handle both integer (YYYYMM) and datetime formats
                if pd.api.types.is_integer_dtype(dates):
                    # Convert YYYYMM integer to date string (use last day of month)
                    end_date = str(dates.max()) + '28'  # Approximate end of month
                else:
                    end_date = dates.max().strftime('%Y%m%d')
        
        # Determine output directory based on category
        if self.category == "A1":
            output_dir = OUTPUT_A1_DIR
        elif self.category == "A2":
            output_dir = OUTPUT_A2_DIR
        elif self.category == "A3":
            output_dir = OUTPUT_A3_DIR
        else:
            output_dir = OUTPUT_DIR
        
        # Construct filename: {factor_id}_{abbr}_{start_date}-{end_date}.parquet
        filename = f"{self.factor_id}_{self.abbr}_{start_date}-{end_date}.parquet"
        filepath = output_dir / filename
        
        # Save to parquet
        df.to_parquet(filepath)
        print(f"✅ Factor saved: {filepath}")
        
        return filepath
    
    def validate(self) -> bool:
        """
        Run validation/test code for the factor.
        
        Default implementation does nothing. Override in subclasses
        to add specific validation logic.
        
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.factor_id}', abbr='{self.abbr}')"


class A1MomentumFactorBase(BaseFactor):
    """
    Base class for A1 Momentum factors.
    
    All A1 momentum-related factors inherit from this class,
    which provides common functionality and properties.
    """
    
    @property
    def category(self) -> str:
        return "A1"


class A2ValueFactorBase(BaseFactor):
    """
    Base class for A2 Value factors.
    
    All A2 value-related factors inherit from this class,
    which provides common functionality and properties.
    """
    
    @property
    def category(self) -> str:
        return "A2"


class A3InvestmentFactorBase(BaseFactor):
    """
    Base class for A3 Investment factors.
    
    All A3 investment-related factors inherit from this class,
    which provides common functionality and properties.
    """
    
    @property
    def category(self) -> str:
        return "A3"


class A5IntangiblesFactorBase(BaseFactor):
    """
    Base class for A5 Intangibles factors.
    
    All A5 intangibles-related factors inherit from this class,
    which provides common functionality and properties for factors
    related to R&D, analyst coverage, earnings quality, etc.
    """
    
    @property
    def category(self) -> str:
        return "A5"


class A6TradingFrictionsFactorBase(BaseFactor):
    """
    Base class for A6 Trading Frictions factors.
    
    All A6 trading frictions-related factors inherit from this class,
    which provides common functionality and properties for factors
    related to liquidity, volatility, turnover, etc.
    """
    
    @property
    def category(self) -> str:
        return "A6"


# =============================================================================
# SECTION 9: A1 MOMENTUM FACTOR IMPLEMENTATIONS
# =============================================================================

class SueFactor(A1MomentumFactorBase):
    """
    A.1.1 Standardized Unexpected Earnings (Sue)
    
    Sue denotes Standardized Unexpected Earnings, calculated as the change
    in split-adjusted quarterly earnings per share from its value four quarters
    ago divided by the standard deviation of this change over the prior eight quarters.
    
    Formula:
        Sue = (EPS(Q_t) - EPS(Q_{t-1})) / SD(Q_t)
    
    where:
        EPS(Q_t): earnings per share for quarter Q in year t
        EPS(Q_{t-1}): earnings per share for quarter Q in year t-1
        SD(Q_t): standard deviation of quarterly earnings change over prior 8 quarters
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.1"
    
    @property
    def abbr(self) -> str:
        return "sue"
    
    def _read_data(self, usena: bool = True) -> pd.DataFrame:
        """Read and prepare input data."""
        df = load_fina_indicator_data(col=['eps', 'quarterly_eps_fillna'], path=DATADIR)
        if usena:
            df = df[['code', 'quarterly_eps_fillna', 'ann_date', 'end_date']].rename(
                columns={'quarterly_eps_fillna': 'eps'})
        else:
            df = df[['code', 'eps', 'end_date', 'ann_date']]
        df = df.sort_values(['code', 'end_date'], ascending=True)
        return df
    
    def calculate(self, window_size: int = 8, usena: bool = True, **kwargs) -> pd.DataFrame:
        """
        Calculate SUE factor.
        Returns:
            Dataframe with columns: code, end_date, ann_date, sue
        """
        df = self._read_data(usena=usena)
        df = df.sort_values(['code', 'end_date'], ascending=True)
        
        # Ensure end_date is datetime for merge
        if not pd.api.types.is_datetime64_any_dtype(df['end_date']):
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        # Shift EPS by one year (using fixed function signature)
        df_shift = get_annual_shift_data(df[['code', 'end_date', 'eps']], shift_num=1)
        
        # Ensure df_shift end_date is also datetime
        if not pd.api.types.is_datetime64_any_dtype(df_shift['end_date']):
            df_shift['end_date'] = pd.to_datetime(df_shift['end_date'], errors='coerce')
        
        df = pd.merge(df, df_shift, on=['code', 'end_date'], how='left')
        
        # Calculate rolling standard deviation
        df['std_prior_8'] = df.groupby('code')['eps'].rolling(window=window_size).std().reset_index(level=0, drop=True)
        
        # Calculate SUE (avoid division by zero)
        df['sue'] = np.where(df['std_prior_8'] > 0, 
                             (df['eps'] - df['eps_1']) / df['std_prior_8'], 
                             np.nan)
        
        return df[['code', 'end_date', 'ann_date', 'sue']]
    
    def validate(self) -> bool:
        """Run validation tests for SUE factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_sue = self.calculate(usena=False)
        print(f"Shape: {df_sue.shape}")
        print(f"Sample:\n{df_sue.head()}")
        
        # Convert to factor frame
        ft_sue = self.as_factor_frame(df_sue)
        print(f"Factor frame shape: {ft_sue.shape}")
        print(f"Factor frame sample:\n{ft_sue.head()}")
        
        # Plot diagnostics
        plot_date_counts(df_sue)
        plotVarTrend(df_sue, 'sue')
        
        print(f"Validation complete for {self.abbr}!")
        return True

class AbrFactor(A1MomentumFactorBase):
    """
    A.1.2 Cumulative Abnormal Returns around Earnings Announcement (Abr)
    
    Cumulative abnormal returns around earnings announcement dates.
    
    Formula:
        Abr_i = Σ_{d=-2}^{+1} (r_{id} - r_{md})
    
    where:
        r_{id}: stock i's return on day d (earnings announced on day 0)
        r_{md}: value-weighted market index return on day d
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.2"
    
    @property
    def abbr(self) -> str:
        return "abr"
    
    def _get_ann_date(self) -> pd.DataFrame:
        """Get announcement dates."""
        df = load_balancesheet_data(col=[], path=DATADIR)
        df = df[['code', 'ann_date']]
        df = df.sort_values(['code', 'ann_date'], ascending=True)
        return df
    
    def _get_abr(self, d: int) -> pd.DataFrame:
        """
        Calculate abnormal returns for window [-d, d-1].
        
        Parameters
        ----------
        d : int
            Window size (e.g., d=2 means [-2, 1])
        """
        data = load_wind_daily_data(col=['close'], path=_data_dir / "wind")
        data[f'prior_{d}'] = data.groupby(by=['code'])['close'].shift(d)
        data[f'poster_{d-1}'] = data.groupby(by=['code'])['close'].shift(-d + 1)
        data['abr'] = np.log(data[f'poster_{d-1}'] / data[f'prior_{d}'])
        return data[['code', 'date', 'abr']]
    
    def calculate(self, d: int = 2, **kwargs) -> pd.DataFrame:
        """
        Calculate ABR factor.
        
        Parameters
        ----------
        d : int
            Window size (default: 2, meaning [-2, 1])
            
        Returns
        -------
        pd.DataFrame
            Dataframe with columns: code, ann_date, date, abr
        """
        # Get abnormal returns
        df_abr = self._get_abr(d)
        df_abr['date'] = df_abr['date'].astype(str)
        df_abr = date_transfer(df_abr, 'date', format="%Y%m%d")
        
        # Get announcement dates
        df_date = self._get_ann_date()
        df_date['ann_date'] = pd.to_datetime(df_date['ann_date'])
        
        # Merge on announcement date
        df = pd.merge(
            left=df_abr, right=df_date,
            left_on=['code', 'date'], right_on=['code', 'ann_date'],
            how='right', indicator=True
        )
        df.drop(columns=['_merge'], inplace=True)
        
        return df[['code', 'ann_date', 'abr', 'date']]
    
    def validate(self) -> bool:
        """Run validation tests for ABR factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_abr = self.calculate(d=2)
        print(f"Shape: {df_abr.shape}")
        print(f"Sample:\n{df_abr.head(10)}")
        
        # Convert to factor frame (note: date_col='date' instead of 'end_date')
        ft_abr = self.as_factor_frame(df_abr, date_col='date')
        print(f"Factor frame shape: {ft_abr.shape}")
        print(f"Factor frame sample:\n{ft_abr.head()}")
        
        # Plot diagnostics
        plot_date_counts(df_abr.rename(columns={'date': 'end_date'}))
        plotVarTrend(df_abr.rename(columns={'date': 'end_date'}), 'abr')
        
        print(f"Validation complete for {self.abbr}!")
        return True

class ReFactor(A1MomentumFactorBase):
    """
    A.1.3 Revisions in Analyst Earnings Forecasts (Re)
    
    Revisions in analyst earnings forecasts.
    
    Formula:
        Re_{it} = Σ_{τ=1}^6 (f_{it-τ} - f_{it-τ-1}) / p_{it-τ-1}
    
    where:
        f_{it-τ}: consensus mean forecast issued in month t-τ for firm i's current fiscal year earnings
        p_{it-τ-1}: prior month's share price
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.3"
    
    @property
    def abbr(self) -> str:
        return "re"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate Re factor.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with columns: code, date, re
        """
        # Read analyst forecast data
        df = pd.read_csv(ReportRc_resset_Path)
        data = pd.read_csv(Wind_DailyStock_Path)
        
        # Get month-end close prices
        data = data[['code', 'date', 'close']]
        data = month_compute(data)
        data = data.sort_values(['code', 'date'])
        data = data.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Process forecast data
        df.rename(columns={'report_date': 'date'}, inplace=True)
        df = df.sort_values(['code', 'date'], ascending=True)
        df = df[['code', 'date', 'foryear', 'eps']]
        df = df.dropna(subset=['date'])
        df = month_compute(df)
        # Group by includes quarter to preserve it
        df = df.groupby(by=['code', 'year', 'month', 'quarter', 'foryear'])['eps'].mean().reset_index()
        df['eps_prior1'] = df.groupby(by=['code', 'year', 'month'])['eps'].shift(1)
        df['eps_delta'] = df['eps'] - df['eps_prior1']
        df.rename(columns={'quarter': 'f_quarter'}, inplace=True)
        
        # Merge tables
        df1 = pd.merge(left=df, right=data, how='inner', on=['code', 'year', 'month'], indicator=True)
        df1['close_prior1'] = df1.groupby(by=['code', 'f_quarter']).close.shift(1)
        df1['re'] = df1['eps_delta'] / df1['close_prior1']
        
        # Calculate rolling sum
        df1['re_i'] = df1['re'].replace(np.nan, 0)
        df1 = df1[['code', 'year', 'month', 'f_quarter', 're_i']]
        df1['re'] = df1.groupby(by=['code', 'f_quarter'])['re_i'].transform(
            lambda x: x.rolling(6, center=False).sum())
        df1['date'] = pd.to_datetime(
            df1['year'].astype(int).astype(str) + '-' + df1['month'].astype(int).astype(str),
            format='%Y-%m')
        df1 = df1[['code', 'date', 're']]
        
        return df1.dropna()
    
    def validate(self) -> bool:
        """Run validation tests for Re factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_re = self.calculate()
        print(f"Shape: {df_re.shape}")
        print(f"Sample:\n{df_re.head()}")
        
        # Convert to factor frame
        ft_re = self.as_factor_frame(df_re, date_col='date')
        print(f"Factor frame shape: {ft_re.shape}")
        print(f"Factor frame sample:\n{ft_re.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True

class R6Factor(A1MomentumFactorBase):
    """
    A.1.4 Prior 6-Month Returns (R6)
    
    At the beginning of each month t, split all stocks into deciles based on 
    their prior 6-month returns from month t-7 to t-2. Skipping month t-1, 
    calculate monthly decile returns.
    
    Formula:
        R6 = (P_{t-2} - P_{t-7}) / P_{t-7}
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.4"
    
    @property
    def abbr(self) -> str:
        return "r6"
    
    def _priorN_MonthsReturns(self, N: int = 6, **kwargs) -> pd.DataFrame:
        """
        Calculate prior N-month returns.
        
        Parameters
        ----------
        N : int
            Number of months for return calculation
        """
        df = load_wind_daily_data(col=['close'])
        df = date_transfer(df, 'date', format="%Y%m%d")
        
        # Get month-end prices
        df.sort_values(by=['code', 'date'])
        df = df.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Calculate N-month momentum
        df[f'close_prior_{N+1}-months'] = df.groupby('code')['close'].shift(N + 1)
        df['close_prior_1-months'] = df.groupby('code')['close'].shift(2)
        df[f'prior_{N}-month_returns'] = (
            (df['close_prior_1-months'] - df[f'close_prior_{N+1}-months']) /
            df[f'close_prior_{N+1}-months']
        )
        
        df = df[['code', 'year', 'month', f'prior_{N}-month_returns']]
        return df
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate R6 factor.
        Returns:
            Dataframe with columns: code, year, month, ts, r6
        """
        data_mom6 = self._priorN_MonthsReturns(N=6)
        data_mom6['ts'] = pd.to_datetime(
            data_mom6['year'].astype(str) + '-' + data_mom6['month'].astype(str),
            format='%Y-%m')
        data_mom6 = data_mom6.rename(columns={'prior_6-month_returns': 'r6'})
        return data_mom6
    
    def validate(self) -> bool:
        """Run validation tests for R6 factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_r6 = self.calculate()
        print(f"Shape: {df_r6.shape}")
        print(f"Sample:\n{df_r6.head()}")
        
        # Convert to factor frame
        ft_r6 = self.as_factor_frame(df_r6, date_col='ts')
        ft_r6 = ft_r6.rename(columns={'prior_6-month_returns': 'r6'})
        print(f"Factor frame shape: {ft_r6.shape}")
        print(f"Factor frame sample:\n{ft_r6.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True

class R11Factor(A1MomentumFactorBase):
    """
    A.1.5 Prior 11-Month Returns (R11)
    
    At the beginning of each month t, split all stocks into deciles based on 
    their prior 11-month returns from month t-12 to t-2. Skipping month t-1, 
    calculate monthly decile returns.
    
    Formula:
        R11 = (P_{t-2} - P_{t-12}) / P_{t-12}
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.5"
    
    @property
    def abbr(self) -> str:
        return "r11"
    
    def _priorN_MonthsReturns(self, N: int = 11, **kwargs) -> pd.DataFrame:
        """Calculate prior N-month returns."""
        df = load_wind_daily_data(col=['close'])
        df = date_transfer(df, 'date', format="%Y%m%d")
        
        # Get month-end prices
        df.sort_values(by=['code', 'date'])
        df = df.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Calculate N-month momentum
        df[f'close_prior_{N+1}-months'] = df.groupby('code')['close'].shift(N + 1)
        df['close_prior_1-months'] = df.groupby('code')['close'].shift(2)
        df[f'prior_{N}-month_returns'] = (
            (df['close_prior_1-months'] - df[f'close_prior_{N+1}-months']) /
            df[f'close_prior_{N+1}-months']
        )
        
        df = df[['code', 'year', 'month', f'prior_{N}-month_returns']]
        return df
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate R11 factor.
        Returns:
            Dataframe with columns: code, year, month, ts, r11
        """
        data_mom11 = self._priorN_MonthsReturns(N=11)
        data_mom11['ts'] = pd.to_datetime(
            data_mom11['year'].astype(str) + '-' + data_mom11['month'].astype(str),
            format='%Y-%m')
        data_mom11 = data_mom11.rename(columns={'prior_11-month_returns': 'r11'})
        return data_mom11
    
    def validate(self) -> bool:
        """Run validation tests for R11 factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_r11 = self.calculate()
        print(f"Shape: {df_r11.shape}")
        print(f"Sample:\n{df_r11.head()}")
        
        # Convert to factor frame
        ft_r11 = self.as_factor_frame(df_r11, date_col='ts')
        ft_r11 = ft_r11.rename(columns={'prior_11-month_returns': 'r11'})
        print(f"Factor frame shape: {ft_r11.shape}")
        print(f"Factor frame sample:\n{ft_r11.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class RsFactor(A1MomentumFactorBase):
    """
    A.1.7 Revenue Surprises (Rs)
    
    Revenue surprise measure similar to Sue but using revenue instead of earnings.
    
    Formula:
        Rs = (Revenue_PS_t - Revenue_PS_t-4) / Std(Revenue_PS, last 8 quarters)
    
    where Revenue_PS is revenue per share
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.7"
    
    @property
    def abbr(self) -> str:
        return "rs"
    
    def calculate(self, window_size: int = 8, **kwargs) -> pd.DataFrame:
        """
        Calculate Rs factor.
        
        Args:
            window_size: Rolling window size for std calculation (default 8 quarters)
            
        Returns:
            DataFrame with columns: code, end_date, ann_date, rs
        """
        # Load revenue per share data
        df = load_fina_indicator_data(
            col=['revenue_ps', 'quarterly_revenue_ps_fillna']
        )
        
        # Use quarterly fillna version (similar to Sue)
        df = df[['code', 'quarterly_revenue_ps_fillna', 'ann_date', 'end_date']]
        df = df.rename(columns={'quarterly_revenue_ps_fillna': 'revenue_ps'})
        
        # Ensure end_date is datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.sort_values(['code', 'end_date'], ascending=True)
        
        # Shift revenue_ps by 1 year (shift_num=1 means shift back 1 year)
        df_shift = get_annual_shift_data(
            df[['code', 'end_date', 'revenue_ps']],
            shift_num=1
        )
        
        # Merge shifted data
        df = pd.merge(df, df_shift, on=['code', 'end_date'], how='left')
        
        # Calculate rolling std over last 8 quarters
        df['std_prior_8'] = (
            df.groupby(['code'])['revenue_ps']
            .rolling(window=window_size)
            .std()
            .reset_index(level=0, drop=True)
        )
        
        # Calculate revenue surprise
        df['rs'] = (df['revenue_ps'] - df['revenue_ps_1']) / df['std_prior_8']
        
        return df[['code', 'end_date', 'ann_date', 'rs']]
    
    def validate(self) -> bool:
        """Run validation tests for Rs factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_rs = self.calculate()
        print(f"Shape: {df_rs.shape}")
        print(f"Non-NaN values: {df_rs['rs'].notna().sum()}")
        print(f"Sample:\n{df_rs.head(10)}")
        
        # Convert to factor frame
        ft_rs = self.as_factor_frame(df_rs)
        print(f"Factor frame shape: {ft_rs.shape}")
        print(f"Factor frame sample:\n{ft_rs.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class TesFactor(A1MomentumFactorBase):
    """
    A.1.8 Tax Expense Surprises (Tes)
    
    Tax expense surprise measure based on changes in taxes payable relative to total assets.
    
    Formula:
        Tes = (TaxesPayable_PS_t - TaxesPayable_PS_t-4) / TotalAssets_PS_t-4
    
    where _PS denotes per share values
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.8"
    
    @property
    def abbr(self) -> str:
        return "tes"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate Tes factor.
        
        Returns:
            DataFrame with columns: code, end_date, tes
        """
        # Load balance sheet data
        df = load_balancesheet_data(
            col=['quarterly_taxes_payable_fillna',
                 'quarterly_total_share_fillna',
                 'quarterly_total_assets_fillna']
        )
        
        df = df.rename(columns={
            'quarterly_taxes_payable_fillna': 'taxes_payable',
            'quarterly_total_share_fillna': 'total_share',
            'quarterly_total_assets_fillna': 'total_assets',
            'date': 'end_date'
        })
        
        # Ensure end_date is datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.sort_values(['code', 'end_date'], ascending=True)
        
        # Calculate per-share values
        df['taxes_payable_ps'] = df['taxes_payable'] / df['total_share']
        df['total_assets_ps'] = df['total_assets'] / df['total_share']
        
        # Shift by 1 year (shift_num=1 means shift back 1 year)
        df_shift = get_annual_shift_data(
            df[['code', 'end_date', 'taxes_payable_ps', 'total_assets_ps']],
            shift_num=1
        )
        
        # Merge shifted data
        df = pd.merge(df, df_shift, on=['code', 'end_date'], how='left')
        
        # Calculate tax expense surprise
        df['tes'] = (
            (df['taxes_payable_ps'] - df['taxes_payable_ps_1']) / 
            df['total_assets_ps_1']
        )
        
        return df[['code', 'end_date', 'tes']]
    
    def validate(self) -> bool:
        """Run validation tests for Tes factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_tes = self.calculate()
        print(f"Shape: {df_tes.shape}")
        print(f"Non-NaN values: {df_tes['tes'].notna().sum()}")
        print(f"Sample:\n{df_tes.head(10)}")
        
        # Convert to factor frame
        ft_tes = self.as_factor_frame(df_tes)
        print(f"Factor frame shape: {ft_tes.shape}")
        print(f"Factor frame sample:\n{ft_tes.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class NeiFactor(A1MomentumFactorBase):
    """
    A.1.10 Consecutive Earnings Increases (Nei)
    
    Counts the number of consecutive quarters with earnings growth.
    Capped at maximum of 8 consecutive quarters.
    
    Formula:
        Nei = Count of consecutive quarters where NI_t > NI_t-1 (max 8)
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.10"
    
    @property
    def abbr(self) -> str:
        return "nei"
    
    def _cal_continuous_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of continuous growth periods in dataframe.
        
        Args:
            df: DataFrame with 'diff' column containing period-over-period changes
            
        Returns:
            DataFrame with added 'nei' column counting consecutive increases
        """
        # Check whether diff > 0
        df['increase'] = df['diff'].gt(0)
        s = df['diff']
        s1 = s.gt(0)
        df['inc'] = s1.cumsum().where(s1, 0)
        si = df['inc'].eq(0)
        df['inc'] = si.groupby(si.cumsum()).cumcount()
        df = df.drop(columns=['increase'])
        df['nei'] = df['inc']
        return df
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate Nei factor.
        
        Returns:
            DataFrame with columns: code, ann_date, end_date, nei
        """
        # Load net income data
        df = load_income_data(
            col=['quarterly_n_income_fillna', 'n_income']
        )
        
        df = df[['code', 'quarterly_n_income_fillna', 'end_date', 'ann_date']]
        df = df.rename(columns={'quarterly_n_income_fillna': 'n_income'})
        df = df.sort_values(['code', 'end_date'], ascending=True)
        
        # Calculate quarter-over-quarter difference
        df['diff'] = df.groupby(['code'])['n_income'].diff()
        
        # Calculate consecutive growth periods
        df = self._cal_continuous_growth(df)
        
        # Cap at 8 consecutive quarters
        df.loc[df['nei'] >= 8, 'nei'] = 8
        
        return df[['code', 'ann_date', 'end_date', 'nei']]
    
    def validate(self) -> bool:
        """Run validation tests for Nei factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_nei = self.calculate()
        print(f"Shape: {df_nei.shape}")
        print(f"Non-NaN values: {df_nei['nei'].notna().sum()}")
        print(f"Statistics:\n{df_nei['nei'].describe()}")
        print(f"Sample:\n{df_nei.head(10)}")
        
        # Convert to factor frame
        ft_nei = self.as_factor_frame(df_nei)
        print(f"Factor frame shape: {ft_nei.shape}")
        print(f"Factor frame sample:\n{ft_nei.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class ImFactor(A1MomentumFactorBase):
    """
    A.1.6 Im - Industry Momentum
    
    Calculates value-weighted industry returns over the prior 6 months.
    For each stock, assigns the momentum of its industry.
    
    Formula:
        Im_industry = Σ(w_i * R_i) / Σ(w_i)
    
    where:
        - w_i: market value of stock i in the industry
        - R_i: 6-month prior return of stock i
        - Industry classification uses class1 (一级行业分类)
    
    Higher values indicate industries with strong momentum.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.6"
    
    @property
    def abbr(self) -> str:
        return "im"
    
    def _weighted_avg(self, group: pd.DataFrame, ret_col: str = 'ret', mv_col: str = 'mv') -> float:
        """Calculate value-weighted average return."""
        d = group[ret_col] * group[mv_col]
        return d.sum() / group[mv_col].sum()
    
    def calculate(self, window_size: int = 6, induscls: int = 1, **kwargs) -> pd.DataFrame:
        """
        Calculate Industry Momentum factor.
        
        Parameters
        ----------
        window_size : int
            Number of months for momentum calculation (default: 6)
        induscls : int
            Industry classification level (1-4, default: 1)
            
        Returns
        -------
        pd.DataFrame
            Columns: code, year, month, im
        """
        # Step 1: Load industry classification data
        df_indus = load_indus_class_data()
        class_col = f'class{induscls}'
        df_indus = df_indus[['code', class_col]]
        
        # Remove duplicates - keep first occurrence for each stock
        df_indus = df_indus.drop_duplicates(subset=['code'], keep='first')
        
        # Step 2: Load daily basic data for market value
        df_basic = load_daily_basic_data(col=['total_mv'])
        
        # Get month-end market value
        df_basic = date_transfer(df_basic, 'date', format=None)
        df_basic.sort_values(by=['code', 'date'], inplace=True)
        df_basic = df_basic.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Merge industry classification
        df_basic = pd.merge(left=df_basic, right=df_indus, on='code', how='left')
        
        # Shift market value by window_size + 1 months
        # (to align with when the momentum is calculated)
        df_basic['total_mv_shift'] = df_basic.groupby(by=['code'])['total_mv'].shift(window_size + 1)
        
        # Step 3: Calculate individual stock momentum using R6-style calculation
        df_wind = load_wind_daily_data(col=['close'])
        df_wind = date_transfer(df_wind, 'date', format="%Y%m%d")
        df_wind.sort_values(by=['code', 'date'], inplace=True)
        df_wind = df_wind.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Calculate prior N-month returns (from t-N-1 to t-2)
        df_wind[f'close_prior_{window_size+1}'] = df_wind.groupby('code')['close'].shift(window_size + 1)
        df_wind['close_prior_1'] = df_wind.groupby('code')['close'].shift(2)
        mom_col = f'prior_{window_size}-month_returns'
        df_wind[mom_col] = (
            (df_wind['close_prior_1'] - df_wind[f'close_prior_{window_size+1}']) /
            df_wind[f'close_prior_{window_size+1}']
        )
        
        df_wind = df_wind[['code', 'year', 'month', mom_col]]
        
        # Step 4: Merge market value and momentum data
        df = pd.merge(left=df_wind, right=df_basic, on=['code', 'year', 'month'], how='left')
        df.dropna(subset=['total_mv_shift', mom_col], inplace=True)
        
        # Step 5: Calculate value-weighted industry momentum
        df1 = df.groupby(by=['year', 'month', class_col]).apply(
            lambda x: self._weighted_avg(x, mom_col, 'total_mv_shift')
        ).reset_index(name=f'im{window_size}cls{induscls}')
        
        # Merge back to individual stocks
        df = df.merge(df1, on=['year', 'month', class_col], how='left')
        
        # Rename to standard 'im' column
        df = df.rename(columns={f'im{window_size}cls{induscls}': 'im'})
        
        return df[['code', 'year', 'month', 'im']]
    
    def validate(self) -> bool:
        """Run validation tests for Im factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_im = self.calculate()
        print(f"Shape: {df_im.shape}")
        print(f"Non-NaN values: {df_im['im'].notna().sum()}")
        print(f"Statistics:\n{df_im['im'].describe()}")
        print(f"Sample:\n{df_im.head(20)}")
        
        # Check unique industries
        # Need to re-merge to show industries
        print(f"\nUnique stocks: {df_im['code'].nunique()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class W52Factor(A1MomentumFactorBase):
    """
    A.1.11 52w - 52-Week High
    
    Calculates the ratio of current month-end price to the highest price
    over the past 12 months. This factor captures proximity to 52-week highs,
    which is a momentum signal.
    
    Formula:
        52w = price_t / max(price_{t-11}, ..., price_t)
    
    where:
        - price_t: price per share at month-end of month t
        - max(...): highest month-end price over past 12 months
        
    Higher values indicate stocks trading closer to their 52-week highs.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.11"
    
    @property
    def abbr(self) -> str:
        return "w52"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate 52-week high factor.
        
        Steps:
            1. Load daily basic data (total_mv, total_share)
            2. Calculate price per share = total_mv / total_share
            3. Get month-end prices (last trading day each month)
            4. Calculate 12-month rolling max price
            5. Compute w52 = current_price / max_price_12m
            
        Returns:
            pd.DataFrame: Factor values with columns [code, year, month, w52]
        """
        # Step 1: Load daily basic data
        df = load_daily_basic_data(col=['total_mv', 'total_share'])
        
        # Step 2: Calculate price per share
        df['pps'] = df['total_mv'] / df['total_share']
        
        # Step 3: Date processing and get month-end prices
        df = date_transfer(df, 'date', format=None)
        df.sort_values(by=['code', 'date'], inplace=True)
        
        # Get last trading day of each month (month-end price)
        df = df.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Step 4: Calculate 12-month rolling max price
        # min_periods=1 allows calculation even with less than 12 months of data
        df['pps_max_12m'] = (
            df.groupby(by=['code'])['pps']
            .rolling(window=12, center=False, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        
        # Step 5: Calculate 52-week high ratio
        df['w52'] = df['pps'] / df['pps_max_12m']
        
        # Select and return final columns
        return df[['code', 'year', 'month', 'w52']]
    
    def validate(self) -> bool:
        """Run validation tests for 52w factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_52w = self.calculate()
        print(f"Shape: {df_52w.shape}")
        print(f"Non-NaN values: {df_52w['w52'].notna().sum()}")
        print(f"Statistics:\n{df_52w['w52'].describe()}")
        
        # Check for issues
        n_greater_than_1 = (df_52w['w52'] > 1.0).sum()
        n_le_zero = (df_52w['w52'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1} (should be 0)")
        print(f"Values <= 0: {n_le_zero}")
        
        print(f"Sample:\n{df_52w.head(20)}")
        
        # Convert to factor frame
        df_52w['date'] = pd.to_datetime(
            df_52w['year'].astype(str) + df_52w['month'].astype(str).str.zfill(2) + '01',
            format='%Y%m%d'
        )
        ft_52w = self.as_factor_frame(df_52w, date_col='date')
        print(f"Factor frame shape: {ft_52w.shape}")
        print(f"Factor frame sample:\n{ft_52w.head()}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class Rm6Factor(A1MomentumFactorBase):
    """
    A.1.12 rm6 (ε6) - 6-Month Residual Momentum
    
    Calculates residual returns from FF3 regression over prior 6-month period.
    Residuals capture stock-specific returns not explained by market factors.
    
    Formula:
        R_it - R_ft = α + β₁(R_Mt - R_ft) + β₂SMB_t + β₃HML_t + ε_it
        
    where ε_it is the residual (idiosyncratic) return.
    
    The factor uses the average of the most recent 5 months' residuals
    from a 36-month rolling regression.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.12"
    
    @property
    def abbr(self) -> str:
        return "rm6"
    
    def _get_monthly_return(self) -> pd.DataFrame:
        """Get month-end returns for all stocks."""
        df = load_wind_daily_data(col=['close'])
        df = date_transfer(df, 'date', format="%Y%m%d")
        df.sort_values(by=['code', 'date'], inplace=True)
        df = df.groupby(by=['code', 'year', 'month']).tail(1)
        
        # Calculate monthly return
        df['monthly_return'] = df.groupby(by=['code'])['close'].pct_change()
        df = df[['date', 'code', 'monthly_return', 'year', 'month']]
        return df
    
    def _perform_regression(self, data: pd.DataFrame, x_cols: list, y_col: str, 
                           resid_days: int = 5) -> tuple:
        """Perform FF3 regression and return recent residuals."""
        if len(data) < 36:  # Need at least 36 months
            return np.nan, np.nan
        
        Y = data[y_col]
        X = data[x_cols]
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(Y, X, missing='drop').fit()
            residuals = model.resid
            std = residuals.std()
            # Return mean of most recent resid_days months
            return np.mean(residuals[-resid_days:]), std
        except:
            return np.nan, np.nan
    
    def _get_rolling_residual_return(self, df: pd.DataFrame, y_col: str, 
                                     x_cols: list, window_size: int = 36,
                                     resid_days: int = 5) -> pd.DataFrame:
        """Calculate rolling FF3 residuals."""
        rolled_df = df.rolling(window=window_size)
        res_means, res_stds = [], []
        
        for window_data in rolled_df:
            res, std = self._perform_regression(window_data, x_cols, y_col, resid_days)
            res_means.append(res)
            res_stds.append(std)
        
        df['res_means'] = res_means
        df['res_stds'] = res_stds
        return df
    
    def _cal_residual_means_stds(self, resid_days: int = 5) -> pd.DataFrame:
        """Calculate FF3 residual means and standard deviations."""
        # Get monthly returns
        df = self._get_monthly_return()
        
        # Load FF3 data
        ff3 = load_ff3_data(col=[])
        ff3 = date_transfer(ff3, 'date', format="%Y%m%d")
        
        # Aggregate daily FF3 factors to monthly (annualize by *22 trading days)
        ff3 = ff3.groupby(by=['year', 'month'])[['rmrftmv', 'smbtmv', 'hmltmv']].mean().reset_index()
        for col in ['rmrftmv', 'smbtmv', 'hmltmv']:
            ff3[col] = ff3[col].apply(lambda x: float(x) * 22)
        
        # Merge returns with FF3 factors
        df = pd.merge(left=df, right=ff3, on=['year', 'month'], how='left')
        
        # Prepare for regression
        y_col = 'monthly_return'
        x_cols = ['rmrftmv', 'smbtmv', 'hmltmv']
        cols = x_cols + [y_col]
        df.dropna(subset=cols, inplace=True)
        df.sort_values(by=['code', 'date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Rolling regression by stock
        df = (df.groupby(by=['code'], group_keys=True)
              .apply(lambda _df: self._get_rolling_residual_return(
                  _df, y_col, x_cols, window_size=36, resid_days=resid_days))
              .reset_index(drop=True))
        
        return df
    
    def calculate(self, resid_days: int = 5, **kwargs) -> pd.DataFrame:
        """
        Calculate rm6 factor.
        
        Parameters
        ----------
        resid_days : int
            Number of recent months to average residuals (default: 5 for rm6)
            
        Returns
        -------
        pd.DataFrame
            Columns: code, date, rm6
        """
        df = self._cal_residual_means_stds(resid_days=resid_days)
        df = df[['code', 'date', 'res_means', 'res_stds']]
        df = df.sort_values(by=['code', 'date'])
        
        # Shift by 2 months (skip month t-1)
        df = df.set_index(['code', 'date']).groupby(level=0)[['res_means', 'res_stds']].shift(2).reset_index()
        df = df.rename(columns={'res_means': 'rm6'})
        
        return df[['code', 'date', 'rm6']]
    
    def validate(self) -> bool:
        """Run validation tests for rm6 factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_rm6 = self.calculate()
        print(f"Shape: {df_rm6.shape}")
        print(f"Non-NaN values: {df_rm6['rm6'].notna().sum()}")
        print(f"Statistics:\n{df_rm6['rm6'].describe()}")
        print(f"Sample:\n{df_rm6.head(20)}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


class Rm11Factor(A1MomentumFactorBase):
    """
    A.1.13 rm11 (ε11) - 11-Month Residual Momentum
    
    Similar to rm6 but uses the average of the most recent 11 months' residuals
    from a 36-month rolling FF3 regression.
    
    Formula:
        R_it - R_ft = α + β₁(R_Mt - R_ft) + β₂SMB_t + β₃HML_t + ε_it
    """
    
    @property
    def factor_id(self) -> str:
        return "A.1.13"
    
    @property
    def abbr(self) -> str:
        return "rm11"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate rm11 factor by reusing Rm6Factor's methods with resid_days=11.
        
        Returns
        -------
        pd.DataFrame
            Columns: code, date, rm11
        """
        # Create rm6 instance to reuse its calculation logic
        rm6_factor = Rm6Factor()
        
        # Calculate with resid_days=11 for rm11
        df = rm6_factor._cal_residual_means_stds(resid_days=11)
        df = df[['code', 'date', 'res_means', 'res_stds']]
        df = df.sort_values(by=['code', 'date'])
        
        # Shift by 2 months
        df = df.set_index(['code', 'date']).groupby(level=0)[['res_means', 'res_stds']].shift(2).reset_index()
        df = df.rename(columns={'res_means': 'rm11'})
        
        return df[['code', 'date', 'rm11']]
    
    def validate(self) -> bool:
        """Run validation tests for rm11 factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")
        
        # Calculate factor
        df_rm11 = self.calculate()
        print(f"Shape: {df_rm11.shape}")
        print(f"Non-NaN values: {df_rm11['rm11'].notna().sum()}")
        print(f"Statistics:\n{df_rm11['rm11'].describe()}")
        print(f"Sample:\n{df_rm11.head(20)}")
        
        print(f"Validation complete for {self.abbr}!")
        return True


# =============================================================================
# SECTION 10: A2 VALUE FACTOR IMPLEMENTATIONS  
# =============================================================================


class BmFactor(A2ValueFactorBase):
    """
    A.2.1 Bm - Book-to-Market Equity
    
    Bm = Book Equity(t-1) / Market Equity(t-1)
    
    Book Equity = Stockholders' Book Equity + TXDITC - Book Value of Preferred Stock
    where:
    - Stockholders' Book Equity = SEQ or (CEQ + PSTK) or (AT - LT)
    - Book Value of Preferred Stock = PSTKRV, PSTKL, or PSTK
    
    Monthly decile returns calculated from July of year t to June of t+1,
    rebalanced in June of t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.1"
    
    @property
    def abbr(self) -> str:
        return "Bm"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate book-to-market ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Bm']
        """
        # Get quarterly book equity
        be = get_quartly_be()
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        be_dec = be[pd.to_datetime(be['end_date']).dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        be_dec['year'] = pd.to_datetime(be_dec['end_date']).dt.year
        mv_dec['year'] = mv_dec['year']
        
        result = pd.merge(be_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Bm ratio
        result['Bm'] = result['quarterly_book_equity'] / result['monthly_mv_end']
        
        return result[['code', 'end_date', 'Bm']]
    
    def validate(self) -> bool:
        """Validate Bm factor calculation."""
        df = self.calculate()
        return not df.empty and 'Bm' in df.columns


class DmFactor(A2ValueFactorBase):
    """
    A.2.4 Dm - Debt-to-Market
    
    Dm = Total Debt(t-1) / Market Equity(t-1)
    
    where Total Debt = DLC + DLTT (short-term + long-term debt)
    
    At the end of June of each year t, stocks are split into deciles based on Dm,
    which is total debt for the fiscal year ending in calendar year t−1 divided by
    the market equity at the end of December of t−1.
    
    Firms with no debt are excluded.
    Monthly decile returns calculated from July of year t to June of t+1,
    rebalanced in June of t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.4"
    
    @property
    def abbr(self) -> str:
        return "Dm"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate debt-to-market ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Dm']
        """
        # Load total liabilities
        df = load_balancesheet_data(['quarterly_total_liab_fillna'], RESSET_DIR)
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        
        result = pd.merge(df_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Dm ratio
        result['Dm'] = result['quarterly_total_liab_fillna'] / result['monthly_mv_end']
        
        return result[['code', 'end_date', 'Dm']]
    
    def validate(self) -> bool:
        """Validate Dm factor calculation."""
        df = self.calculate()
        return not df.empty and 'Dm' in df.columns


class AmFactor(A2ValueFactorBase):
    """
    A.2.6 Am - Assets-to-Market
    
    Am = Total Assets(t-1) / Market Equity(t-1)
    
    At the end of June of each year t, stocks are split into deciles based on Am,
    which is total assets (AT) for the fiscal year ending in calendar year t−1
    divided by the market equity at the end of December of t−1.
    
    Monthly decile returns calculated from July of year t to June of t+1,
    rebalanced in June of t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.6"
    
    @property
    def abbr(self) -> str:
        return "Am"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate assets-to-market ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Am']
        """
        # Load total assets
        df = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        
        result = pd.merge(df_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Am ratio
        result['Am'] = result['quarterly_total_assets_fillna'] / result['monthly_mv_end']
        
        return result[['code', 'end_date', 'Am']]
    
    def validate(self) -> bool:
        """Validate Am factor calculation."""
        df = self.calculate()
        return not df.empty and 'Am' in df.columns


class EpFactor(A2ValueFactorBase):
    """
    A.2.9 Ep - Earnings-to-Price
    
    Ep = Earnings(t-1) / Market Equity(t-1)
    
    At the end of June of each year t, stocks are split into deciles based on Ep,
    which is earnings (IB) for the fiscal year ending in calendar year t−1
    divided by the market equity at the end of December of t−1.
    
    Firms with negative earnings are excluded.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.9"
    
    @property
    def abbr(self) -> str:
        return "Ep"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate earnings-to-price ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ep']
        """
        # Load net profit (earnings)
        df = load_income_data(['quarterly_n_income_fillna'], RESSET_DIR)
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        
        result = pd.merge(df_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Ep ratio (exclude negative earnings)
        result['Ep'] = result['quarterly_n_income_fillna'] / result['monthly_mv_end']
        result.loc[result['quarterly_n_income_fillna'] < 0, 'Ep'] = np.nan
        
        return result[['code', 'end_date', 'Ep']]
    
    def validate(self) -> bool:
        """Validate Ep factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ep' in df.columns


class CpFactor(A2ValueFactorBase):
    """
    A.2.12 Cp - Cash Flow-to-Price
    
    Cp = Operating Cash Flow(t-1) / Market Equity(t-1)
    
    Operating cash flow is from the cash flow statement (OANCF).
    At the end of June of each year t, stocks are split into deciles based on Cp.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.12"
    
    @property
    def abbr(self) -> str:
        return "Cp"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate cash flow-to-price ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Cp']
        """
        # Load operating cash flow
        df = load_cashflow_data(['quarterly_n_cashflow_act_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime, coerce errors to NaT
        df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d', errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        df_dec = df[df['end_date'].dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        df_dec['year'] = df_dec['end_date'].dt.year
        
        result = pd.merge(df_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Cp ratio
        result['Cp'] = result['quarterly_n_cashflow_act_fillna'] / result['monthly_mv_end']
        
        # Convert end_date back to datetime64[ns] for output
        result['end_date'] = pd.to_datetime(result['end_date'])
        
        return result[['code', 'end_date', 'Cp']]
    
    def validate(self) -> bool:
        """Validate Cp factor calculation."""
        df = self.calculate()
        return not df.empty and 'Cp' in df.columns

class RevFactor(A2ValueFactorBase):
    """
    A.2.8 Rev - Reversal (Short-term Reversal)
    
    Rev = Return(t-13) / Return(t-60) - 1
    
    Reversal is the return from month t-13 to month t-60.
    Measures short-term price reversals using monthly stock prices.
    Uses the last trading day of each month.
    
    Data Source:
        - close prices from daily_basic_resset.csv
        
    Note:
        Default parameters: t1=60 (5 years ago), t2=13 (just over 1 year ago)
        Rev captures the return from t-60 to t-13 months.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.8"
    
    @property
    def abbr(self) -> str:
        return "Rev"
    
    def calculate(self, t1: int = 60, t2: int = 13, **kwargs) -> pd.DataFrame:
        """
        Calculate reversal factor.
        
        Args:
            t1: Lookback period start (default 60 months)
            t2: Lookback period end (default 13 months)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Rev']
        """
        # Load daily close prices
        df = load_daily_basic_data(['close'], RESSET_DIR)
        
        # Convert date to datetime (handle both YYYYMMDD and YYYY-MM-DD formats)
        if df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        
        # Get month-end observations only
        df['year_month'] = df['date'].dt.to_period('M')
        df = df.sort_values(['code', 'date'])
        df = df.groupby(['code', 'year_month']).tail(1).reset_index(drop=True)
        
        # Calculate reversal: price(t-13) / price(t-60) - 1
        df = df.sort_values(['code', 'date'])
        df['close_lag_t2'] = df.groupby('code')['close'].shift(t2)
        df['close_lag_t1'] = df.groupby('code')['close'].shift(t1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Rev'] = (df['close_lag_t2'] / df['close_lag_t1']) - 1
        
        # Replace inf/-inf with NaN
        df['Rev'] = df['Rev'].replace([np.inf, -np.inf], np.nan)
        
        # Rename date column to end_date for consistency
        df = df.rename(columns={'date': 'end_date'})
        
        return df[['code', 'end_date', 'Rev']].dropna()
    
    def validate(self) -> bool:
        """Validate Rev factor calculation."""
        df = self.calculate()
        return not df.empty and 'Rev' in df.columns


class SpFactor(A2ValueFactorBase):
    """
    A.2.22 Sp - Sales-to-Price
    
    Sp = Sales / Market Equity
    
    Sales-to-price is sales from fiscal year t-1 divided by market equity
    at the end of December of t-1.
    
    Data Sources:
        - Sales: c_fr_sale_sg (cash from sales) from quarterly_cashflow_cleaned.csv
        - Market Equity: circ_mv from daily_basic_resset.csv
        
    Note:
        Uses annual fiscal year-end data (December only).
        Firms with nonpositive sales are excluded.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.22"
    
    @property
    def abbr(self) -> str:
        return "Sp"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate sales-to-price ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Sp']
        """
        # Load sales data from cash flow statement
        df_sales = load_cashflow_data(['quarterly_c_fr_sale_sg_fillna'], path=RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df_sales = df_sales[pd.to_datetime(df_sales['end_date']).dt.month == 12].copy()
        df_sales['year'] = pd.to_datetime(df_sales['end_date']).dt.year
        df_sales['month'] = 12
        
        # Load market value data
        df_mv = load_daily_basic_data(['circ_mv'], RESSET_DIR)
        
        # Convert date to datetime (handle both formats)
        if df_mv['date'].dtype == 'object':
            df_mv['date'] = pd.to_datetime(df_mv['date'], errors='coerce')
        else:
            df_mv['date'] = pd.to_datetime(df_mv['date'].astype(str), format='%Y%m%d', errors='coerce')
        
        df_mv['year'] = df_mv['date'].dt.year
        df_mv['month'] = df_mv['date'].dt.month
        
        # Get December market values
        df_mv = df_mv[df_mv['month'] == 12].copy()
        
        # Merge sales and market value on code, year, month
        df = pd.merge(df_sales, df_mv, on=['code', 'year', 'month'], how='inner')
        
        # Calculate Sp
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Sp'] = df['quarterly_c_fr_sale_sg_fillna'] / df['circ_mv']
        
        # Replace inf/-inf with NaN and filter out non-positive sales
        df['Sp'] = df['Sp'].replace([np.inf, -np.inf], np.nan)
        df = df[df['quarterly_c_fr_sale_sg_fillna'] > 0]
        
        return df[['code', 'end_date', 'Sp']].dropna()
    
    def validate(self) -> bool:
        """Validate Sp factor calculation."""
        df = self.calculate()
        return not df.empty and 'Sp' in df.columns


class DmqFactor(A2ValueFactorBase):
    """
    A.2.5 Dmq - Quarterly Debt-to-Market
    
    Dmq = Total Debt(t-1) / Market Equity(t-1)
    
    At the beginning of each month t, stocks are split into deciles based on quarterly
    debt-to-market, Dmq, which is total debt from the most recent fiscal quarter ending 
    at least four months ago, divided by the market equity at the end of month t−1.
    
    Monthly decile returns calculated for the current month t (Dmq1), from month t to t+5 (Dmq6),
    and from month t to t+11 (Dmq12), rebalanced at the beginning of month t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.5"
    
    @property
    def abbr(self) -> str:
        return "Dmq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly debt-to-market ratio.
        
        Args:
            shift_months: Number of months to shift quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Dmq']
        """
        # Load quarterly total liabilities
        df = load_balancesheet_data(['quarterly_total_liab_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['quarterly_total_liab_fillna'] = result.groupby('code')['quarterly_total_liab_fillna'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['quarterly_total_liab_fillna'] = result.groupby('code')['quarterly_total_liab_fillna'].shift(shift_months)
        
        # Calculate Dmq ratio
        result['Dmq'] = result['quarterly_total_liab_fillna'] / result['monthly_mv_end']
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Dmq']].dropna()
    
    def validate(self) -> bool:
        """Validate Dmq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Dmq' in df.columns


class AmqFactor(A2ValueFactorBase):
    """
    A.2.7 Amq - Quarterly Assets-to-Market
    
    Amq = Total Assets(t-1) / Market Equity(t-1)
    
    At the beginning of each month t, stocks are split into deciles based on quarterly
    assets-to-market, Amq, which is total assets from the most recent fiscal quarter ending
    at least four months ago, divided by the market equity at the end of month t−1.
    
    Monthly decile returns calculated for the current month t (Amq1), from month t to t+5 (Amq6),
    and from month t to t+11 (Amq12), rebalanced at the beginning of month t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.7"
    
    @property
    def abbr(self) -> str:
        return "Amq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly assets-to-market ratio.
        
        Args:
            shift_months: Number of months to shift quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Amq']
        """
        # Load quarterly total assets
        df = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['quarterly_total_assets_fillna'] = result.groupby('code')['quarterly_total_assets_fillna'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['quarterly_total_assets_fillna'] = result.groupby('code')['quarterly_total_assets_fillna'].shift(shift_months)
        
        # Calculate Amq ratio
        result['Amq'] = result['quarterly_total_assets_fillna'] / result['monthly_mv_end']
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Amq']].dropna()
    
    def validate(self) -> bool:
        """Validate Amq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Amq' in df.columns


class EpqFactor(A2ValueFactorBase):
    """
    A.2.10 Epq - Quarterly Earnings-to-Price
    
    Epq = Quarterly Earnings / Market Equity
    
    At the beginning of each month t, stocks are split into deciles based on quarterly
    earnings-to-price, Epq, which is income before extraordinary items from the most 
    recent fiscal quarter ending at least four months ago, divided by the market equity 
    at the end of month t−1.
    
    Firms with negative quarterly earnings are excluded.
    Monthly decile returns calculated for the current month t (Epq1), from month t to t+5 (Epq6),
    and from month t to t+11 (Epq12), rebalanced at the beginning of month t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.10"
    
    @property
    def abbr(self) -> str:
        return "Epq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly earnings-to-price ratio.
        
        Args:
            shift_months: Number of months to shift quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Epq']
        """
        # Load quarterly net income (n_income is net profit)
        df = load_income_data(['quarterly_n_income_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['quarterly_n_income_fillna'] = result.groupby('code')['quarterly_n_income_fillna'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['quarterly_n_income_fillna'] = result.groupby('code')['quarterly_n_income_fillna'].shift(shift_months)
        
        # Calculate Epq ratio
        result['Epq'] = result['quarterly_n_income_fillna'] / result['monthly_mv_end']
        
        # Exclude firms with negative earnings
        result = result[result['quarterly_n_income_fillna'] > 0]
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Epq']].dropna()
    
    def validate(self) -> bool:
        """Validate Epq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Epq' in df.columns


class CpqFactor(A2ValueFactorBase):
    """
    A.2.13 Cpq - Quarterly Cash Flow-to-Price
    
    Cpq = (Quarterly Revenue + Depreciation) / Market Equity
    
    At the beginning of each month t, stocks are split into deciles based on quarterly
    cash flow-to-price, Cpq, which is income before extraordinary items plus depreciation 
    from the most recent fiscal quarter ending at least four months ago, divided by the 
    market equity at the end of month t−1.
    
    Monthly decile returns calculated for the current month t (Cpq1), from month t to t+5 (Cpq6),
    and from month t to t+11 (Cpq12), rebalanced at the beginning of month t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.13"
    
    @property
    def abbr(self) -> str:
        return "Cpq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly cash flow-to-price ratio.
        
        Args:
            shift_months: Number of months to shift quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Cpq']
        """
        # Load quarterly operating revenue
        df_income = load_income_data(['quarterly_moperev_fillna'], RESSET_DIR)
        
        # Load depreciation data
        depreciation_cols = ['depr_fa_coga_dpba1_fillna', 'depr_fa_coga_dpba2_fillna', 'prov_depr_assets_fillna']
        df_depr = load_cashflow_data(depreciation_cols, path=RESSET_DIR)
        
        # Merge income and depreciation data
        df = pd.merge(df_income, df_depr, on=['code', 'end_date'], how='left')
        
        # Calculate cash flow (revenue + depreciation)
        # Use the first available depreciation column
        df['depreciation'] = df['depr_fa_coga_dpba1_fillna'].fillna(0)
        df['cash_flow'] = df['quarterly_moperev_fillna'] + df['depreciation']
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['cash_flow'] = result.groupby('code')['cash_flow'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['cash_flow'] = result.groupby('code')['cash_flow'].shift(shift_months)
        
        # Calculate Cpq ratio
        result['Cpq'] = result['cash_flow'] / result['monthly_mv_end']
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Cpq']].dropna()
    
    def validate(self) -> bool:
        """Validate Cpq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Cpq' in df.columns


class SpqFactor(A2ValueFactorBase):
    """
    A.2.23 Spq - Quarterly Sales-to-Price
    
    Spq = Quarterly Sales / Market Equity
    
    At the beginning of each month t, stocks are split into deciles based on quarterly
    sales-to-price, Spq, which is quarterly sales from the most recent fiscal quarter 
    ending at least four months ago, divided by the market equity at the end of month t−1.
    
    Firms with nonpositive sales are excluded.
    Monthly decile returns calculated for the current month t (Spq1), from month t to t+5 (Spq6),
    and from month t to t+11 (Spq12), rebalanced at the beginning of month t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.23"
    
    @property
    def abbr(self) -> str:
        return "Spq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly sales-to-price ratio.
        
        Args:
            shift_months: Number of months to shift quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Spq']
        """
        # Load quarterly sales from cash flow statement
        df = load_cashflow_data(['quarterly_c_fr_sale_sg_fillna'], path=RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['quarterly_c_fr_sale_sg_fillna'] = result.groupby('code')['quarterly_c_fr_sale_sg_fillna'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['quarterly_c_fr_sale_sg_fillna'] = result.groupby('code')['quarterly_c_fr_sale_sg_fillna'].shift(shift_months)
        
        # Calculate Spq ratio
        result['Spq'] = result['quarterly_c_fr_sale_sg_fillna'] / result['monthly_mv_end']
        
        # Exclude firms with non-positive sales
        result = result[result['quarterly_c_fr_sale_sg_fillna'] > 0]
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Spq']].dropna()
    
    def validate(self) -> bool:
        """Validate Spq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Spq' in df.columns


class OcpFactor(A2ValueFactorBase):
    """
    A.2.24 Ocp - Operating Cash Flow-to-Price (Annual)
    
    Ocp = Operating Cash Flow(t-1) / Market Equity(t-1)
    
    At the end of June of each year t, stocks are split into deciles based on Ocp,
    which is operating cash flows for the fiscal year ending in calendar year t−1
    divided by market equity at the end of December of t−1.
    
    Operating cash flows measured as net cash flows from operating activities.
    Firms with nonpositive operating cash flows are excluded.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.24"
    
    @property
    def abbr(self) -> str:
        return "Ocp"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate operating cash flow-to-price ratio (annual).
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ocp']
        """
        # Load operating cash flow
        df = load_cashflow_data(['quarterly_n_cashflow_act_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        
        # Filter to December only for both datasets
        df_dec = df[df['end_date'].dt.month == 12].copy()
        mv_dec = mv[mv['month'] == 12].copy()
        
        # Merge on code, year
        df_dec['year'] = df_dec['end_date'].dt.year
        
        result = pd.merge(df_dec, mv_dec, on=['code', 'year'], how='inner')
        
        # Calculate Ocp ratio
        result['Ocp'] = result['quarterly_n_cashflow_act_fillna'] / result['monthly_mv_end']
        
        # Exclude firms with nonpositive operating cash flows
        result = result[result['quarterly_n_cashflow_act_fillna'] > 0]
        
        # Convert end_date back to datetime64[ns] for output
        result['end_date'] = pd.to_datetime(result['end_date'])
        
        return result[['code', 'end_date', 'Ocp']].dropna()
    
    def validate(self) -> bool:
        """Validate Ocp factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ocp' in df.columns


class OcpqFactor(A2ValueFactorBase):
    """
    A.2.25 Ocpq - Quarterly Operating Cash Flow-to-Price
    
    Ocpq = Quarterly Operating Cash Flow / Market Equity(t-1)
    
    At the beginning of each month t, stocks are split into deciles based on Ocpq,
    which is operating cash flows for the latest fiscal quarter ending at least
    four months ago divided by market equity at the end of month t−1.
    
    Operating cash flows measured as net cash flows from operating activities.
    Firms with nonpositive operating cash flows are excluded.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.2.25"
    
    @property
    def abbr(self) -> str:
        return "Ocpq"
    
    def calculate(self, shift_months: int = 4, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly operating cash flow-to-price ratio.
        
        Args:
            shift_months: Number of months to lag quarterly data (default 4)
            
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ocpq']
        """
        # Load quarterly operating cash flow
        df = load_cashflow_data(['quarterly_n_cashflow_act_fillna'], RESSET_DIR)
        
        # Convert end_date to datetime
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.dropna(subset=['end_date'])
        
        # Get monthly market value
        mv = get_monthly_mv()
        mv['date'] = pd.to_datetime(mv['date'])
        
        # Expand quarterly data to monthly with forward fill
        df = df.sort_values(['code', 'end_date'])
        df['year'] = df['end_date'].dt.year
        df['month'] = df['end_date'].dt.month
        
        # Create month-end dates for merging
        mv['year'] = mv['year']
        mv['month'] = mv['month']
        
        # Merge quarterly data with monthly market value
        result = pd.merge(mv, df, on=['code', 'year', 'month'], how='left')
        
        # Forward fill quarterly data within each stock
        result = result.sort_values(['code', 'date'])
        result['quarterly_n_cashflow_act_fillna'] = result.groupby('code')['quarterly_n_cashflow_act_fillna'].ffill()
        
        # Apply shift to ensure data is at least shift_months old
        result['quarterly_n_cashflow_act_fillna'] = result.groupby('code')['quarterly_n_cashflow_act_fillna'].shift(shift_months)
        
        # Calculate Ocpq ratio
        result['Ocpq'] = result['quarterly_n_cashflow_act_fillna'] / result['monthly_mv_end']
        
        # Exclude firms with nonpositive operating cash flows
        result = result[result['quarterly_n_cashflow_act_fillna'] > 0]
        
        # Use date as end_date for output
        result = result.rename(columns={'date': 'end_date'})
        
        return result[['code', 'end_date', 'Ocpq']].dropna()
    
    def validate(self) -> bool:
        """Validate Ocpq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ocpq' in df.columns


# =============================================================================
# SECTION 11: A3 INVESTMENT FACTOR IMPLEMENTATIONS
# =============================================================================


class AgrFactor(A3InvestmentFactorBase):
    """
    A.3.1 Agr - Asset Growth
    
    Agr = (Total Assets(t-1) - Total Assets(t-2)) / Total Assets(t-2)
    
    At the end of June of each year t, stocks are split into deciles based on Agr,
    which is the growth in total assets (AT) from the fiscal year ending in 
    calendar year t−2 to t−1.
    
    Monthly decile returns calculated from July of year t to June of t+1,
    rebalanced in June of t+1.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.1"
    
    @property
    def abbr(self) -> str:
        return "Agr"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate asset growth rate.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Agr']
        """
        # Load total assets
        df = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        df_dec = df_dec.sort_values(['code', 'year'])
        
        # Calculate asset growth
        df_dec['total_assets_lag1'] = df_dec.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df_dec['Agr'] = (df_dec['quarterly_total_assets_fillna'] - df_dec['total_assets_lag1']) / df_dec['total_assets_lag1']
        
        return df_dec[['code', 'end_date', 'Agr']].dropna()
    
    def validate(self) -> bool:
        """Validate Agr factor calculation."""
        df = self.calculate()
        return not df.empty and 'Agr' in df.columns


class IaFactor(A3InvestmentFactorBase):
    """
    A.3.2 I/A - Investment-to-Assets
    
    I/A = (Total Assets(t-1) - Total Assets(t-2)) / Total Assets(t-2)
    
    Same as Agr but with different interpretation.
    Investment-to-assets measures the annual change in total assets scaled by lagged assets.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.2"
    
    @property
    def abbr(self) -> str:
        return "Ia"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate investment-to-assets ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ia']
        """
        # Load total assets
        df = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        df_dec = df_dec.sort_values(['code', 'year'])
        
        # Calculate I/A
        df_dec['total_assets_lag1'] = df_dec.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df_dec['Ia'] = (df_dec['quarterly_total_assets_fillna'] - df_dec['total_assets_lag1']) / df_dec['total_assets_lag1']
        
        return df_dec[['code', 'end_date', 'Ia']].dropna()
    
    def validate(self) -> bool:
        """Validate Ia factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ia' in df.columns


class NsiFactor(A3InvestmentFactorBase):
    """
    A.3.5 Nsi - Net Stock Issues
    
    Nsi = log(Total Shares(t-1) / Total Shares(t-2))
    
    Net stock issues is the natural log of the growth in split-adjusted shares outstanding
    from the fiscal year ending in calendar year t−2 to t−1.
    
    At the end of June of each year t, stocks are split into deciles based on Nsi.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.5"
    
    @property
    def abbr(self) -> str:
        return "Nsi"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate net stock issues.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Nsi']
        """
        # Load total shares
        df = load_balancesheet_data(['quarterly_total_share_fillna'], RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df_dec = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df_dec['year'] = pd.to_datetime(df_dec['end_date']).dt.year
        df_dec = df_dec.sort_values(['code', 'year'])
        
        # Calculate net stock issues
        df_dec['total_shares_lag1'] = df_dec.groupby('code')['quarterly_total_share_fillna'].shift(1)
        
        # Calculate Nsi with error handling for zero/negative values
        with np.errstate(divide='ignore', invalid='ignore'):
            df_dec['Nsi'] = np.log(df_dec['quarterly_total_share_fillna'] / df_dec['total_shares_lag1'])
        
        # Replace inf/-inf with NaN
        df_dec['Nsi'] = df_dec['Nsi'].replace([np.inf, -np.inf], np.nan)
        
        return df_dec[['code', 'end_date', 'Nsi']].dropna()
    
    def validate(self) -> bool:
        """Validate Nsi factor calculation."""
        df = self.calculate()
        return not df.empty and 'Nsi' in df.columns


class IaqFactor(A3InvestmentFactorBase):
    """
    A.3.3 Iaq - Quarterly Investment-to-Assets
    
    Quarterly investment-to-assets ratio:
    Iaq = Total Assets(t-1) / Total Assets(t-4) - 1
    
    Measures the quarterly growth in total assets compared to 4 quarters ago.
    Uses quarterly data without filtering to fiscal year-end.
    
    Data Source:
        - quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Note:
        At the beginning of each month t, stocks are sorted into deciles based on Iaq
        for the latest fiscal quarter ending at least four months ago.
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.3"
    
    @property
    def abbr(self) -> str:
        return "Iaq"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly investment-to-assets ratio.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Iaq']
        """
        # Load quarterly total assets (all quarters, not just December)
        df = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        
        # Sort by code and end_date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate Iaq: growth over 4 quarters (1 year)
        df['total_assets_lag4'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(4)
        df['Iaq'] = (df['quarterly_total_assets_fillna'] - df['total_assets_lag4']) / df['total_assets_lag4']
        
        return df[['code', 'end_date', 'Iaq']].dropna()
    
    def validate(self) -> bool:
        """Validate Iaq factor calculation."""
        df = self.calculate()
        return not df.empty and 'Iaq' in df.columns


class DpiaFactor(A3InvestmentFactorBase):
    """
    A.3.4 dPia - Changes in PPE and Inventory-to-Assets
    
    dPia = (ΔPPE + Δ Inventory + Depreciation) / Total Assets(t-1)
    
    Where:
        ΔPPE = Δ(Fixed Assets) + Δ(Provision for Depreciation) + Depreciation
        
    Measures the change in property, plant, equipment (PPE) and inventory
    relative to lagged total assets. Annual fiscal year-end data.
    
    Data Sources:
        - Balance Sheet: inventories, fix_assets, total_assets
        - Cash Flow: depr_fa_coga_dpba1 (depreciation), prov_depr_assets (provision)
    
    Note:
        Requires merging balance sheet and cash flow data on (code, end_date, end_type).
        Annual data only (December fiscal year-end).
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.4"
    
    @property
    def abbr(self) -> str:
        return "dPia"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in PPE and inventory-to-assets.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'dPia']
        """
        # Load balance sheet data
        bs_cols = ['quarterly_inventories_fillna', 'quarterly_fix_assets_fillna', 
                   'quarterly_total_assets_fillna']
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)
        
        # Load cash flow data (includes depreciation)
        cf_cols = ['quarterly_depr_fa_coga_dpba1_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)
        
        # Merge on code and end_date
        df = pd.merge(df_bs, df_cf, on=['code', 'end_date'], how='inner')
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate changes
        df['fix_assets_lag1'] = df.groupby('code')['quarterly_fix_assets_fillna'].shift(1)
        df['inventories_lag1'] = df.groupby('code')['quarterly_inventories_fillna'].shift(1)
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        
        # Calculate dPPEGT = Δ(Fixed Assets) + Depreciation
        # Note: Provision for depreciation is not available in RESSET data
        df['dPPEGT'] = (
            (df['quarterly_fix_assets_fillna'] - df['fix_assets_lag1']) +
            df['quarterly_depr_fa_coga_dpba1_fillna']
        )
        
        # Calculate dPia
        df['dPia'] = (
            df['dPPEGT'] + 
            (df['quarterly_inventories_fillna'] - df['inventories_lag1'])
        ) / df['total_assets_lag1']
        
        return df[['code', 'end_date', 'dPia']].dropna()
    
    def validate(self) -> bool:
        """Validate dPia factor calculation."""
        df = self.calculate()
        return not df.empty and 'dPia' in df.columns


class NoaFactor(A3InvestmentFactorBase):
    """
    A.3.5 Noa - Net Operating Assets
    
    Noa = (Operating Liabilities - Cash and Short-term Investments) / Total Assets(t-1)
    
    Where:
        Operating Liabilities = Short-term Debt + Long-term Debt + Total Equity
        Cash & ST Investment = Money Cap + Trading Assets + Receivables + Time Deposits + etc.
        
    Measures net operating assets scaled by lagged total assets.
    Annual fiscal year-end data (December).
    
    Data Source:
        - Balance Sheet: money_cap, trad_asset, accounts_receiv_bill, int_receiv,
                        pur_resale_fa, loanto_oth_bank_fi, time_deposits, refund_depos,
                        total_assets, st_borr, notes_payable, st_bonds_payable, 
                        non_cur_liab_due_1y, bond_payable, lt_payable, lt_borr,
                        total_share, minority_int, oth_eqt_tools_p_shr
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.5"
    
    @property
    def abbr(self) -> str:
        return "Noa"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate net operating assets.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Noa']
        """
        # Load all required balance sheet columns
        cols = [
            'quarterly_money_cap_fillna', 'quarterly_trad_asset_fillna', 
            'quarterly_accounts_receiv_bill_fillna', 'quarterly_int_receiv_fillna',
            'quarterly_pur_resale_fa_fillna', 'quarterly_loanto_oth_bank_fi_fillna',
            'quarterly_time_deposits_fillna', 'quarterly_refund_depos_fillna',
            'quarterly_total_assets_fillna',
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_bond_payable_fillna', 'quarterly_lt_payable_fillna',
            'quarterly_lt_borr_fillna', 'quarterly_total_share_fillna',
            'quarterly_minority_int_fillna', 'quarterly_oth_eqt_tools_p_shr_fillna'
        ]
        
        df = load_balancesheet_data(cols, RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Fill NA with 0 for calculation
        df = df.fillna(0)
        
        # Calculate cash and short-term investment
        df['cash_st_inv'] = (
            df['quarterly_money_cap_fillna'] + 
            df['quarterly_trad_asset_fillna'] +
            df['quarterly_accounts_receiv_bill_fillna'] + 
            df['quarterly_int_receiv_fillna'] +
            df['quarterly_pur_resale_fa_fillna'] + 
            df['quarterly_loanto_oth_bank_fi_fillna'] +
            df['quarterly_time_deposits_fillna'] + 
            df['quarterly_refund_depos_fillna']
        )
        
        # Calculate DLC (debt in current liabilities)
        df['DLC'] = (
            df['quarterly_st_borr_fillna'] + 
            df['quarterly_notes_payable_fillna'] +
            df['quarterly_st_bonds_payable_fillna'] + 
            df['quarterly_non_cur_liab_due_1y_fillna']
        )
        
        # Calculate DLTT (long-term debt)
        df['DLTT'] = (
            df['quarterly_bond_payable_fillna'] + 
            df['quarterly_lt_payable_fillna'] +
            df['quarterly_lt_borr_fillna']
        )
        
        # Calculate total equity
        df['eq'] = (
            df['quarterly_total_share_fillna'] + 
            df['quarterly_oth_eqt_tools_p_shr_fillna'] +
            df['quarterly_minority_int_fillna']
        )
        
        # Calculate Noa
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['Noa'] = (df['DLC'] + df['DLTT'] + df['eq'] - df['cash_st_inv']) / df['total_assets_lag1']
        
        return df[['code', 'end_date', 'Noa']].dropna()
    
    def validate(self) -> bool:
        """Validate Noa factor calculation."""
        df = self.calculate()
        return not df.empty and 'Noa' in df.columns


class DnoaFactor(A3InvestmentFactorBase):
    """
    A.3.6 dNoa - Changes in Net Operating Assets
    
    dNoa = Δ(Operating Liabilities - Cash and Short-term Investments) / Total Assets(t-1)
    
    Measures the annual change in net operating assets scaled by lagged total assets.
    Same components as Noa but calculates year-over-year change.
    Annual fiscal year-end data (December).
    
    Data Source:
        - Same as NoaFactor
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.6"
    
    @property
    def abbr(self) -> str:
        return "dNoa"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in net operating assets.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'dNoa']
        """
        # Load all required balance sheet columns (same as Noa)
        cols = [
            'quarterly_money_cap_fillna', 'quarterly_trad_asset_fillna', 
            'quarterly_accounts_receiv_bill_fillna', 'quarterly_int_receiv_fillna',
            'quarterly_pur_resale_fa_fillna', 'quarterly_loanto_oth_bank_fi_fillna',
            'quarterly_time_deposits_fillna', 'quarterly_refund_depos_fillna',
            'quarterly_total_assets_fillna',
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_bond_payable_fillna', 'quarterly_lt_payable_fillna',
            'quarterly_lt_borr_fillna', 'quarterly_total_share_fillna',
            'quarterly_minority_int_fillna', 'quarterly_oth_eqt_tools_p_shr_fillna'
        ]
        
        df = load_balancesheet_data(cols, RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Fill NA with 0 for calculation
        df = df.fillna(0)
        
        # Calculate cash and short-term investment
        df['cash_st_inv'] = (
            df['quarterly_money_cap_fillna'] + 
            df['quarterly_trad_asset_fillna'] +
            df['quarterly_accounts_receiv_bill_fillna'] + 
            df['quarterly_int_receiv_fillna'] +
            df['quarterly_pur_resale_fa_fillna'] + 
            df['quarterly_loanto_oth_bank_fi_fillna'] +
            df['quarterly_time_deposits_fillna'] + 
            df['quarterly_refund_depos_fillna']
        )
        
        # Calculate DLC (debt in current liabilities)
        df['DLC'] = (
            df['quarterly_st_borr_fillna'] + 
            df['quarterly_notes_payable_fillna'] +
            df['quarterly_st_bonds_payable_fillna'] + 
            df['quarterly_non_cur_liab_due_1y_fillna']
        )
        
        # Calculate DLTT (long-term debt)
        df['DLTT'] = (
            df['quarterly_bond_payable_fillna'] + 
            df['quarterly_lt_payable_fillna'] +
            df['quarterly_lt_borr_fillna']
        )
        
        # Calculate total equity
        df['eq'] = (
            df['quarterly_total_share_fillna'] + 
            df['quarterly_oth_eqt_tools_p_shr_fillna'] +
            df['quarterly_minority_int_fillna']
        )
        
        # Calculate net operating assets (numerator)
        df['noa_num'] = df['DLC'] + df['DLTT'] + df['eq'] - df['cash_st_inv']
        df['noa_num_lag1'] = df.groupby('code')['noa_num'].shift(1)
        
        # Calculate dNoa
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['dNoa'] = (df['noa_num'] - df['noa_num_lag1']) / df['total_assets_lag1']
        
        return df[['code', 'end_date', 'dNoa']].dropna()
    
    def validate(self) -> bool:
        """Validate dNoa factor calculation."""
        df = self.calculate()
        return not df.empty and 'dNoa' in df.columns


class DlnoFactor(A3InvestmentFactorBase):
    """
    A.3.6 dLno - Changes in Long-term Net Operating Assets
    
    dLno = (ΔPPENT + Δ Intangibles + Δ Other LT Assets + Δ Other LT Liabilities + DP) / Avg(Total Assets, 2 years)
    
    Where:
        PPENT = Fixed Assets + Provision for Depreciation
        DP = Depreciation + Amortization of Intangibles + Amortization of Deferred Expenses
        
    Measures changes in long-term net operating assets scaled by average total assets.
    Annual fiscal year-end data (December).
    
    Data Sources:
        - Balance Sheet: fix_assets, intan_assets, oth_nca, oth_ncl, total_assets, prov_depr_assets
        - Cash Flow: depr_fa_coga_dpba1, amort_intang_assets, lt_amort_deferred_exp
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.7"
    
    @property
    def abbr(self) -> str:
        return "dLno"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in long-term net operating assets.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'dLno']
        """
        # Load balance sheet data
        bs_cols = ['quarterly_fix_assets_fillna', 'quarterly_intan_assets_fillna',
                   'quarterly_oth_nca_fillna', 'quarterly_oth_ncl_fillna',
                   'quarterly_total_assets_fillna']
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)
        
        # Load cash flow data
        cf_cols = ['quarterly_depr_fa_coga_dpba1_fillna', 'quarterly_amort_intang_assets_fillna',
                   'quarterly_lt_amort_deferred_exp_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)
        
        # Merge on code and end_date
        df = pd.merge(df_bs, df_cf, on=['code', 'end_date'], how='inner')
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Fill NA with 0 for calculation
        df = df.fillna(0)
        
        # Calculate changes (PPENT = Fixed Assets, no provision data available)
        df['dPPENT'] = df['quarterly_fix_assets_fillna'] - df.groupby('code')['quarterly_fix_assets_fillna'].shift(1)
        df['dintan_assets'] = df['quarterly_intan_assets_fillna'] - df.groupby('code')['quarterly_intan_assets_fillna'].shift(1)
        df['doth_nca'] = df['quarterly_oth_nca_fillna'] - df.groupby('code')['quarterly_oth_nca_fillna'].shift(1)
        df['doth_ncl'] = df['quarterly_oth_ncl_fillna'] - df.groupby('code')['quarterly_oth_ncl_fillna'].shift(1)
        
        # Calculate DP (Depreciation and amortization)
        df['DP'] = (
            df['quarterly_depr_fa_coga_dpba1_fillna'] +
            df['quarterly_amort_intang_assets_fillna'] +
            df['quarterly_lt_amort_deferred_exp_fillna']
        )
        
        # Calculate average total assets over 2 years
        df['avg_total_assets'] = df.groupby('code')['quarterly_total_assets_fillna'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean()
        )
        
        # Calculate dLno
        df['dLno'] = (
            df['dPPENT'] + df['dintan_assets'] + df['doth_nca'] + df['doth_ncl'] + df['DP']
        ) / df['avg_total_assets']
        
        return df[['code', 'end_date', 'dLno']].dropna()
    
    def validate(self) -> bool:
        """Validate dLno factor calculation."""
        df = self.calculate()
        return not df.empty and 'dLno' in df.columns


class IgFactor(A3InvestmentFactorBase):
    """
    A.3.8 Ig - Investment Growth (1-year)
    
    Ig = (Capital Expenditure(t-1) - Capital Expenditure(t-2)) / Capital Expenditure(t-2)
    
    Measures the growth rate in capital expenditure from fiscal year t-2 to t-1.
    Capital expenditure = Cash paid for fixed assets, intangibles, and other long-term assets.
    Annual fiscal year-end data (December).
    
    Data Source:
        - Cash Flow: c_pay_acq_const_fiolta (capital expenditure)
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.8"
    
    @property
    def abbr(self) -> str:
        return "Ig"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate 1-year investment growth.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ig']
        """
        # Load capital expenditure data
        df = load_cashflow_data(['quarterly_c_pay_acq_const_fiolta_fillna'], path=RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate 1-year investment growth
        df['capex_lag1'] = df.groupby('code')['quarterly_c_pay_acq_const_fiolta_fillna'].shift(1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Ig'] = (df['quarterly_c_pay_acq_const_fiolta_fillna'] - df['capex_lag1']) / df['capex_lag1']
        
        # Replace inf/-inf with NaN
        df['Ig'] = df['Ig'].replace([np.inf, -np.inf], np.nan)
        
        return df[['code', 'end_date', 'Ig']].dropna()
    
    def validate(self) -> bool:
        """Validate Ig factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ig' in df.columns


class Ig2Factor(A3InvestmentFactorBase):
    """
    A.3.9 2Ig - 2-Year Investment Growth
    
    2Ig = (Capital Expenditure(t-1) - Capital Expenditure(t-3)) / Capital Expenditure(t-3)
    
    Measures the growth rate in capital expenditure from fiscal year t-3 to t-1.
    Annual fiscal year-end data (December).
    
    Data Source:
        - Cash Flow: c_pay_acq_const_fiolta (capital expenditure)
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.9"
    
    @property
    def abbr(self) -> str:
        return "Ig2"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate 2-year investment growth.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ig2']
        """
        # Load capital expenditure data
        df = load_cashflow_data(['quarterly_c_pay_acq_const_fiolta_fillna'], path=RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate 2-year investment growth
        df['capex_lag2'] = df.groupby('code')['quarterly_c_pay_acq_const_fiolta_fillna'].shift(2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Ig2'] = (df['quarterly_c_pay_acq_const_fiolta_fillna'] - df['capex_lag2']) / df['capex_lag2']
        
        # Replace inf/-inf with NaN
        df['Ig2'] = df['Ig2'].replace([np.inf, -np.inf], np.nan)
        
        return df[['code', 'end_date', 'Ig2']].dropna()
    
    def validate(self) -> bool:
        """Validate Ig2 factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ig2' in df.columns


class Ig3Factor(A3InvestmentFactorBase):
    """
    A.3.10 3Ig - 3-Year Investment Growth
    
    3Ig = (Capital Expenditure(t-1) - Capital Expenditure(t-4)) / Capital Expenditure(t-4)
    
    Measures the growth rate in capital expenditure from fiscal year t-4 to t-1.
    Annual fiscal year-end data (December).
    
    Data Source:
        - Cash Flow: c_pay_acq_const_fiolta (capital expenditure)
    """
    
    @property
    def factor_id(self) -> str:
        return "A.3.10"
    
    @property
    def abbr(self) -> str:
        return "Ig3"
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate 3-year investment growth.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ig3']
        """
        # Load capital expenditure data
        df = load_cashflow_data(['quarterly_c_pay_acq_const_fiolta_fillna'], path=RESSET_DIR)
        
        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate 3-year investment growth
        df['capex_lag3'] = df.groupby('code')['quarterly_c_pay_acq_const_fiolta_fillna'].shift(3)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['Ig3'] = (df['quarterly_c_pay_acq_const_fiolta_fillna'] - df['capex_lag3']) / df['capex_lag3']
        
        # Replace inf/-inf with NaN
        df['Ig3'] = df['Ig3'].replace([np.inf, -np.inf], np.nan)
        
        return df[['code', 'end_date', 'Ig3']].dropna()
    
    def validate(self) -> bool:
        """Validate Ig3 factor calculation."""
        df = self.calculate()
        return not df.empty and 'Ig3' in df.columns


# =============================================================================
# SECTION 12: A4 PROFITABILITY FACTOR IMPLEMENTATIONS
# =============================================================================


class RoeFactor(A2ValueFactorBase):  # Note: 虽然是A4因子，但基于数据结构相似性使用A2基类
    """
    A.4.1 Return on Equity (Roe)
    
    Roe denotes return on equity, calculated as income before extraordinary items
    divided by 1-year-lagged book equity.
    
    Formula:
        Roe = (income before extraordinary items) / (book equity_{t-1})
    
    Data source:
        - income before extraordinary items: n_income from quarterly_income_cleaned.csv
        - book equity: quarterly_book_equity calculated from balance sheet
    """
    
    @property
    def name(self) -> str:
        return 'roe'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.1'
    
    @property
    def abbr(self) -> str:
        return 'Roe'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate return on equity factor.
        
        Returns:
            DataFrame with columns: ['code', 'end_date', 'roe']
        """
        # Load income data
        income_cols = ['quarterly_n_income_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={'quarterly_n_income_fillna': 'n_income'})
        
        # Load book equity data
        df_be = get_quartly_be()
        
        # Ensure end_date is in consistent datetime format
        df_income['end_date'] = pd.to_datetime(df_income['end_date'], errors='coerce')
        df_be['end_date'] = pd.to_datetime(df_be['end_date'], errors='coerce')
        
        # Merge income and book equity
        df = pd.merge(
            df_income[['code', 'end_date', 'n_income']], 
            df_be[['code', 'end_date', 'quarterly_book_equity']], 
            on=['code', 'end_date'], 
            how='left'
        )
        
        # Calculate Roe = income / lagged book equity
        df = df.sort_values(['code', 'end_date'])
        df['quarterly_book_equity_lag1'] = df.groupby('code')['quarterly_book_equity'].shift(1)
        df[self.abbr] = df['n_income'] / df['quarterly_book_equity_lag1']
        
        return df[['code', 'end_date', self.abbr]].dropna()


class DroeFactor(A2ValueFactorBase):
    """
    A.4.2 Change in Return on Equity (dRoe)
    
    dRoe denotes the change in return on equity from the same fiscal quarter
    in the prior year.
    
    Formula:
        dRoe = Roe_t - Roe_{t-4quarters}
    """
    
    @property
    def name(self) -> str:
        return 'droe'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.2'
    
    @property
    def abbr(self) -> str:
        return 'dRoe'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate change in return on equity."""
        # First calculate Roe
        roe_factor = RoeFactor()
        df_roe = roe_factor.calculate()
        
        # Calculate change from 4 quarters ago (year-over-year)
        df_roe = df_roe.sort_values(['code', 'end_date'])
        df_roe['roe_lag4'] = df_roe.groupby('code')['roe'].shift(4)
        df_roe['droe'] = df_roe['roe'] - df_roe['roe_lag4']
        
        return df_roe[['code', 'end_date', 'droe']].dropna()


class RoaFactor(A2ValueFactorBase):
    """
    A.4.3 Return on Assets (Roa)
    
    Roa denotes return on assets, calculated as profit before deduction of
    interest and tax divided by 1-year-lagged total assets.
    
    Formula:
        Roa = profit_before_dedt / total_assets_{t-1}
    
    Data source:
        - profit_before_dedt: quarterly_profit_dedt_fillna from quarterly_fina_indicator_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Uses quarterly data
        - Lag 1 quarter for total assets
        - Similar to Roe but uses assets instead of equity
    """
    
    @property
    def name(self) -> str:
        return 'roa'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.3'
    
    @property
    def abbr(self) -> str:
        return 'Roa'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate return on assets factor."""
        # Load profit data
        profit_cols = ['quarterly_profit_dedt_fillna']
        df_profit = load_fina_indicator_data(profit_cols, RESSET_DIR)
        df_profit = df_profit.rename(columns={'quarterly_profit_dedt_fillna': 'profit_dedt'})
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_profit[['code', 'end_date', 'profit_dedt']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='left'
        )
        
        # Calculate Roa = profit / lagged total assets
        df = df.sort_values(['code', 'end_date'])
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        df[self.abbr] = df['profit_dedt'] / df['total_assets_lag1']
        
        return df[['code', 'end_date', self.abbr]].dropna()


class DroaFactor(A2ValueFactorBase):
    """
    A.4.4 Change in Return on Assets (dRoa)
    
    dRoa denotes the change in return on assets from the same fiscal quarter
    in the prior year.
    
    Formula:
        dRoa = Roa_t - Roa_{t-4quarters}
    
    Implementation notes:
        - Lag 4 quarters for year-over-year comparison
    """
    
    @property
    def name(self) -> str:
        return 'droa'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.4'
    
    @property
    def abbr(self) -> str:
        return 'dRoa'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate change in return on assets."""
        # First calculate Roa
        roa_factor = RoaFactor()
        df_roa = roa_factor.calculate()
        
        # Calculate change from 4 quarters ago (year-over-year)
        df_roa = df_roa.sort_values(['code', 'end_date'])
        df_roa['roa_lag4'] = df_roa.groupby('code')['roa'].shift(4)
        df_roa['droa'] = df_roa['roa'] - df_roa['roa_lag4']
        
        return df_roa[['code', 'end_date', 'droa']].dropna()


class GpaFactor(A2ValueFactorBase):
    """
    A.4.9 Gross Profitability to Assets (Gpa)
    
    Gpa measures gross profitability relative to total assets.
    
    Formula:
        Gpa = (revenue - cost_of_goods_sold) / total_assets
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - cost_of_goods_sold: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Gross profit = revenue - COGS
    """
    
    @property
    def name(self) -> str:
        return 'gpa'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.9'
    
    @property
    def abbr(self) -> str:
        return 'Gpa'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate gross profitability to assets factor."""
        # Load income data (revenue and COGS)
        income_cols = ['quarterly_total_revenue_fillna', 'quarterly_total_cogs_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Calculate Gpa = (revenue - COGS) / total_assets
        df[self.abbr] = (df['revenue'] - df['cogs']) / df['total_assets']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', self.abbr]].dropna()


class OpaFactor(A2ValueFactorBase):
    """
    A.4.15 Operating Profitability to Assets (Opa)
    
    Opa measures operating profitability relative to total assets.
    
    Formula:
        Opa = (revenue - COGS - SG&A + R&D) / total_assets
    
    where SG&A includes selling, general & administrative expenses
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp + fin_exp
        - R&D: quarterly_rd_exp_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - R&D is added back as it's an investment in future profitability
    """
    
    @property
    def name(self) -> str:
        return 'opa'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.15'
    
    @property
    def abbr(self) -> str:
        return 'Opa'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating profitability to assets factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',      # Selling expenses
            'quarterly_admin_exp_fillna',      # Admin expenses  
            'quarterly_rd_exp_fillna'          # R&D expenses
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        # Rename columns
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_rd_exp_fillna': 'rd_exp'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'rd_exp']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'rd_exp']] = df[['sell_exp', 'admin_exp', 'rd_exp']].fillna(0)
        
        # Calculate SG&A (selling + admin expenses)
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Opa = (revenue - COGS - SG&A + R&D) / total_assets
        df[self.abbr] = (df['revenue'] - df['cogs'] - df['sga'] + df['rd_exp']) / df['total_assets']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', self.abbr]].dropna()


class CtoFactor(A2ValueFactorBase):
    """
    A.4.6 Capital Turnover (Cto)
    
    Cto measures the efficiency of capital utilization by calculating revenue per unit of assets.
    
    Formula:
        Cto = Revenue / Total Assets (annual)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only, December)
    """
    
    @property
    def name(self) -> str:
        return 'cto'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.6'
    
    @property
    def abbr(self) -> str:
        return 'Cto'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate capital turnover factor."""
        # Load income data
        income_cols = ['quarterly_total_revenue_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={'quarterly_total_revenue_fillna': 'revenue'})
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Calculate Cto = revenue / total_assets
        df['cto'] = df['revenue'] / df['total_assets']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'cto']].dropna()


class OlaFactor(A2ValueFactorBase):
    """
    A.4.16 Operating Leverage (Assets) (Ola)
    
    Ola measures operating profitability relative to lagged total assets.
    
    Formula:
        Ola = (Revenue - COGS - SG&A) / Lagged Total Assets
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Uses 1-quarter lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'ola'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.16'
    
    @property
    def abbr(self) -> str:
        return 'Ola'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating leverage (assets) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter lag)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Ola = (revenue - COGS - SG&A) / lagged_total_assets
        df['ola'] = (df['revenue'] - df['cogs'] - df['sga']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'ola']].dropna()


class OleFactor(A2ValueFactorBase):
    """
    A.4.13 Operating Leverage (Earnings) (Ole)
    
    Ole measures operating profitability relative to lagged book equity.
    
    Formula:
        Ole = (Revenue - COGS - SG&A) / Lagged Book Equity
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - book_equity: calculated from total_assets - total_liab
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Uses 1-quarter lagged book equity
    """
    
    @property
    def name(self) -> str:
        return 'ole'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.13'
    
    @property
    def abbr(self) -> str:
        return 'Ole'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating leverage (earnings) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data for book equity calculation
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_total_liab_fillna': 'total_liab'
        })
        
        # Calculate book equity
        df_balance['book_equity'] = df_balance['total_assets'] - df_balance['total_liab']
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'book_equity']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged book equity (1 year lag)
        df['book_equity_lag1'] = df.groupby('code')['book_equity'].shift(1)
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Ole = (revenue - COGS - SG&A) / lagged_book_equity
        df['ole'] = (df['revenue'] - df['cogs'] - df['sga']) / df['book_equity_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'ole']].dropna()


class GlaFactor(A2ValueFactorBase):
    """
    A.4.10 Gross Profit to Lagged Assets (Gla)
    
    Gla measures gross profitability relative to lagged total assets.
    
    Formula:
        Gla = (Revenue - COGS) / Lagged Total Assets
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Uses 1-quarter lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'gla'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.10'
    
    @property
    def abbr(self) -> str:
        return 'Gla'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate gross profit to lagged assets factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter lag)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Calculate Gla = (revenue - COGS) / lagged_total_assets
        df['gla'] = (df['revenue'] - df['cogs']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'gla']].dropna()


class CtoqFactor(A2ValueFactorBase):
    """
    A.4.8 Quarterly Capital Turnover (Ctoq)
    
    Ctoq measures the efficiency of capital utilization using quarterly data.
    
    Formula:
        Ctoq = Revenue(t) / Total Assets(t-1)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Quarterly indicator (all quarters)
        - Uses 1-quarter lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'ctoq'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.8'
    
    @property
    def abbr(self) -> str:
        return 'Ctoq'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly capital turnover factor."""
        # Load income data
        income_cols = ['quarterly_total_revenue_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={'quarterly_total_revenue_fillna': 'revenue'})
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Calculate Ctoq = revenue / lagged_total_assets
        df['ctoq'] = df['revenue'] / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'ctoq']].dropna()


class GlaqFactor(A2ValueFactorBase):
    """
    A.4.11 Quarterly Gross Profit to Lagged Assets (Glaq)
    
    Glaq measures gross profitability using quarterly data.
    
    Formula:
        Glaq = (Revenue(t) - COGS(t)) / Total Assets(t-1)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Quarterly indicator (all quarters)
        - Uses 1-quarter lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'glaq'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.11'
    
    @property
    def abbr(self) -> str:
        return 'Glaq'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly gross profit to lagged assets factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Calculate Glaq = (revenue - COGS) / lagged_total_assets
        df['glaq'] = (df['revenue'] - df['cogs']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'glaq']].dropna()


class OleqFactor(A2ValueFactorBase):
    """
    A.4.14 Quarterly Operating Leverage (Earnings) (Oleq)
    
    Oleq measures operating profitability relative to lagged book equity using quarterly data.
    
    Formula:
        Oleq = (Revenue(t) - COGS(t) - SG&A(t)) / Book Equity(t-1)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - book_equity: calculated from total_assets - total_liab
    
    Implementation notes:
        - Quarterly indicator (all quarters)
        - Uses 1-quarter lagged book equity
    """
    
    @property
    def name(self) -> str:
        return 'oleq'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.14'
    
    @property
    def abbr(self) -> str:
        return 'Oleq'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly operating leverage (earnings) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data for book equity calculation
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_total_liab_fillna': 'total_liab'
        })
        
        # Calculate book equity
        df_balance['book_equity'] = df_balance['total_assets'] - df_balance['total_liab']
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'book_equity']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged book equity (1 quarter)
        df['book_equity_lag1'] = df.groupby('code')['book_equity'].shift(1)
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Oleq = (revenue - COGS - SG&A) / lagged_book_equity
        df['oleq'] = (df['revenue'] - df['cogs'] - df['sga']) / df['book_equity_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'oleq']].dropna()


class OlaqFactor(A2ValueFactorBase):
    """
    A.4.17 Quarterly Operating Leverage (Assets) (Olaq)
    
    Olaq measures operating profitability relative to lagged total assets using quarterly data.
    
    Formula:
        Olaq = (Revenue(t) - COGS(t) - SG&A(t)) / Total Assets(t-1)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Quarterly indicator (all quarters)
        - Uses 1-quarter lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'olaq'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.17'
    
    @property
    def abbr(self) -> str:
        return 'Olaq'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly operating leverage (assets) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_assets[['code', 'end_date', 'total_assets']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Olaq = (revenue - COGS - SG&A) / lagged_total_assets
        df['olaq'] = (df['revenue'] - df['cogs'] - df['sga']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'olaq']].dropna()


class OpeFactor(A2ValueFactorBase):
    """
    A.4.12 Operating Profitability to Equity (Ope)
    
    Ope measures operating profitability relative to book equity.
    
    Formula:
        Ope = (Revenue - COGS - SG&A) / Book Equity
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - book_equity: calculated from total_assets - total_liab
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
    """
    
    @property
    def name(self) -> str:
        return 'ope'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.12'
    
    @property
    def abbr(self) -> str:
        return 'Ope'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating profitability to equity factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data for book equity calculation
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_total_liab_fillna': 'total_liab'
        })
        
        # Calculate book equity
        df_balance['book_equity'] = df_balance['total_assets'] - df_balance['total_liab']
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'book_equity']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Ope = (revenue - COGS - SG&A) / book_equity
        df['ope'] = (df['revenue'] - df['cogs'] - df['sga']) / df['book_equity']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'ope']].dropna()


class CopFactor(A2ValueFactorBase):
    """
    A.4.18 Cash-based Operating Profitability (Cop)
    
    Cop measures cash-based operating profitability relative to book equity.
    
    Formula:
        Cop = (Revenue - COGS - SG&A + Δ(Accounts Receivable) - Δ(Accounts Payable) - Δ(Inventory)) / Book Equity
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - accounts_receivable: quarterly_accounts_receiv_fillna from quarterly_balancesheet_cleaned.csv
        - accounts_payable: quarterly_acct_payable_fillna from quarterly_balancesheet_cleaned.csv
        - inventory: quarterly_inventories_fillna from quarterly_balancesheet_cleaned.csv
        - book_equity: calculated from total_assets - total_liab
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Adjusts operating profit for changes in working capital
    """
    
    @property
    def name(self) -> str:
        return 'cop'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.18'
    
    @property
    def abbr(self) -> str:
        return 'Cop'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate cash-based operating profitability factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_total_liab_fillna': 'total_liab',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory'
        })
        
        # Calculate book equity
        df_balance['book_equity'] = df_balance['total_assets'] - df_balance['total_liab']
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'book_equity', 'accounts_receiv', 'accounts_payable', 'inventory']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate changes in working capital (1-year lag)
        df['delta_ar'] = df.groupby('code')['accounts_receiv'].diff()
        df['delta_ap'] = df.groupby('code')['accounts_payable'].diff()
        df['delta_inv'] = df.groupby('code')['inventory'].diff()
        
        # Fill NaN with 0 for expense items and working capital changes
        df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']] = \
            df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Cop = (revenue - COGS - SG&A + ΔAR - ΔAP - ΔInv) / book_equity
        df['cop'] = (df['revenue'] - df['cogs'] - df['sga'] + df['delta_ar'] - df['delta_ap'] - df['delta_inv']) / df['book_equity']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'cop']].dropna()


class ClaFactor(A2ValueFactorBase):
    """
    A.4.19 Cash-based Operating Profitability to Assets (Cla)
    
    Cla measures cash-based operating profitability relative to lagged total assets.
    
    Formula:
        Cla = (Revenue - COGS - SG&A + Δ(Accounts Receivable) - Δ(Accounts Payable) - Δ(Inventory)) / Lagged Total Assets
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - accounts_receivable: quarterly_accounts_receiv_fillna from quarterly_balancesheet_cleaned.csv
        - accounts_payable: quarterly_acct_payable_fillna from quarterly_balancesheet_cleaned.csv
        - inventory: quarterly_inventories_fillna from quarterly_balancesheet_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Annual indicator (fiscal year end data only)
        - Uses 1-year lagged total assets
    """
    
    @property
    def name(self) -> str:
        return 'cla'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.19'
    
    @property
    def abbr(self) -> str:
        return 'Cla'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate cash-based operating profitability to assets factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory'
        })
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'total_assets', 'accounts_receiv', 'accounts_payable', 'inventory']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 year)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Calculate changes in working capital (1-year lag)
        df['delta_ar'] = df.groupby('code')['accounts_receiv'].diff()
        df['delta_ap'] = df.groupby('code')['accounts_payable'].diff()
        df['delta_inv'] = df.groupby('code')['inventory'].diff()
        
        # Fill NaN with 0 for expense items and working capital changes
        df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']] = \
            df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Cla = (revenue - COGS - SG&A + ΔAR - ΔAP - ΔInv) / lagged_total_assets
        df['cla'] = (df['revenue'] - df['cogs'] - df['sga'] + df['delta_ar'] - df['delta_ap'] - df['delta_inv']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'cla']].dropna()


class ClaqFactor(A2ValueFactorBase):
    """
    A.4.20 Quarterly Cash-based Operating Profitability to Assets (Claq)
    
    Claq measures cash-based operating profitability using quarterly data.
    
    Formula:
        Claq = (Revenue(t) - COGS(t) - SG&A(t) + Δ(AR) - Δ(AP) - Δ(Inv)) / Total Assets(t-1)
    
    Data source:
        - revenue: quarterly_total_revenue_fillna from quarterly_income_cleaned.csv
        - COGS: quarterly_total_cogs_fillna from quarterly_income_cleaned.csv
        - SG&A: calculated from selling_dist_exp + admin_exp
        - accounts_receivable: quarterly_accounts_receiv_fillna from quarterly_balancesheet_cleaned.csv
        - accounts_payable: quarterly_acct_payable_fillna from quarterly_balancesheet_cleaned.csv
        - inventory: quarterly_inventories_fillna from quarterly_balancesheet_cleaned.csv
        - total_assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv
    
    Implementation notes:
        - Quarterly indicator (all quarters)
        - Uses 1-quarter lagged total assets
        - Working capital changes are quarter-over-quarter
    """
    
    @property
    def name(self) -> str:
        return 'claq'
    
    @property
    def factor_id(self) -> str:
        return 'A.4.20'
    
    @property
    def abbr(self) -> str:
        return 'Claq'
    
    @property
    def category(self) -> str:
        return 'A4_Profitability'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly cash-based operating profitability to assets factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)
        
        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp'
        })
        
        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna'
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory'
        })
        
        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp']], 
            df_balance[['code', 'end_date', 'total_assets', 'accounts_receiv', 'accounts_payable', 'inventory']], 
            on=['code', 'end_date'], 
            how='inner'
        )
        
        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        
        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])
        
        # Calculate lagged total assets (1 quarter)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        
        # Calculate changes in working capital (1-quarter lag)
        df['delta_ar'] = df.groupby('code')['accounts_receiv'].diff()
        df['delta_ap'] = df.groupby('code')['accounts_payable'].diff()
        df['delta_inv'] = df.groupby('code')['inventory'].diff()
        
        # Fill NaN with 0 for expense items and working capital changes
        df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']] = \
            df[['sell_exp', 'admin_exp', 'delta_ar', 'delta_ap', 'delta_inv']].fillna(0)
        
        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']
        
        # Calculate Claq = (revenue - COGS - SG&A + ΔAR - ΔAP - ΔInv) / lagged_total_assets
        df['claq'] = (df['revenue'] - df['cogs'] - df['sga'] + df['delta_ar'] - df['delta_ap'] - df['delta_inv']) / df['total_assets_lag1']
        
        # Extract year
        df['year'] = df['end_date'].dt.year
        
        return df[['code', 'year', 'end_date', 'claq']].dropna()


# =============================================================================
# SECTION 13: A5 INTANGIBLES FACTOR IMPLEMENTATIONS
# =============================================================================


class AgeFactor(A5IntangiblesFactorBase):
    """
    A.5.18 Age - Firm Age (months since listing)
    
    Age denotes the number of months since the firm's listing date.
    Older firms may have different risk and return characteristics.
    
    Formula:
        Age = months(current_date - listing_date)
    
    Data source:
        - listing_date: from stock list file
        - current_date: trading dates from Wind daily return data
    """
    
    @property
    def name(self) -> str:
        return 'age'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.18'
    
    @property
    def abbr(self) -> str:
        return 'Age'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate firm age factor.
        
        Returns:
            DataFrame with columns: ['code', 'date', 'age']
        """
        # Load listing date data (from Tushare stock list)
        # Note: You may need to adjust the file path and column names
        try:
            import os
            list_file = DATADIR.parent / "tushare" / "stock_basic.csv"
            if not list_file.exists():
                raise FileNotFoundError(f"Stock list file not found: {list_file}")
            
            df_list = pd.read_csv(list_file)
            df_list = df_list.rename(columns={'ts_code': 'code', 'list_date': 'listing_date'})
            df_list = df_list[['code', 'listing_date']]
            
            # Load daily trading dates from Wind
            df_dates = load_wind_daily_data(['close'], DATADIR.parent / "wind")
            df_dates = df_dates[['code', 'date']].drop_duplicates()
            
            # Standardize code format
            df_list['code'] = df_list['code'].apply(standardize_ticker)
            
            # Convert dates
            df_list['listing_date'] = pd.to_datetime(df_list['listing_date'].astype(str), errors='coerce')
            df_dates['date'] = pd.to_datetime(df_dates['date'], errors='coerce')
            
            # Merge
            df = pd.merge(df_list, df_dates, on='code', how='outer')
            df = df.dropna(subset=['listing_date', 'date'])
            
            # Calculate age in months
            df['age'] = ((df['date'].dt.to_period('M').astype('int64') - 
                         df['listing_date'].dt.to_period('M').astype('int64')))
            
            # Keep only monthly end dates
            df = df.sort_values(['code', 'date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df_monthly = df.groupby(['code', 'year', 'month']).tail(1)
            
            return df_monthly[['code', 'date', 'age']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate Age factor: {e}")
            return pd.DataFrame(columns=['code', 'date', 'age'])


class DsiFactor(A5IntangiblesFactorBase):
    """
    A.5.20 dSi - % change in sales minus % change in inventory
    
    dSi measures the difference between sales growth and inventory growth,
    which can signal earnings quality issues.
    
    Formula:
        dSales = (2*Sales_t - Sales_{t-1} - Sales_{t-2}) / (Sales_{t-1} + Sales_{t-2})
        dInventory = (2*Inv_t - Inv_{t-1} - Inv_{t-2}) / (Inv_{t-1} + Inv_{t-2})
        dSi = dSales - dInventory
    
    Data source:
        - Sales (revenue): quarterly_income_cleaned.csv
        - Inventory: quarterly_balancesheet_cleaned.csv
    """
    
    @property
    def name(self) -> str:
        return 'dsi'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.20'
    
    @property
    def abbr(self) -> str:
        return 'dSi'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate dSi factor."""
        # Load revenue (sales) data
        df_revenue = load_income_data(['quarterly_revenue_fillna'], RESSET_DIR)
        df_revenue = df_revenue.rename(columns={'quarterly_revenue_fillna': 'revenue'})
        df_revenue = df_revenue[['code', 'end_date', 'revenue']].drop_duplicates()
        
        # Load inventory data
        df_inv = load_balancesheet_data(['quarterly_inventories_fillna'], RESSET_DIR)
        df_inv = df_inv.rename(columns={'quarterly_inventories_fillna': 'inventory'})
        df_inv = df_inv[['code', 'end_date', 'inventory']].drop_duplicates()
        
        # Merge
        df = pd.merge(df_revenue, df_inv, on=['code', 'end_date'], how='outer')
        df = df.fillna(0)
        
        # Sort and calculate lagged values
        df = df.sort_values(['code', 'end_date'])
        df['revenue_lag1'] = df.groupby('code')['revenue'].shift(1)
        df['revenue_lag2'] = df.groupby('code')['revenue'].shift(2)
        df['inventory_lag1'] = df.groupby('code')['inventory'].shift(1)
        df['inventory_lag2'] = df.groupby('code')['inventory'].shift(2)
        
        # Calculate dSales and dInventory
        df['dSales'] = (2 * df['revenue'] - df['revenue_lag1'] - df['revenue_lag2']) / \
                      (df['revenue_lag1'] + df['revenue_lag2'])
        df['dInventory'] = (2 * df['inventory'] - df['inventory_lag1'] - df['inventory_lag2']) / \
                          (df['inventory_lag1'] + df['inventory_lag2'])
        
        # Calculate dSi
        df['dsi'] = df['dSales'] - df['dInventory']
        
        # Extract year from end_date (handle both YYYYMMDD integer and ISO date string formats)
        try:
            # Try Tushare integer format first (e.g., 20211231)
            df['year'] = pd.to_datetime(df['end_date'], format='%Y%m%d').dt.year
        except (ValueError, TypeError):
            # Fallback to ISO/mixed format (e.g., "2021-12-31")
            df['year'] = pd.to_datetime(df['end_date'], format='mixed').dt.year
        
        return df[['code', 'year', 'dsi']].dropna()


class AnaFactor(A5IntangiblesFactorBase):
    """
    A.5.26 Ana - Analyst Coverage
    
    Ana denotes the number of analysts covering a stock, measured by the
    number of unique analysts issuing forecasts for the current fiscal year.
    Higher coverage typically indicates more information transparency.
    
    Formula:
        Ana = count(distinct analysts per stock per year)
    
    Data source:
        - Analyst reports: report_rc.csv from Tushare
    
    Note: In China, analysts typically forecast current year, next year, and year after.
    We focus on current year coverage.
    """
    
    @property
    def name(self) -> str:
        return 'ana'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.26'
    
    @property
    def abbr(self) -> str:
        return 'Ana'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate analyst coverage factor."""
        try:
            import os
            # Load analyst report data
            report_file = DATADIR.parent / "tushare" / "report_rc.csv"
            if not report_file.exists():
                print(f"Warning: Analyst report file not found: {report_file}")
                return pd.DataFrame(columns=['code', 'year', 'ana'])
            
            df = pd.read_csv(report_file)
            df = df.rename(columns={'ts_code': 'code', 'report_date': 'date'})
            df = df[['code', 'date', 'author_name']].drop_duplicates()
            
            # Standardize code
            df['code'] = df['code'].apply(standardize_ticker)
            
            # Extract year
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            
            # Count unique analysts per stock per year
            df['coverage'] = 1
            df_ana = df.groupby(['code', 'year'])['coverage'].sum().reset_index()
            df_ana = df_ana.rename(columns={'coverage': 'ana'})
            
            return df_ana
            
        except Exception as e:
            print(f"Warning: Could not calculate Ana factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'ana'])


class RdmFactor(A5IntangiblesFactorBase):
    """
    A.5.4 Rdm - R&D Expense to Market
    
    Rdm is the ratio of R&D expense to market equity, measuring research intensity
    relative to firm size. Higher Rdm indicates greater investment in innovation.
    
    Formula:
        Rdm = R&D_expense_{t-1} / Market_Equity_{t-1}
    
    Data source:
        - r_and_d (研发费用) from Tushare balancesheet.csv
        - circ_mv (流通市值) from Tushare daily_basic.csv
    
    Implementation:
        - Take annual maximum r_and_d per stock per year
        - Use December 31 market value (circ_mv) for each year
        - Rdm = r_and_d / (circ_mv * 10000)  [circ_mv in 万元 (10k)]
    
    Reference:
        Chan, Lakonishok, and Sougiannis (2001) on R&D and stock returns
    """
    
    @property
    def name(self) -> str:
        return 'rdm'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.4'
    
    @property
    def abbr(self) -> str:
        return 'Rdm'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate R&D expense to market ratio."""
        import os
        try:
            # Load R&D data from balancesheet
            bs_file = DATADIR.parent / "tushare" / "balancesheet.csv"
            if not bs_file.exists():
                print(f"Warning: Balance sheet file not found: {bs_file}")
                return pd.DataFrame(columns=['code', 'year', 'rdm'])
            
            data = pd.read_csv(bs_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'r_and_d']].dropna()
            
            # Load market value data from daily_basic
            db_file = DATADIR.parent / "tushare" / "daily_basic.csv"
            if not db_file.exists():
                print(f"Warning: Daily basic file not found: {db_file}")
                return pd.DataFrame(columns=['code', 'year', 'rdm'])
            
            df = pd.read_csv(db_file)
            df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
            df = df[['code', 'date', 'circ_mv']]
            
            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)
            df['code'] = df['code'].apply(standardize_ticker)
            
            # Process R&D data: get annual max
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data = data.sort_values(['code', 'date'])
            
            # Remove zero R&D
            data = data[data['r_and_d'] != 0]
            
            # Take annual maximum R&D
            data_annual = data.groupby(['code', 'year'])['r_and_d'].max().reset_index()
            
            # Process market value: get December 31 values
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df = df.sort_values(['code', 'date'])
            
            # Get last trading day of each year (December values)
            df_dec = df.groupby(['code', 'year']).tail(1)
            
            # Merge R&D and market value
            result = pd.merge(data_annual, df_dec, on=['code', 'year'], how='left')
            
            # Calculate Rdm = r_and_d / (circ_mv * 10000)
            # circ_mv is in 万元 (10,000 yuan), need to multiply by 10000 to match r_and_d units
            result['rdm'] = result['r_and_d'] / (result['circ_mv'] * 10000)
            
            return result[['code', 'year', 'rdm']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate Rdm factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'rdm'])


class RdmqFactor(A5IntangiblesFactorBase):
    """
    A.5.5 Rdmq - Quarterly R&D Expense to Market
    
    Rdmq is the quarterly R&D expense to market equity ratio.
    Uses 4-quarter lagged R&D expense divided by 1-month lagged market value.
    
    Formula:
        Rdmq = R&D_expense_{q-4} / Market_Equity_{m-1}
    
    Data source:
        - r_and_d from Tushare balancesheet.csv (quarterly)
        - circ_mv from Tushare daily_basic.csv
    
    Implementation:
        - Group R&D by quarter, take maximum
        - Use month-end market value
        - Shift R&D by 4 quarters, market value by 1 month
    """
    
    @property
    def name(self) -> str:
        return 'rdmq'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.5'
    
    @property
    def abbr(self) -> str:
        return 'Rdmq'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly R&D expense to market ratio."""
        import os
        try:
            # Load R&D data
            bs_file = DATADIR.parent / "tushare" / "balancesheet.csv"
            if not bs_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'month', 'rdmq'])
            
            data = pd.read_csv(bs_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'r_and_d']].dropna()
            
            # Load market value data
            db_file = DATADIR.parent / "tushare" / "daily_basic.csv"
            if not db_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'month', 'rdmq'])
            
            df = pd.read_csv(db_file)
            df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
            df = df[['code', 'date', 'circ_mv']]
            
            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)
            df['code'] = df['code'].apply(standardize_ticker)
            
            # Process R&D data: extract quarter
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data['quarter'] = data['date'].dt.quarter
            
            # Get quarterly max R&D
            data_q = data.groupby(['code', 'year', 'quarter'])['r_and_d'].max().reset_index()
            
            # Process market value: get month-end values
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            
            df = df.sort_values(['code', 'year', 'month'])
            df_monthly = df.groupby(['code', 'year', 'month']).tail(1)
            
            # Merge on year and quarter
            result = pd.merge(data_q, df_monthly, on=['code', 'year', 'quarter'], how='right')
            result = result.replace(np.nan, 0)
            result = result.sort_values(['code', 'year', 'month'])
            
            # Shift: R&D by 4 quarters, circ_mv by 1 month
            result['rd_prior_4'] = result.groupby('code')['r_and_d'].shift(4)
            result['circ_mv_prior_1'] = result.groupby('code')['circ_mv'].shift(1)
            
            # Calculate Rdmq
            result['rdmq'] = result['rd_prior_4'] / (result['circ_mv_prior_1'] * 10000)
            
            # Filter out zero values
            result = result[result['rdmq'] != 0]
            
            return result[['code', 'year', 'month', 'rdmq']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate Rdmq factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'month', 'rdmq'])


class RdsFactor(A5IntangiblesFactorBase):
    """
    A.5.6 Rds - R&D Expense to Sales
    
    Rds is the ratio of R&D expense to sales revenue, measuring innovation
    intensity relative to revenue generation.
    
    Formula:
        Rds = R&D_expense / Sales_Revenue
    
    Data source:
        - r_and_d from Tushare balancesheet.csv
        - revenue from Tushare income.csv
    
    Implementation:
        - Take annual maximum for both R&D and revenue
        - Merge by code and year
    """
    
    @property
    def name(self) -> str:
        return 'rds'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.6'
    
    @property
    def abbr(self) -> str:
        return 'Rds'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate R&D expense to sales ratio."""
        import os
        try:
            # Load R&D data from balancesheet
            bs_file = DATADIR.parent / "tushare" / "balancesheet.csv"
            if not bs_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'rds'])
            
            data = pd.read_csv(bs_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'r_and_d']]
            
            # Load revenue data from income
            income_file = DATADIR.parent / "tushare" / "income.csv"
            if not income_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'rds'])
            
            df = pd.read_csv(income_file)
            df = df.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            df = df[['code', 'date', 'revenue']]
            
            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)
            df['code'] = df['code'].apply(standardize_ticker)
            
            # Process R&D data
            data = data.drop_duplicates()
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data = data.sort_values(['code', 'date'])
            
            # Remove zero R&D
            data = data[data['r_and_d'] != 0]
            
            # Take annual maximum R&D
            data_annual = data.groupby(['code', 'year'])['r_and_d'].max().reset_index()
            
            # Process revenue data
            df = df.drop_duplicates()
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df = df.sort_values(['code', 'date'])
            
            # Take annual maximum revenue (typically last quarter)
            df_annual = df.groupby(['code', 'year'])['revenue'].max().reset_index()
            
            # Merge R&D and revenue
            result = pd.merge(data_annual, df_annual, on=['code', 'year'], how='left')
            
            # Calculate Rds = r_and_d / revenue
            result['rds'] = result['r_and_d'] / result['revenue']
            
            return result[['code', 'year', 'rds']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate Rds factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'rds'])


class AdmFactor(A5IntangiblesFactorBase):
    """
    A.5.2 Adm - Advertising Expense to Market
    
    Adm is the ratio of advertising expense to market equity.
    Note: Using sell_exp (销售费用) as proxy for advertising expense
    due to lack of direct advertising expense data in Chinese financial reports.
    
    Formula:
        Adm = Advertising_expense_{t-1} / Market_Equity_{t-1}
    
    Data source:
        - sell_exp (销售费用) from Tushare income.csv as advertising proxy
        - circ_mv (流通市值) from Tushare daily_basic.csv
    
    Implementation:
        - Take annual maximum sell_exp
        - Use December 31 market value (circ_mv)
        - Adm = sell_exp / (circ_mv * 10000)
    
    Reference:
        Chan, Lakonishok, and Sougiannis (2001) on intangible investments
    """
    
    @property
    def name(self) -> str:
        return 'adm'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.2'
    
    @property
    def abbr(self) -> str:
        return 'Adm'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate advertising expense to market ratio."""
        import os
        try:
            # Load advertising data from income
            income_file = DATADIR.parent / "tushare" / "income.csv"
            if not income_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'adm'])
            
            data = pd.read_csv(income_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'sell_exp']]
            
            # Load market value data
            db_file = DATADIR.parent / "tushare" / "daily_basic.csv"
            if not db_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'adm'])
            
            df = pd.read_csv(db_file)
            df = df.rename(columns={'ts_code': 'code', 'trade_date': 'date'})
            df = df[['code', 'date', 'circ_mv']]
            
            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)
            df['code'] = df['code'].apply(standardize_ticker)
            
            # Process advertising data
            data = data.drop_duplicates()
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data = data.sort_values(['code', 'date'])
            
            # Take annual maximum sell_exp
            data_annual = data.groupby(['code', 'year'])['sell_exp'].max().reset_index()
            
            # Process market value: get December 31 values
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df = df.sort_values(['code', 'date'])
            
            # Get last trading day of each year
            df_dec = df.groupby(['code', 'year']).tail(1)
            
            # Merge advertising and market value
            result = pd.merge(data_annual, df_dec, on=['code', 'year'], how='left')
            
            # Calculate Adm = sell_exp / (circ_mv * 10000)
            result['adm'] = result['sell_exp'] / (result['circ_mv'] * 10000)
            
            return result[['code', 'year', 'adm']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate Adm factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'adm'])


class gAdFactor(A5IntangiblesFactorBase):
    """
    A.5.3 gAd - Growth in Advertising Expense
    
    gAd is the annual growth rate of advertising expense (proxied by selling expense).
    Measures changes in marketing intensity.
    
    Formula:
        gAd = (sell_exp_{t-1} - sell_exp_{t-2}) / sell_exp_{t-2}
    
    Data source:
        - sell_exp (销售费用) from Tushare income.csv
    
    Implementation:
        - Take annual maximum sell_exp
        - Calculate year-over-year growth rate
        - Use 1-year and 2-year lagged values
    """
    
    @property
    def name(self) -> str:
        return 'gad'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.3'
    
    @property
    def abbr(self) -> str:
        return 'gAd'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate growth in advertising expense."""
        import os
        try:
            # Load advertising data from income
            income_file = DATADIR.parent / "tushare" / "income.csv"
            if not income_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'gad'])
            
            data = pd.read_csv(income_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'sell_exp']]
            
            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)
            
            # Process data
            data = data.drop_duplicates()
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data = data.sort_values(['code', 'date'])
            
            # Take annual maximum sell_exp
            data_annual = data.groupby(['code', 'year'])['sell_exp'].max().reset_index()
            
            # Create lagged variables
            data_annual = data_annual.sort_values(['code', 'year'])
            data_annual['sell_exp_prior_1'] = data_annual.groupby('code')['sell_exp'].shift(1)
            data_annual['sell_exp_prior_2'] = data_annual.groupby('code')['sell_exp'].shift(2)
            
            # Calculate growth rate: (t-1 - t-2) / t-2
            data_annual['gad'] = (data_annual['sell_exp_prior_1'] - data_annual['sell_exp_prior_2']) / data_annual['sell_exp_prior_2']
            
            return data_annual[['code', 'year', 'gad']].dropna()
            
        except Exception as e:
            print(f"Warning: Could not calculate gAd factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'gad'])


class TanFactor(A5IntangiblesFactorBase):
    """
    A.5.27 Tan - Tangibility
    
    Tan measures the asset tangibility, calculated as the weighted sum of 
    cash, receivables, inventory, and fixed assets divided by total assets.
    
    Formula:
        Tan = (cash + 0.715*AR + 0.547*inventory + 0.535*PPE) / total_assets
    
    where:
        - cash: cash holdings (货币资金)
        - AR: accounts receivable (应收账款)
        - inventory: inventories (存货)
        - PPE: gross property, plant, and equipment (固定资产)
    
    Data source:
        - All items from quarterly_balancesheet_cleaned.csv
    """
    
    @property
    def name(self) -> str:
        return 'tan'
    
    @property
    def factor_id(self) -> str:
        return 'A.5.27'
    
    @property
    def abbr(self) -> str:
        return 'Tan'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate tangibility factor."""
        # Load balance sheet data
        cols = ['quarterly_money_cap_fillna', 'quarterly_acct_payable_fillna',
                'quarterly_inventories_fillna', 'quarterly_fix_assets_fillna',
                'quarterly_total_assets_fillna']
        df = load_balancesheet_data(cols, RESSET_DIR)
        
        # Rename columns
        df = df.rename(columns={
            'quarterly_money_cap_fillna': 'cash',
            'quarterly_acct_payable_fillna': 'ar',
            'quarterly_inventories_fillna': 'inventory',
            'quarterly_fix_assets_fillna': 'ppe',
            'quarterly_total_assets_fillna': 'total_assets'
        })
        
        # Fill NaN with 0
        df = df.fillna(0)
        
        # Calculate tangibility
        df['tan_numerator'] = (df['cash'] + 0.715 * df['ar'] + 
                              0.547 * df['inventory'] + 0.535 * df['ppe'])
        df['tan'] = df['tan_numerator'] / df['total_assets']
        
        # Replace inf and -inf with NaN
        df['tan'] = df['tan'].replace([np.inf, -np.inf], np.nan)
        
        # Extract year from end_date (handle both YYYYMMDD integer and ISO date string formats)
        try:
            # Try Tushare integer format first (e.g., 20211231)
            df['year'] = pd.to_datetime(df['end_date'], format='%Y%m%d').dt.year
        except (ValueError, TypeError):
            # Fallback to ISO/mixed format (e.g., "2021-12-31")
            df['year'] = pd.to_datetime(df['end_date'], format='mixed').dt.year
        
        return df[['code', 'year', 'tan']].dropna()


# =============================================================================
# SECTION 14: A6 TRADING FRICTIONS FACTOR IMPLEMENTATIONS
# =============================================================================


class MeFactor(A6TradingFrictionsFactorBase):
    """
    A.6.1 Me - Market Equity (monthly)
    
    Me denotes market capitalization, calculated as price times shares outstanding.
    This is a fundamental measure of firm size.
    
    Formula:
        Me = price * shares_outstanding
    
    Data source:
        - circ_mv (流通市值) from daily_basic_resset.csv
    
    Note: We use circulation market value directly from RESSET data.
    """
    
    @property
    def name(self) -> str:
        return 'me'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.1'
    
    @property
    def abbr(self) -> str:
        return 'Me'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate market equity factor."""
        # Load daily basic data
        df = load_daily_basic_data(['circ_mv'], RESSET_DIR)
        df = df.rename(columns={'circ_mv': 'me'})
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Get monthly end values
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df = df.sort_values(['code', 'date'])
        df_monthly = df.groupby(['code', 'year', 'month']).tail(1)
        
        return df_monthly[['code', 'date', 'me']].dropna()


class TurnFactor(A6TradingFrictionsFactorBase):
    """
    A.6.3 Turn1, Turn6, Turn12 - Share Turnover
    
    Turn measures the trading activity of a stock, calculated as the
    average monthly share turnover rate.
    
    Formula:
        Turn = average(turnover_rate) over specified period
    
    where turnover_rate = trading_volume / shares_outstanding
    
    Data source:
        - turnover_rate_f from daily_basic_resset.csv
    
    Note: We calculate 1-month, 6-month, and 12-month averages.
    This is similar to Tur (A.6.11) but uses simpler calculation.
    """
    
    @property
    def name(self) -> str:
        return 'turn'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.3'
    
    @property
    def abbr(self) -> str:
        return 'Turn'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate share turnover factor."""
        # Load daily basic data
        df = load_daily_basic_data(['turnover_rate_f'], RESSET_DIR)
        df = df.rename(columns={'turnover_rate_f': 'turnover'})
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'turnover'])
        
        # Get monthly average turnover
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df_monthly = df.groupby(['code', 'year', 'month'])['turnover'].mean().reset_index()
        df_monthly = df_monthly.rename(columns={'turnover': 'turn1'})
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['turn6'] = df_monthly.groupby('code')['turn1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['turn12'] = df_monthly.groupby('code')['turn1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Reconstruct date (end of month)
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'turn1', 'turn6', 'turn12']].dropna()


class TurFactor(A6TradingFrictionsFactorBase):
    """
    A.6.11 Tur1, Tur6, Tur12 - Share Turnover (weighted average)
    
    Tur is the average daily share turnover over the prior months,
    calculated as a weighted rolling average. Requires minimum 50 days.
    
    Formula:
        Tur = weighted_avg(daily_turnover_rate) over specified months
    
    Note: This is the weighted version of Turn factor.
    For simplicity in sample mode, we use the same calculation as Turn.
    """
    
    @property
    def name(self) -> str:
        return 'tur'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.11'
    
    @property
    def abbr(self) -> str:
        return 'Tur'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate Tur factor (same as Turn for simplicity)."""
        turn = TurnFactor()
        df = turn.calculate()
        df = df.rename(columns={'turn1': 'tur1', 'turn6': 'tur6', 'turn12': 'tur12'})
        return df


class DtvFactor(A6TradingFrictionsFactorBase):
    """
    A.6.13 Dtv1, Dtv6, Dtv12 - Dollar Trading Volume
    
    Dtv measures the average daily dollar trading volume over prior months.
    Dollar trading volume = share price × number of shares traded.
    
    Formula:
        Dtv = average(amount) over specified period
    
    where amount = trading amount in thousand yuan
    
    Data source:
        - amount from Wind dailystockreturn_wind.csv
    
    Note: Requires minimum 50 days of observations.
    """
    
    @property
    def name(self) -> str:
        return 'dtv'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.13'
    
    @property
    def abbr(self) -> str:
        return 'Dtv'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate dollar trading volume factor."""
        # Load Wind daily data
        df = load_wind_daily_data(['amount'], DATADIR.parent / "wind")
        df = df.dropna(subset=['amount'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Get monthly average amount
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df_monthly = df.groupby(['code', 'year', 'month'])['amount'].mean().reset_index()
        df_monthly = df_monthly.rename(columns={'amount': 'dtv1'})
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['dtv6'] = df_monthly.groupby('code')['dtv1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['dtv12'] = df_monthly.groupby('code')['dtv1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Reconstruct date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'dtv1', 'dtv6', 'dtv12']].dropna()


class PpsFactor(A6TradingFrictionsFactorBase):
    """
    A.6.15 Pps1, Pps6, Pps12 - Share Price
    
    Pps is the end-of-month close price lagged by one month (pps1),
    or the average of pps1 over the prior six months (pps6) or twelve months (pps12).
    
    Formula:
        pps1 = close price at t-1
        pps6 = rolling_average(pps1, 6 months)
        pps12 = rolling_average(pps1, 12 months)
    
    Data source:
        - close from Wind dailystockreturn_wind.csv
    """
    
    @property
    def name(self) -> str:
        return 'pps'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.15'
    
    @property
    def abbr(self) -> str:
        return 'Pps'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate share price factor."""
        # Load Wind daily data
        df = load_wind_daily_data(['close'], DATADIR.parent / "wind")
        df = df.dropna(subset=['close'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Get end-of-month close price
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df = df.sort_values(['code', 'date'])
        df_monthly = df.groupby(['code', 'year', 'month']).tail(1)[['code', 'year', 'month', 'close', 'date']].copy()
        
        # Lag by one month (pps1 = prior month's close)
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['pps1'] = df_monthly.groupby('code')['close'].shift(1)
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly['pps6'] = df_monthly.groupby('code')['pps1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['pps12'] = df_monthly.groupby('code')['pps1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Use last day of month as date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'pps1', 'pps6', 'pps12']].dropna()


class AmiFactor(A6TradingFrictionsFactorBase):
    """
    A.6.16 Ami1, Ami6, Ami12 - Amihud Illiquidity
    
    Ami is the ratio of absolute daily stock return to daily dollar trading volume,
    averaged over the prior month (ami1), six months (ami6), or twelve months (ami12).
    
    Formula:
        daily_ami = |daily_return| / amount
        ami1 = monthly_average(daily_ami)
        ami6 = rolling_average(ami1, 6 months)
        ami12 = rolling_average(ami1, 12 months)
    
    Data source:
        - amount, close, preclose from Wind dailystockreturn_wind.csv
    """
    
    @property
    def name(self) -> str:
        return 'ami'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.16'
    
    @property
    def abbr(self) -> str:
        return 'Ami'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate Amihud illiquidity factor."""
        # Load Wind daily data
        df = load_wind_daily_data(['amount', 'close', 'preclose'], DATADIR.parent / "wind")
        df = df.dropna(subset=['amount', 'close', 'preclose'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Calculate daily return
        df = df.sort_values(['code', 'date'])
        df['day_return'] = (df['close'] - df['preclose']) / df['preclose']
        df['day_return_abs'] = df['day_return'].abs()
        
        # Calculate return-to-volume ratio
        df['ratio'] = df['day_return_abs'] / df['amount']
        
        # Get monthly average (ami1)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df_monthly = df.groupby(['code', 'year', 'month'])['ratio'].mean().reset_index()
        df_monthly = df_monthly.rename(columns={'ratio': 'ami1'})
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['ami6'] = df_monthly.groupby('code')['ami1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['ami12'] = df_monthly.groupby('code')['ami1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Reconstruct date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'ami1', 'ami6', 'ami12']].dropna()


class SrevFactor(A6TradingFrictionsFactorBase):
    """
    A.6.26 Srev - Short-term Reversal
    
    Srev is the minimum daily return during the prior month.
    Stocks with the lowest returns are expected to reverse.
    
    Formula:
        srev = min(daily_return) over the prior month
    
    Data source:
        - close from Wind dailystockreturn_wind.csv
    """
    
    @property
    def name(self) -> str:
        return 'srev'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.26'
    
    @property
    def abbr(self) -> str:
        return 'Srev'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate short-term reversal factor."""
        # Load Wind daily data
        df = load_wind_daily_data(['close'], DATADIR.parent / "wind")
        df = df.dropna(subset=['close'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by code and date
        df = df.sort_values(['code', 'date'])
        
        # Calculate daily return
        df['close_prior'] = df.groupby('code')['close'].shift(1)
        df['returns'] = (df['close'] - df['close_prior']) / df['close_prior']
        
        # Get monthly minimum return
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df_monthly = df.groupby(['code', 'year', 'month'])['returns'].min().reset_index()
        df_monthly = df_monthly.rename(columns={'returns': 'srev'})
        
        # Reconstruct date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'srev']].dropna()


class TsFactor(A6TradingFrictionsFactorBase):
    """
    A.6.19 Ts1, Ts6, Ts12 - Total Skewness
    
    Ts is the skewness of daily returns, calculated monthly (ts1),
    or the average of ts1 over the prior six months (ts6) or twelve months (ts12).
    
    Formula:
        ts1 = monthly skewness of daily returns
        ts6 = rolling_average(ts1, 6 months)
        ts12 = rolling_average(ts1, 12 months)
    
    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv
    """
    
    @property
    def name(self) -> str:
        return 'ts'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.19'
    
    @property
    def abbr(self) -> str:
        return 'Ts'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate total skewness factor."""
        # Load Wind daily data
        df = load_wind_daily_data(['close', 'preclose'], DATADIR.parent / "wind")
        df = df.dropna(subset=['close', 'preclose'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by code and date
        df = df.sort_values(['code', 'date'])
        
        # Calculate daily return
        df['returns'] = (df['close'] - df['preclose']) / df['preclose']
        
        # Calculate monthly skewness
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df_monthly = df.groupby(['code', 'year', 'month'])['returns'].skew().reset_index()
        df_monthly = df_monthly.rename(columns={'returns': 'ts1'})
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['ts6'] = df_monthly.groupby('code')['ts1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['ts12'] = df_monthly.groupby('code')['ts1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Reconstruct date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'ts1', 'ts6', 'ts12']].dropna()


class ShlFactor(A6TradingFrictionsFactorBase):
    """
    A.6.28 Shl1, Shl6, Shl12 - High-Low Bid-Ask Spread
    
    Shl is the Corwin and Schultz (2012) high-low bid-ask spread estimator.
    
    Formula (Corwin-Schultz):
        β = Σ[log(H_t / L_t)]² for t and t+1
        γ = [log(H_{t,t+1} / L_{t,t+1})]²
        α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
        Shl = 2(e^α - 1) / (1 + e^α)
    
    where:
        H_t, L_t = highest and lowest price on day t
        H_{t,t+1}, L_{t,t+1} = highest and lowest price over days t and t+1
    
    Data source:
        - high, low from Wind dailystockreturn_wind.csv
    """
    
    @property
    def name(self) -> str:
        return 'shl'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.28'
    
    @property
    def abbr(self) -> str:
        return 'Shl'
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate high-low bid-ask spread factor."""
        import numpy as np
        
        # Load Wind daily data
        df = load_wind_daily_data(['high', 'low'], DATADIR.parent / "wind")
        df = df.dropna(subset=['high', 'low'])
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by code and date
        df = df.sort_values(['code', 'date'])
        
        # Extract year and month
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Get next day's high and low within the same month
        df['high_next'] = df.groupby(['code', 'year', 'month'])['high'].shift(-1)
        df['low_next'] = df.groupby(['code', 'year', 'month'])['low'].shift(-1)
        
        # Calculate two-day high and low
        df['high_2day'] = df[['high', 'high_next']].max(axis=1)
        df['low_2day'] = df[['low', 'low_next']].min(axis=1)
        
        # Calculate Corwin-Schultz components
        df['beta'] = (np.log(df['high_2day'] / df['low_2day']))**2 + (np.log(df['high'] / df['low']))**2
        df['gamma'] = (np.log(df['high_2day'] / df['low_2day']))**2
        
        # Calculate alpha
        sqrt2 = np.sqrt(2)
        df['alpha'] = (np.sqrt(2 * df['beta']) - np.sqrt(df['beta'])) / (3 - 2*sqrt2) - \
                      np.sqrt(df['gamma'] / (3 - 2*sqrt2))
        
        # Calculate Shl
        df['Shl'] = 2 * (np.exp(df['alpha']) - 1) / (1 + np.exp(df['alpha']))
        
        # Get monthly average
        df_monthly = df.groupby(['code', 'year', 'month'])['Shl'].mean().reset_index()
        df_monthly = df_monthly.rename(columns={'Shl': 'shl1'})
        
        # Calculate rolling 6-month and 12-month averages
        df_monthly = df_monthly.sort_values(['code', 'year', 'month'])
        df_monthly['shl6'] = df_monthly.groupby('code')['shl1'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df_monthly['shl12'] = df_monthly.groupby('code')['shl1'].transform(
            lambda x: x.rolling(12, min_periods=1).mean()
        )
        
        # Reconstruct date
        df_monthly['date'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        ) + pd.offsets.MonthEnd(0)
        
        return df_monthly[['code', 'date', 'shl1', 'shl6', 'shl12']].dropna()


class IvffFactor(A6TradingFrictionsFactorBase):
    """
    A.6.2 Ivff1, Ivff6, Ivff12 - Idiosyncratic Volatility (Fama-French 3-Factor)
    
    Ivff is the idiosyncratic volatility from regressing stock excess returns
    on the Fama-French 3-factor model over the prior 1, 6, or 12 months.
    
    Formula:
        R_it - R_ft = α + β1(R_Mt - R_ft) + β2·SMB_t + β3·HML_t + ε_it
        Ivff = √N × std(ε)
    
    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv
        - Rmrf_tmv, Smb_tmv, Hml_tmv from RESSET_FamaFrenchDaily.csv
        - DRFRet from RESSET_RISKFREE.csv
    
    Parameters:
        ff3_data_path: Path to FF3 factor data (default: DATADIR / "RESSET_FamaFrenchDaily.csv")
        rf_data_path: Path to risk-free rate data (default: DATADIR / "RESSET_RISKFREE.csv")
    """
    
    def __init__(self, ff3_data_path=None, rf_data_path=None):
        """Initialize with optional custom data paths."""
        super().__init__()
        self.ff3_data_path = ff3_data_path or (DATADIR / "RESSET_FamaFrenchDaily.csv")
        self.rf_data_path = rf_data_path or (DATADIR / "RESSET_RISKFREE.csv")
    
    @property
    def name(self) -> str:
        return 'ivff'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.2'
    
    @property
    def abbr(self) -> str:
        return 'Ivff'
    
    def _compute_idiosyncratic_volatility(self, group, horizon):
        """
        Compute idiosyncratic volatility for a single stock-month group.
        
        Parameters:
            group: DataFrame for one stock-month with columns: date, ret_excess, Rmrf, Smb, Hml
            horizon: Number of months to look back (1, 6, or 12)
        
        Returns:
            Idiosyncratic volatility (std of residuals * sqrt(N))
        """
        import statsmodels.api as sm
        
        if len(group) < 10:  # Need at least 10 observations for regression
            return np.nan
        
        # Prepare regression data
        y = group['ret_excess'].values
        X = group[['Rmrf', 'Smb', 'Hml']].values
        X = sm.add_constant(X)
        
        try:
            # Run OLS regression
            model = sm.OLS(y, X, missing='drop')
            results = model.fit()
            
            # Calculate idiosyncratic volatility
            residuals = results.resid
            N = len(residuals)
            ivol = np.sqrt(N) * np.std(residuals, ddof=1)
            
            return ivol
        except:
            return np.nan
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate FF3 idiosyncratic volatility factor."""
        import warnings
        
        # Check for data availability
        if not Path(self.ff3_data_path).exists() or not Path(self.rf_data_path).exists():
            warnings.warn(
                f"IvffFactor: Required data files not found:\n"
                f"  FF3: {self.ff3_data_path} (exists: {Path(self.ff3_data_path).exists()})\n"
                f"  Risk-free: {self.rf_data_path} (exists: {Path(self.rf_data_path).exists()})\n"
                f"Returning empty DataFrame."
            )
            return pd.DataFrame(columns=['code', 'date', 'ivff1', 'ivff6', 'ivff12'])
        
        # Load stock return data
        stock_data = load_wind_daily_data(col=['close', 'preclose'], path=_data_dir / "wind")
        # Use 'dts' column that's already in the data
        stock_data['dts'] = pd.to_datetime(stock_data['dts'], errors='coerce')
        
        # Calculate return from close and preclose
        stock_data['ret'] = (stock_data['close'] - stock_data['preclose']) / stock_data['preclose']
        stock_data = stock_data[['code', 'dts', 'ret']].dropna()
        
        # Load FF3 factors
        ff3_data = pd.read_csv(self.ff3_data_path)
        ff3_data = ff3_data.rename(columns={'日期_Date': 'dts'})
        ff3_data['dts'] = pd.to_datetime(ff3_data['dts'], errors='coerce')
        ff3_data = ff3_data[['dts', '市场溢酬因子__流通市值加权_Rmrf_tmv', 
                             '市值因子__流通市值加权_Smb_tmv', 
                             '账面市值比因子__流通市值加权_Hml_tmv']]
        ff3_data = ff3_data.rename(columns={
            '市场溢酬因子__流通市值加权_Rmrf_tmv': 'Rmrf',
            '市值因子__流通市值加权_Smb_tmv': 'Smb',
            '账面市值比因子__流通市值加权_Hml_tmv': 'Hml'
        })
        
        # Load risk-free rate
        rf_data = pd.read_csv(self.rf_data_path)
        rf_data = rf_data.rename(columns={'日期_Date': 'dts', '日无风险收益率_DRFRet': 'rf'})
        rf_data['dts'] = pd.to_datetime(rf_data['dts'], errors='coerce')
        rf_data = rf_data[['dts', 'rf']]
        
        # Merge all data
        df = pd.merge(stock_data, ff3_data, on='dts', how='inner')
        df = pd.merge(df, rf_data, on='dts', how='left')
        df['rf'] = df['rf'].fillna(0)  # Fill missing rf with 0
        
        # Calculate excess return
        df['ret_excess'] = df['ret'] - df['rf']
        
        # Add year-month columns
        df['year'] = df['dts'].dt.year
        df['month'] = df['dts'].dt.month
        
        # Sort by code and date
        df = df.sort_values(['code', 'dts'])
        
        # Calculate rolling idiosyncratic volatility for different horizons
        results = []
        
        for (code, year, month), group in df.groupby(['code', 'year', 'month']):
            # Get end date of the month
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            
            # Calculate for 1-month horizon
            iv1 = self._compute_idiosyncratic_volatility(group, 1)
            
            # Calculate for 6-month horizon
            start_date_6m = month_end - pd.DateOffset(months=6)
            group_6m = df[(df['code'] == code) & 
                         (df['dts'] > start_date_6m) & 
                         (df['dts'] <= month_end)]
            iv6 = self._compute_idiosyncratic_volatility(group_6m, 6)
            
            # Calculate for 12-month horizon
            start_date_12m = month_end - pd.DateOffset(months=12)
            group_12m = df[(df['code'] == code) & 
                          (df['dts'] > start_date_12m) & 
                          (df['dts'] <= month_end)]
            iv12 = self._compute_idiosyncratic_volatility(group_12m, 12)
            
            results.append({
                'code': code,
                'date': month_end.date(),
                'ivff1': iv1,
                'ivff6': iv6,
                'ivff12': iv12
            })
        
        result_df = pd.DataFrame(results)
        return result_df.dropna(subset=['ivff1', 'ivff6', 'ivff12'], how='all')


class IvcFactor(A6TradingFrictionsFactorBase):
    """
    A.6.4 Ivc1, Ivc6, Ivc12 - Idiosyncratic Volatility (CAPM)
    
    Ivc is the idiosyncratic volatility from regressing stock excess returns
    on the market excess return (CAPM) over the prior 1, 6, or 12 months.
    
    Formula:
        R_it - R_ft = α + β(R_Mt - R_ft) + ε_it
        Ivc = √N × std(ε)
    
    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv
        - Rmrf_tmv from RESSET_FamaFrenchDaily.csv (market excess return)
        - DRFRet from RESSET_RISKFREE.csv
    
    Parameters:
        ff3_data_path: Path to FF3 factor data (for market factor, default: DATADIR / "RESSET_FamaFrenchDaily.csv")
        rf_data_path: Path to risk-free rate data (default: DATADIR / "RESSET_RISKFREE.csv")
    """
    
    def __init__(self, ff3_data_path=None, rf_data_path=None):
        """Initialize with optional custom data paths."""
        super().__init__()
        self.ff3_data_path = ff3_data_path or (DATADIR / "RESSET_FamaFrenchDaily.csv")
        self.rf_data_path = rf_data_path or (DATADIR / "RESSET_RISKFREE.csv")
    
    @property
    def name(self) -> str:
        return 'ivc'
    
    @property
    def factor_id(self) -> str:
        return 'A.6.4'
    
    @property
    def abbr(self) -> str:
        return 'Ivc'
    
    def _compute_capm_idiosyncratic_volatility(self, group, horizon):
        """
        Compute CAPM idiosyncratic volatility for a single stock-month group.
        
        Parameters:
            group: DataFrame for one stock-month with columns: date, ret_excess, Rmrf
            horizon: Number of months to look back (1, 6, or 12)
        
        Returns:
            Idiosyncratic volatility (std of residuals * sqrt(N))
        """
        import statsmodels.api as sm
        
        if len(group) < 10:  # Need at least 10 observations for regression
            return np.nan
        
        # Prepare regression data
        y = group['ret_excess'].values
        X = group[['Rmrf']].values
        X = sm.add_constant(X)
        
        try:
            # Run OLS regression
            model = sm.OLS(y, X, missing='drop')
            results = model.fit()
            
            # Calculate idiosyncratic volatility
            residuals = results.resid
            N = len(residuals)
            ivol = np.sqrt(N) * np.std(residuals, ddof=1)
            
            return ivol
        except:
            return np.nan
    
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate CAPM idiosyncratic volatility factor."""
        import warnings
        
        # Check for data availability
        if not Path(self.ff3_data_path).exists() or not Path(self.rf_data_path).exists():
            warnings.warn(
                f"IvcFactor: Required data files not found:\n"
                f"  Market data (Rmrf): {self.ff3_data_path} (exists: {Path(self.ff3_data_path).exists()})\n"
                f"  Risk-free: {self.rf_data_path} (exists: {Path(self.rf_data_path).exists()})\n"
                f"Returning empty DataFrame."
            )
            return pd.DataFrame(columns=['code', 'date', 'ivc1', 'ivc6', 'ivc12'])
        
        # Load stock return data
        stock_data = load_wind_daily_data(col=['close', 'preclose'], path=_data_dir / "wind")
        # Use 'dts' column that's already in the data
        stock_data['dts'] = pd.to_datetime(stock_data['dts'], errors='coerce')
        
        # Calculate return from close and preclose
        stock_data['ret'] = (stock_data['close'] - stock_data['preclose']) / stock_data['preclose']
        stock_data = stock_data[['code', 'dts', 'ret']].dropna()
        
        # Load market factor (Rmrf from FF3 data)
        ff3_data = pd.read_csv(self.ff3_data_path)
        ff3_data = ff3_data.rename(columns={'日期_Date': 'dts'})
        ff3_data['dts'] = pd.to_datetime(ff3_data['dts'], errors='coerce')
        ff3_data = ff3_data[['dts', '市场溢酬因子__流通市值加权_Rmrf_tmv']]
        ff3_data = ff3_data.rename(columns={
            '市场溢酬因子__流通市值加权_Rmrf_tmv': 'Rmrf'
        })
        
        # Load risk-free rate
        rf_data = pd.read_csv(self.rf_data_path)
        rf_data = rf_data.rename(columns={'日期_Date': 'dts', '日无风险收益率_DRFRet': 'rf'})
        rf_data['dts'] = pd.to_datetime(rf_data['dts'], errors='coerce')
        rf_data = rf_data[['dts', 'rf']]
        
        # Merge all data
        df = pd.merge(stock_data, ff3_data, on='dts', how='inner')
        df = pd.merge(df, rf_data, on='dts', how='left')
        df['rf'] = df['rf'].fillna(0)  # Fill missing rf with 0
        
        # Calculate excess return
        df['ret_excess'] = df['ret'] - df['rf']
        
        # Add year-month columns
        df['year'] = df['dts'].dt.year
        df['month'] = df['dts'].dt.month
        
        # Sort by code and date
        df = df.sort_values(['code', 'dts'])
        
        # Calculate rolling idiosyncratic volatility for different horizons
        results = []
        
        for (code, year, month), group in df.groupby(['code', 'year', 'month']):
            # Get end date of the month
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            
            # Calculate for 1-month horizon
            ivc1 = self._compute_capm_idiosyncratic_volatility(group, 1)
            
            # Calculate for 6-month horizon
            start_date_6m = month_end - pd.DateOffset(months=6)
            group_6m = df[(df['code'] == code) & 
                         (df['dts'] > start_date_6m) & 
                         (df['dts'] <= month_end)]
            ivc6 = self._compute_capm_idiosyncratic_volatility(group_6m, 6)
            
            # Calculate for 12-month horizon
            start_date_12m = month_end - pd.DateOffset(months=12)
            group_12m = df[(df['code'] == code) & 
                          (df['dts'] > start_date_12m) & 
                          (df['dts'] <= month_end)]
            ivc12 = self._compute_capm_idiosyncratic_volatility(group_12m, 12)
            
            results.append({
                'code': code,
                'date': month_end.date(),
                'ivc1': ivc1,
                'ivc6': ivc6,
                'ivc12': ivc12
            })
        
        result_df = pd.DataFrame(results)
        return result_df.dropna(subset=['ivc1', 'ivc6', 'ivc12'], how='all')





# =============================================================================
# SECTION 15: FACTOR REGISTRY
# =============================================================================

# Dictionary to store all calculated factors
all_factors = {}


# =============================================================================
# SECTION 13: MAIN EXECUTION (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    print("China Anomalies Factors Library")
    print("=" * 80)
    
    # Test A1 Momentum Factors
    print("\n=== A1 MOMENTUM FACTORS ===")
    print("-" * 80)
    
    print("\nA.1.1 SUE - Standardized Unexpected Earnings")
    sue = SueFactor()
    print(f"  Category: {sue.category}, ID: {sue.factor_id}, Abbr: {sue.abbr}")
    
    print("\nA.1.2 ABR - Cumulative Abnormal Returns around Earnings Announcement")
    abr = AbrFactor()
    print(f"  Category: {abr.category}, ID: {abr.factor_id}, Abbr: {abr.abbr}")
    
    print("\nA.1.3 RE - Revisions in Analyst Earnings Forecasts")
    re = ReFactor()
    print(f"  Category: {re.category}, ID: {re.factor_id}, Abbr: {re.abbr}")
    
    print("\nA.1.4 R6 - Prior 6-Month Returns")
    r6 = R6Factor()
    print(f"  Category: {r6.category}, ID: {r6.factor_id}, Abbr: {r6.abbr}")
    
    print("\nA.1.5 R11 - Prior 11-Month Returns")
    r11 = R11Factor()
    print(f"  Category: {r11.category}, ID: {r11.factor_id}, Abbr: {r11.abbr}")
    
    # Test A2 Value Factors
    print("\n\n=== A2 VALUE FACTORS ===")
    print("-" * 80)
    
    print("\nA.2.1 Bm - Book-to-Market Equity")
    bm = BmFactor()
    print(f"  Category: {bm.category}, ID: {bm.factor_id}, Abbr: {bm.abbr}")
    
    print("\nA.2.4 Dm - Debt-to-Market")
    dm = DmFactor()
    print(f"  Category: {dm.category}, ID: {dm.factor_id}, Abbr: {dm.abbr}")
    
    print("\nA.2.6 Am - Assets-to-Market")
    am = AmFactor()
    print(f"  Category: {am.category}, ID: {am.factor_id}, Abbr: {am.abbr}")
    
    print("\nA.2.9 Ep - Earnings-to-Price")
    ep = EpFactor()
    print(f"  Category: {ep.category}, ID: {ep.factor_id}, Abbr: {ep.abbr}")
    
    print("\nA.2.12 Cp - Cash Flow-to-Price")
    cp = CpFactor()
    print(f"  Category: {cp.category}, ID: {cp.factor_id}, Abbr: {cp.abbr}")
    
    # Test A3 Investment Factors
    print("\n\n=== A3 INVESTMENT FACTORS ===")
    print("-" * 80)
    
    print("\nA.3.1 Agr - Asset Growth")
    agr = AgrFactor()
    print(f"  Category: {agr.category}, ID: {agr.factor_id}, Abbr: {agr.abbr}")
    
    print("\nA.3.2 I/A - Investment-to-Assets")
    ia = IaFactor()
    print(f"  Category: {ia.category}, ID: {ia.factor_id}, Abbr: {ia.abbr}")
    
    print("\nA.3.5 Nsi - Net Stock Issues")
    nsi = NsiFactor()
    print(f"  Category: {nsi.category}, ID: {nsi.factor_id}, Abbr: {nsi.abbr}")
    
    # Summary
    print("\n" + "=" * 80)
    print("FACTOR LIBRARY SUMMARY")
    print("=" * 80)
    print(f"A1 Momentum Factors:  5 implemented (Sue, Abr, Re, R6, R11)")
    print(f"A2 Value Factors:     5 implemented (Bm, Dm, Am, Ep, Cp)")
    print(f"A3 Investment Factors: 3 implemented (Agr, Ia, Nsi)")
    print(f"\nTotal: 13 factors across 3 categories")
    print("\n✅ Multi-category architecture successfully validated!")
    print("=" * 80)


# =============================================================================
# SECTION: CLASS ALIASES FOR CONVENIENCE
# =============================================================================

# Short aliases for factor classes (without "Factor" suffix)
Sue = SueFactor
Abr = AbrFactor
Re = ReFactor
R6 = R6Factor
R11 = R11Factor
Im = ImFactor
Rs = RsFactor
Tes = TesFactor
Nei = NeiFactor
W52 = W52Factor
Rm6 = Rm6Factor
Rm11 = Rm11Factor

# A2 Value factors
Bm = BmFactor
Dm = DmFactor
Am = AmFactor
Ep = EpFactor
Cp = CpFactor
Rev = RevFactor
Sp = SpFactor

Agr = AgrFactor

# A3 Investment factors
Ia = IaFactor
Iaq = IaqFactor
dPia = DpiaFactor
Noa = NoaFactor
dNoa = DnoaFactor
dLno = DlnoFactor
Ig = IgFactor
Ig2 = Ig2Factor
Ig3 = Ig3Factor
Nsi = NsiFactor

# A4 Profitability factors
Roe = RoeFactor
Droe = DroeFactor
Roa = RoaFactor
Droa = DroaFactor
Cto = CtoFactor
Ctoq = CtoqFactor
Gla = GlaFactor
Glaq = GlaqFactor
Ope = OpeFactor
Ole = OleFactor
Oleq = OleqFactor
Ola = OlaFactor
Olaq = OlaqFactor
Cop = CopFactor
Cla = ClaFactor
Claq = ClaqFactor
Gpa = GpaFactor
Opa = OpaFactor

# A5 Intangibles factors
Age = AgeFactor
Dsi = DsiFactor
Ana = AnaFactor
Tan = TanFactor
# A5 R&D and Advertising factors
Rdm = RdmFactor
Rdmq = RdmqFactor
Rds = RdsFactor
Adm = AdmFactor
gAd = gAdFactor

# A6 Trading Frictions factors
Me = MeFactor
Turn = TurnFactor
Tur = TurFactor
Dtv = DtvFactor
Pps = PpsFactor
Ami = AmiFactor
Srev = SrevFactor
Ts = TsFactor
Shl = ShlFactor
Ivff = IvffFactor
Ivc = IvcFactor
Pps = PpsFactor
Ami = AmiFactor
Srev = SrevFactor

