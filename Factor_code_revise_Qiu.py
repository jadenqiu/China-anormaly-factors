"""
A股异象因子修正合集
修正A.1.3 Re因子、A.1.11 Rm6因子、A.1.11 W52因子

功能：整合修正因子类，修复原始实现问题
作者：Jundong Qiu
日期：2026-01-25
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入原始因子类和工具函数
from china_anomalies_factors import (
    ReFactor, Rm6Factor, W52Factor, CpFactor, Rm11Factor, DlnoFactor, RoaFactor, CtoFactor, OpeFactor,
    OleFactor, OleqFactor, OlaFactor, OlaqFactor, CopFactor, ClaFactor, ClaqFactor, AdmFactor, gAdFactor,
    RdmFactor, RdsFactor, AnaFactor,
    ReportRc_resset_Path, Wind_DailyStock_Path, Resset_DailyBasic_Path, RESSET_DIR, DATADIR,
    standardize_ticker, date_transfer, load_cashflow_data, get_monthly_mv, load_balancesheet_data,
    load_income_data
)


class FixedReFactor(ReFactor):
    """修正A.1.3 Re因子：修复分子数据应使用预期EPS问题"""

    def calculate(self, **kwargs):
        """计算Re因子 - 修正数据合并逻辑"""
        # 读取分析师预测数据
        df = pd.read_csv(ReportRc_resset_Path, dtype={'code': str})
        data = pd.read_csv(Wind_DailyStock_Path, dtype={'code': str})

        # 处理月度收盘价数据
        data = data[['code', 'date', 'close']]
        data = self._month_compute(data)
        data = data.sort_values(['code', 'date'])
        data = data.groupby(by=['code', 'year', 'month']).tail(1)
        data['code'] = data['code'].apply(standardize_ticker)

        # 处理预测数据并计算EPS变化
        df['code'] = df['code'].apply(standardize_ticker)
        df.rename(columns={'report_date': 'date'}, inplace=True)
        df = df.sort_values(['code', 'date'])
        df = df[['code', 'date', 'foryear', 'eps']].dropna(subset=['date'])
        df = self._month_compute(df)

        df = df.groupby(by=['code', 'year', 'month', 'quarter', 'foryear'])['eps'].mean().reset_index()
        df['eps_prior1'] = df.groupby(by=['code', 'year', 'month'])['eps'].shift(1)
        df['eps_delta'] = df['eps'] - df['eps_prior1']
        df.rename(columns={'quarter': 'f_quarter'}, inplace=True)

        # 合并数据并计算Re因子
        merged = pd.merge(df, data, on=['code', 'year', 'month'], how='inner')
        merged['close_prior1'] = merged.groupby(by=['code', 'f_quarter']).close.shift(1)
        merged['re'] = merged['eps_delta'] / merged['close_prior1']

        # 计算6期滚动和
        merged['re_i'] = merged['re'].replace(np.nan, 0)
        merged['re'] = merged.groupby(by=['code', 'f_quarter'])['re_i'].transform(
            lambda x: x.rolling(6, center=False).sum())

        merged['date'] = pd.to_datetime(
            merged['year'].astype(str) + '-' + merged['month'].astype(str), format='%Y-%m')

        return merged[['code', 'date', 're']].dropna()

    def _month_compute(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """日期处理工具函数"""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            series = df[date_col].astype(str)
            sample = series.iloc[0]
            fmt = '%Y-%m-%d' if '-' in sample else ('%Y%m' if len(sample) == 6 else '%Y%m%d')
            df[date_col] = pd.to_datetime(series, format=fmt, errors='coerce')

        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype('Int64')
        return df


class FixedW52Factor(W52Factor):
    """修正A.1.11 W52因子：修复52周高价计算逻辑"""

    def calculate(self, sample_ratio=0.01, **kwargs) -> pd.DataFrame:
        """计算W52因子 - 修正最高价滚动计算"""
        # 读取日度基本面数据
        df = pd.read_csv(Resset_DailyBasic_Path, dtype={'code': str})
        df = df[['code', 'date', 'total_mv', 'total_share']]

        # 计算每股价格
        df['pps'] = df['total_mv'] / df['total_share']
        df = date_transfer(df, 'date', format=None)
        df = df.sort_values(['code', 'date'])

        # 计算12个月滚动最高价（252个交易日）
        df['pps_max_12m_daily'] = (
            df.groupby('code')['pps']
            .rolling(window=252, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

        # 获取月度数据并计算W52比率
        monthly_data = df.groupby(['code', 'year', 'month']).tail(1).copy()
        monthly_data['pps_max_12m'] = monthly_data['pps_max_12m_daily']
        monthly_data['w52'] = monthly_data['pps'] / monthly_data['pps_max_12m']
        monthly_data['w52'] = monthly_data['w52'].clip(0, 1)  # 处理异常值

        return monthly_data[['code', 'year', 'month', 'w52']]

class FixedRm6Factor(Rm6Factor):
    """修正A.1.12 Rm6因子：增加残差因子标准化步骤"""

    def calculate(self, resid_days: int = 5, **kwargs) -> pd.DataFrame:
        """计算Rm6因子 - 修正滚动回归逻辑"""
        df = self._cal_residual_means_stds(resid_days=resid_days)
        df = df[['code', 'date', 'res_means', 'res_stds', 'res_stand_means']].sort_values(['code', 'date'])

        # 滞后2个月处理
        df = df.set_index(['code', 'date']).groupby(level=0)[
            ['res_means', 'res_stds', 'res_stand_means']].shift(2).reset_index()
        df = df.rename(columns={'res_stand_means': 'rm6'})

        return df[['code', 'date', 'rm6']]

    def _cal_residual_means_stds(self, resid_days: int = 5) -> pd.DataFrame:
        """计算FF3残差的均值和标准差"""
        # 获取月度收益率数据
        df = self._get_monthly_return()

        # 加载并处理FF3因子数据
        ff3 = pd.read_csv("data/resset/FF3_cleaned.csv")
        ff3['date'] = pd.to_datetime(ff3['date'], errors='coerce')
        ff3 = ff3[ff3.mktflg.isin(['A'])]
        ff3 = date_transfer(ff3, 'date', format="%Y%m%d")

        # 月度聚合（年化处理）
        ff3 = ff3.groupby(by=['year', 'month'])[['rmrftmv', 'smbtmv', 'hmltmv']].mean().reset_index()
        for col in ['rmrftmv', 'smbtmv', 'hmltmv']:
            ff3[col] = ff3[col].apply(lambda x: float(x) * 22)

        # 合并数据并执行滚动回归
        df = pd.merge(df, ff3, on=['year', 'month'], how='left')
        df.dropna(subset=['rmrftmv', 'smbtmv', 'hmltmv', 'monthly_return'], inplace=True)
        df.sort_values(by=['code', 'date'], inplace=True)

        return (df.groupby(by=['code'], group_keys=True)
                .apply(lambda _df: self._get_rolling_residual_return(
            _df, 'monthly_return', ['rmrftmv', 'smbtmv', 'hmltmv'], 36, resid_days))
                .reset_index(drop=True))

    def _get_monthly_return(self) -> pd.DataFrame:
        """获取月度收益率数据"""
        df = pd.read_csv("data/wind/dailystockreturn_wind.csv", dtype={'code': str})
        df = df[['code', 'date', 'close']]
        df = date_transfer(df, 'date', format="%Y%m%d")
        df.sort_values(by=['code', 'date'], inplace=True)
        df = df.groupby(by=['code', 'year', 'month']).tail(1)
        df['monthly_return'] = df.groupby(by=['code'])['close'].pct_change()
        return df[['date', 'code', 'monthly_return', 'year', 'month']]

    def _get_rolling_residual_return(self, df: pd.DataFrame, y_col: str, x_cols: list,
                                     window_size: int, resid_days: int) -> pd.DataFrame:
        """执行滚动FF3回归"""
        res_means, res_stds = [], []

        for i in range(len(df)):
            if i < window_size - 1:
                res_means.append(np.nan)
                res_stds.append(np.nan)
                continue

            window_data = df.iloc[i - window_size + 1:i + 1]
            res, std = self._perform_regression(window_data, x_cols, y_col, resid_days)
            res_means.append(res)
            res_stds.append(std)

        df['res_means'] = res_means
        df['res_stds'] = res_stds
        df['res_stand_means'] = df['res_means'] / df['res_stds']
        return df

    def _perform_regression(self, data: pd.DataFrame, x_cols: list, y_col: str, resid_days: int) -> tuple:
        """执行OLS回归并返回残差统计量"""
        if len(data) < 36:
            return np.nan, np.nan

        Y = data[y_col]
        X = sm.add_constant(data[x_cols])

        try:
            model = sm.OLS(Y, X, missing='drop').fit()
            residuals = model.resid
            std = residuals.std()
            return np.mean(residuals[-resid_days:]), std
        except:
            return np.nan, np.nan

class FixedRm11Factor(Rm11Factor):
    """修正A.1.13 Rm11因子：增加残差因子标准化步骤"""

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
        df = df[['code', 'date', 'res_means', 'res_stds', 'res_stand_means']].sort_values(['code', 'date'])

        # Shift by 2 months
        df = df.set_index(['code', 'date']).groupby(level=0)[
            ['res_means', 'res_stds', 'res_stand_means']].shift(2).reset_index()
        df = df.rename(columns={'res_stand_means': 'rm11'})

        return df[['code', 'date', 'rm11']]


class FixedCpFactor(CpFactor):
    """修正A.2.12 Cp因子：排除现金流非正样本"""
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

        # Drop rows with non-positive cash flow
        df = df[df['quarterly_n_cashflow_act_fillna'] > 0]

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

class FixedDlnoFactor(DlnoFactor):
    """
    修正的dLno因子计算

    修正点：
    1. 修正ΔLO的符号错误（应该是减号）
    2. 改进数据处理逻辑
    """
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
        #df = df.fillna(0)

        # Calculate changes (PPENT = Fixed Assets, no provision data available)
        df['dPPENT'] = df['quarterly_fix_assets_fillna'] - df.groupby('code')['quarterly_fix_assets_fillna'].shift(1)
        df['dintan_assets'] = df['quarterly_intan_assets_fillna'] - df.groupby('code')[
            'quarterly_intan_assets_fillna'].shift(1)
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
                             df['dPPENT'] + df['dintan_assets'] + df['doth_nca'] - df['doth_ncl'] + df['DP']
                     ) / df['avg_total_assets']

        return df[['code', 'end_date', 'dLno']].dropna()


class FixedRoaFactor(RoaFactor):
    """修正Roa因子计算，分子替换为净利润并增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate return on assets factor."""
        # Load profit data
        income_cols = ['quarterly_n_income_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)
        df_income = df_income.rename(columns={'quarterly_n_income_fillna': 'n_income'})

        # Load total assets data
        assets_cols = ['quarterly_total_assets_fillna']
        df_assets = load_balancesheet_data(assets_cols, RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'n_income']],
            df_assets[['code', 'end_date', 'total_assets']],
            on=['code', 'end_date'],
            how='left'
        )

        # Filter out zero total assets
        df = df[df['total_assets'] > 0]

        # Calculate Roa = profit / lagged total assets
        df = df.sort_values(['code', 'end_date'])
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)
        df[self.abbr] = df['n_income'] / df['total_assets_lag1']

        return df[['code', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Roa factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Roa'].notna().sum()}")
        print(f"Statistics:\n{df['Roa'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedCtoFactor(CtoFactor):
    """修正Cto因子计算，增加total_assets滞后一期处理、数据>0过滤，增加validate函数"""
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

        df = df[df['total_assets'] > 0]
        df = df[df['revenue'] > 0]
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Calculate Cto = revenue / total_assets
        df['Cto'] = df['revenue'] / df['total_assets_lag1']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Cto factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Cto'].notna().sum()}")
        print(f"Statistics:\n{df['Cto'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedOpeFactor(OpeFactor):
    """修正Ope因子计算，修正分子计算补全减int_exp、数据>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating profitability to equity factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_int_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_int_exp_fillna': 'int_exp'
        })

        # Load balance sheet data for book equity calculation
        balance_cols = ['quarterly_total_hldr_eqy_inc_min_int_fillna']
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={'quarterly_total_hldr_eqy_inc_min_int_fillna': 'book_equity'})

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'int_exp']],
            df_balance[['code', 'end_date', 'book_equity']],
            on=['code', 'end_date'],
            how='inner'
        )

        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        df = df[df['book_equity'] > 0]
        df = df[df['revenue'] > 0]

        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'int_exp']] = df[['sell_exp', 'admin_exp', 'int_exp']].fillna(0)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # Calculate Ope = (revenue - COGS - SG&A) / book_equity
        df['Ope'] = (df['revenue'] - df['cogs'] - df['sga'] - df['int_exp']) / df['book_equity']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Ope factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ope'].notna().sum()}")
        print(f"Statistics:\n{df['Ope'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedOleFactor(OleFactor):
    """修正Ole因子计算，修正分子计算补全减int_exp、数据>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate operating profitability to equity factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_int_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_int_exp_fillna': 'int_exp'
        })

        # Load balance sheet data for book equity calculation
        balance_cols = ['quarterly_total_hldr_eqy_inc_min_int_fillna']
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={'quarterly_total_hldr_eqy_inc_min_int_fillna': 'book_equity'})

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'int_exp']],
            df_balance[['code', 'end_date', 'book_equity']],
            on=['code', 'end_date'],
            how='inner'
        )

        # Filter for fiscal year end (December 31)
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        df = df[df['book_equity'] > 0]
        df = df[df['revenue'] > 0]

        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])

        # Calculate lagged book equity (1 year lag)
        df['book_equity_lag1'] = df.groupby('code')['book_equity'].shift(1)

        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'int_exp']] = df[['sell_exp', 'admin_exp', 'int_exp']].fillna(0)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # Calculate Ole = (revenue - COGS - SG&A - Int) / book_equity
        df['Ole'] = (df['revenue'] - df['cogs'] - df['sga'] - df['int_exp']) / df['book_equity_lag1']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Ole factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ole'].notna().sum()}")
        print(f"Statistics:\n{df['Ole'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedOleqFactor(OleqFactor):
    """修正Oleq因子计算，修正分子计算补全减int_exp、数据>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate quarterly operating leverage (earnings) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_int_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_int_exp_fillna': 'int_exp'
        })

        # Load balance sheet data for book equity calculation
        balance_cols = ['quarterly_total_hldr_eqy_inc_min_int_fillna']
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={'quarterly_total_hldr_eqy_inc_min_int_fillna': 'book_equity'})

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'int_exp']],
            df_balance[['code', 'end_date', 'book_equity']],
            on=['code', 'end_date'],
            how='inner'
        )

        # Convert date
        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')

        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])

        df = df[df['book_equity'] > 0]
        df = df[df['revenue'] > 0]

        # Calculate lagged book equity (1 quarter)
        df['book_equity_lag1'] = df.groupby('code')['book_equity'].shift(1)

        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'int_exp']] = df[['sell_exp', 'admin_exp', 'int_exp']].fillna(0)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # Calculate Oleq = (revenue - COGS - SG&A) / lagged_book_equity
        df['Oleq'] = (df['revenue'] - df['cogs'] - df['sga'] - df['int_exp']) / df['book_equity_lag1']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', 'Oleq']].dropna()

    def validate(self) -> bool:
        """Validate Ole factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ole'].notna().sum()}")
        print(f"Statistics:\n{df['Ole'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedOlaFactor(OlaFactor):
    """修正Ola因子计算，修正分子计算补全减rd_exp、数据>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate Olarating leverage (assets) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_rd_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

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

        df = df[df['total_assets'] > 0]
        df = df[df['revenue'] > 0]

        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])

        # Calculate lagged total assets (1 quarter lag)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'rd_exp']] = df[['sell_exp', 'admin_exp', 'rd_exp']].fillna(0)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # Calculate Ola = (revenue - COGS - SG&A) / lagged_total_assets
        df['Ola'] = (df['revenue'] - df['cogs'] - df['sga'] + df['rd_exp']) / df['total_assets_lag1']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Ola factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ola'].notna().sum()}")
        print(f"Statistics:\n{df['Ola'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedOlaqFactor(OlaqFactor):
    """修正Ola因子计算，修正分子计算补全减rd_exp、数据>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate Olarating leverage (assets) factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_rd_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

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

        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')

        df = df[df['total_assets'] > 0]
        df = df[df['revenue'] > 0]

        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])

        # Calculate lagged total assets (1 quarter lag)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Fill NaN with 0 for expense items
        df[['sell_exp', 'admin_exp', 'rd_exp']] = df[['sell_exp', 'admin_exp', 'rd_exp']].fillna(0)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # Calculate Ola = (revenue - COGS - SG&A) / lagged_total_assets
        df['Ola'] = (df['revenue'] - df['cogs'] - df['sga'] + df['rd_exp']) / df['total_assets_lag1']

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Ola factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ola'].notna().sum()}")
        print(f"Statistics:\n{df['Ola'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedCopFactor(CopFactor):
    """修正Cop因子计算，修正分子计算补全rd_exp、delta_prepay、delta_def_rev、delta_acc_exp，增加分母>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate cash-based operating profitability factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_rd_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_rd_exp_fillna': 'rd_exp'
        })

        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna',
            'quarterly_prepayment_fillna',  # 添加预付费用
            'quarterly_deferred_inc_fillna',  # 添加递延收入
            'quarterly_acc_exp_fillna'  # 添加应计费用
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory',
            'quarterly_prepayment_fillna': 'prepayment',
            'quarterly_deferred_inc_fillna': 'deferred_inc',
            'quarterly_acc_exp_fillna': 'acc_exp'
        })

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'rd_exp']],
            df_balance[['code', 'end_date', 'total_assets', 'accounts_receiv', 'accounts_payable',
                        'inventory', 'prepayment', 'deferred_inc', 'acc_exp']],  # 添加新科目
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
        df['delta_prepay'] = df.groupby('code')['prepayment'].diff()
        df['delta_def_rev'] = df.groupby('code')['deferred_inc'].diff()
        df['delta_acc_exp'] = df.groupby('code')['acc_exp'].diff()

        # Fill NaN with 0 for expense items and working capital changes
        # 填充缺失值
        change_cols = ['delta_ar', 'delta_ap', 'delta_inv', 'delta_prepay', 'delta_def_rev', 'delta_acc_exp']
        df[change_cols] = df[change_cols].fillna(0)
        df['rd_exp'] = df['rd_exp'].fillna(0)  # R&D缺失设为0
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)

        df = df[df['total_assets'] > 0]

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # 修正Cop计算（修正符号和添加缺失项）
        df['Cop'] = (
                            df['revenue'] -
                            df['cogs'] -
                            df['sga'] +
                            df['rd_exp'] -  # 添加R&D
                            df['delta_ar'] -  # 修正：减号（原为加号）
                            df['delta_inv'] -
                            df['delta_prepay'] +  # 添加Δ预付费用
                            df['delta_def_rev'] +  # 添加Δ递延收入
                            df['delta_ap'] +  # 修正：加号（原为减号）
                            df['delta_acc_exp']  # 添加Δ应计费用
                    ) / df['total_assets']  # 修正分母：总资产

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Cop factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Cop'].notna().sum()}")
        print(f"Statistics:\n{df['Cop'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedClaFactor(ClaFactor):
    """修正Cla因子计算，修正分子计算补全rd_exp、delta_prepay、delta_def_rev、delta_acc_exp，增加分母>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate cash-based operating profitability factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_rd_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_rd_exp_fillna': 'rd_exp'
        })

        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna',
            'quarterly_prepayment_fillna',  # 添加预付费用
            'quarterly_deferred_inc_fillna',  # 添加递延收入
            'quarterly_acc_exp_fillna'  # 添加应计费用
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory',
            'quarterly_prepayment_fillna': 'prepayment',
            'quarterly_deferred_inc_fillna': 'deferred_inc',
            'quarterly_acc_exp_fillna': 'acc_exp'
        })

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'rd_exp']],
            df_balance[['code', 'end_date', 'total_assets', 'accounts_receiv', 'accounts_payable',
                        'inventory', 'prepayment', 'deferred_inc', 'acc_exp']],  # 添加新科目
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
        df['delta_prepay'] = df.groupby('code')['prepayment'].diff()
        df['delta_def_rev'] = df.groupby('code')['deferred_inc'].diff()
        df['delta_acc_exp'] = df.groupby('code')['acc_exp'].diff()

        # Fill NaN with 0 for expense items and working capital changes
        # 填充缺失值
        change_cols = ['delta_ar', 'delta_ap', 'delta_inv', 'delta_prepay', 'delta_def_rev', 'delta_acc_exp']
        df[change_cols] = df[change_cols].fillna(0)
        df['rd_exp'] = df['rd_exp'].fillna(0)  # R&D缺失设为0
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)

        df = df[df['total_assets'] > 0]
        # Calculate lagged total assets (1 year)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # 修正Cla计算（修正符号和添加缺失项）
        df['Cla'] = (
                            df['revenue'] -
                            df['cogs'] -
                            df['sga'] +
                            df['rd_exp'] -  # 添加R&D
                            df['delta_ar'] -  # 修正：减号（原为加号）
                            df['delta_inv'] -
                            df['delta_prepay'] +  # 添加Δ预付费用
                            df['delta_def_rev'] +  # 添加Δ递延收入
                            df['delta_ap'] +  # 修正：加号（原为减号）
                            df['delta_acc_exp']  # 添加Δ应计费用
                    ) / df['total_assets_lag1']  # 修正分母：总资产

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Cla factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Cla'].notna().sum()}")
        print(f"Statistics:\n{df['Cla'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedClaqFactor(ClaqFactor):
    """修正Claq因子计算，修正分子计算补全rd_exp、delta_prepay、delta_def_rev、delta_acc_exp，增加分母>0过滤，增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate cash-based operating profitability factor."""
        # Load income data
        income_cols = [
            'quarterly_total_revenue_fillna',
            'quarterly_total_cogs_fillna',
            'quarterly_sell_exp_fillna',
            'quarterly_admin_exp_fillna',
            'quarterly_rd_exp_fillna'
        ]
        df_income = load_income_data(income_cols, RESSET_DIR)

        df_income = df_income.rename(columns={
            'quarterly_total_revenue_fillna': 'revenue',
            'quarterly_total_cogs_fillna': 'cogs',
            'quarterly_sell_exp_fillna': 'sell_exp',
            'quarterly_admin_exp_fillna': 'admin_exp',
            'quarterly_rd_exp_fillna': 'rd_exp'
        })

        # Load balance sheet data
        balance_cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_liab_fillna',
            'quarterly_accounts_receiv_fillna',
            'quarterly_acct_payable_fillna',
            'quarterly_inventories_fillna',
            'quarterly_prepayment_fillna',  # 添加预付费用
            'quarterly_deferred_inc_fillna',  # 添加递延收入
            'quarterly_acc_exp_fillna'  # 添加应计费用
        ]
        df_balance = load_balancesheet_data(balance_cols, RESSET_DIR)
        df_balance = df_balance.rename(columns={
            'quarterly_total_assets_fillna': 'total_assets',
            'quarterly_accounts_receiv_fillna': 'accounts_receiv',
            'quarterly_acct_payable_fillna': 'accounts_payable',
            'quarterly_inventories_fillna': 'inventory',
            'quarterly_prepayment_fillna': 'prepayment',
            'quarterly_deferred_inc_fillna': 'deferred_inc',
            'quarterly_acc_exp_fillna': 'acc_exp'
        })

        # Merge
        df = pd.merge(
            df_income[['code', 'end_date', 'revenue', 'cogs', 'sell_exp', 'admin_exp', 'rd_exp']],
            df_balance[['code', 'end_date', 'total_assets', 'accounts_receiv', 'accounts_payable',
                        'inventory', 'prepayment', 'deferred_inc', 'acc_exp']],  # 添加新科目
            on=['code', 'end_date'],
            how='inner'
        )

        df['end_date'] = pd.to_datetime(df['end_date'], format='mixed', errors='coerce')

        # Sort by code and date
        df = df.sort_values(['code', 'end_date'])

        # Calculate changes in working capital (1-year lag)
        df['delta_ar'] = df.groupby('code')['accounts_receiv'].diff()
        df['delta_ap'] = df.groupby('code')['accounts_payable'].diff()
        df['delta_inv'] = df.groupby('code')['inventory'].diff()
        df['delta_prepay'] = df.groupby('code')['prepayment'].diff()
        df['delta_def_rev'] = df.groupby('code')['deferred_inc'].diff()
        df['delta_acc_exp'] = df.groupby('code')['acc_exp'].diff()

        # Fill NaN with 0 for expense items and working capital changes
        # 填充缺失值
        change_cols = ['delta_ar', 'delta_ap', 'delta_inv', 'delta_prepay', 'delta_def_rev', 'delta_acc_exp']
        df[change_cols] = df[change_cols].fillna(0)
        df['rd_exp'] = df['rd_exp'].fillna(0)  # R&D缺失设为0
        df[['sell_exp', 'admin_exp']] = df[['sell_exp', 'admin_exp']].fillna(0)

        df = df[df['total_assets'] > 0]
        # Calculate lagged total assets (1 year)
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Calculate SG&A
        df['sga'] = df['sell_exp'] + df['admin_exp']

        # 修正Cla计算（修正符号和添加缺失项）
        df['Claq'] = (
                            df['revenue'] -
                            df['cogs'] -
                            df['sga'] +
                            df['rd_exp'] -  # 添加R&D
                            df['delta_ar'] -  # 修正：减号（原为加号）
                            df['delta_inv'] -
                            df['delta_prepay'] +  # 添加Δ预付费用
                            df['delta_def_rev'] +  # 添加Δ递延收入
                            df['delta_ap'] +  # 修正：加号（原为减号）
                            df['delta_acc_exp']  # 添加Δ应计费用
                    ) / df['total_assets_lag1']  # 修正分母：总资产

        # Extract year
        df['year'] = df['end_date'].dt.year

        return df[['code', 'year', 'end_date', self.abbr]].dropna()

    def validate(self) -> bool:
        """Validate Claq factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Claq'].notna().sum()}")
        print(f"Statistics:\n{df['Claq'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedAdmFactor(AdmFactor):
    """修正adm factor，sell_exp取年末值，增加数据>0过滤及validate函数"""
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

            # Take sell_exp from the year end
            data_annual = data.groupby(['code', 'year']).tail(1).reset_index(drop=True)

            # Process market value: get December 31 values
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df = df.sort_values(['code', 'date'])

            # Get last trading day of each year
            df_dec = df.groupby(['code', 'year']).tail(1)

            # Merge advertising and market value
            result = pd.merge(data_annual, df_dec, on=['code', 'year'], how='left')
            result = result[result['circ_mv'] > 0]
            result = result[result['sell_exp'] > 0]

            # Calculate Adm = sell_exp / (circ_mv * 10000)
            result['Adm'] = result['sell_exp'] / (result['circ_mv'] * 10000)

            return result[['code', 'year', self.abbr]].dropna()

        except Exception as e:
            print(f"Warning: Could not calculate Adm factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'adm'])

    def validate(self) -> bool:
        """Validate Adm factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Adm'].notna().sum()}")
        print(f"Statistics:\n{df['Adm'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedgAdFactor(gAdFactor):
    """修正gAd factor，sell_exp取年末值，增加数据>0过滤及validate函数"""

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

            # Take sell_exp from the year end
            data_annual = data.groupby(['code', 'year']).tail(1).reset_index(drop=True)

            data_annual = data_annual[data_annual['sell_exp'] > 0]

            # Create lagged variables
            data_annual = data_annual.sort_values(['code', 'year'])
            data_annual['sell_exp_prior_1'] = data_annual.groupby('code')['sell_exp'].shift(1)
            data_annual['sell_exp_prior_2'] = data_annual.groupby('code')['sell_exp'].shift(2)

            # Calculate growth rate: (t-1 - t-2) / t-2
            data_annual['gAd'] = (data_annual['sell_exp_prior_1'] - data_annual['sell_exp_prior_2']) / data_annual[
                'sell_exp_prior_2']

            return data_annual[['code', 'year', self.abbr]].dropna()

        except Exception as e:
            print(f"Warning: Could not calculate gAd factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'gAd'])

    def validate(self) -> bool:
        """Validate gAd factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['gAd'].notna().sum()}")
        print(f"Statistics:\n{df['gAd'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedRdmFactor(RdmFactor):
    """修正Rdm factor，rd_exp数据改为利润表项目并取年末值，增加数据>0过滤及validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate R&D expense to market ratio."""
        import os
        try:
            # Load R&D data from income
            income_file = DATADIR.parent / "tushare" / "income.csv"
            if not income_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'Rdm'])

            data = pd.read_csv(income_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'rd_exp']]

            # Load market value data from daily_basic
            db_file = DATADIR.parent / "tushare" / "daily_basic.csv"
            if not db_file.exists():
                print(f"Warning: Daily basic file not found: {db_file}")
                return pd.DataFrame(columns=['code', 'year', 'Rdm'])

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
            data = data[data['rd_exp'] > 0]

            # Take rd_exp from the year end
            data_annual = data.groupby(['code', 'year']).tail(1).reset_index(drop=True)

            # Process market value: get December 31 values
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df = df.sort_values(['code', 'date'])

            # Get last trading day of each year (December values)
            df_dec = df.groupby(['code', 'year']).tail(1)

            # Merge R&D and market value
            result = pd.merge(data_annual, df_dec, on=['code', 'year'], how='left')

            # Calculate Rdm = rd_exp / (circ_mv * 10000)
            # circ_mv is in 万元 (10,000 yuan), need to multiply by 10000 to match rd_exp units
            result['Rdm'] = result['rd_exp'] / (result['circ_mv'] * 10000)

            return result[['code', 'year', self.abbr]].dropna()

        except Exception as e:
            print(f"Warning: Could not calculate Rdm factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'Rdm'])

    def validate(self) -> bool:
        """Validate Rdm factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Rdm'].notna().sum()}")
        print(f"Statistics:\n{df['Rdm'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedRdsFactor(RdsFactor):
    """修正Rds factor，rd_exp数据改为利润表项目并取年末值，增加数据>0过滤及validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate R&D expense to market ratio."""
        import os
        try:
            # Load R&D data from income
            income_file = DATADIR.parent / "tushare" / "income.csv"
            if not income_file.exists():
                return pd.DataFrame(columns=['code', 'year', 'Rds'])

            data = pd.read_csv(income_file)
            data = data.rename(columns={'ts_code': 'code', 'ann_date': 'date'})
            data = data[['code', 'date', 'rd_exp', 'revenue']]

            # Standardize codes
            data['code'] = data['code'].apply(standardize_ticker)

            # Process R&D data
            data['date'] = pd.to_datetime(data['date'].astype(str), errors='coerce')
            data = data.dropna(subset=['date'])
            data['year'] = data['date'].dt.year
            data = data.sort_values(['code', 'date'])

            # Remove zero R&D
            data = data[data['rd_exp'] > 0]
            data = data[data['revenue'] > 0]

            # Take rd_exp from the year end
            data_annual = data.groupby(['code', 'year']).tail(1).reset_index(drop=True)

            # Calculate Rds = rd_exp / revenue
            data_annual['Rds'] = data_annual['rd_exp'] / data_annual['revenue']

            return data_annual[['code', 'year', self.abbr]].dropna()

        except Exception as e:
            print(f"Warning: Could not calculate Rds factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'Rds'])

    def validate(self) -> bool:
        """Validate Rds factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Rds'].notna().sum()}")
        print(f"Statistics:\n{df['Rds'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True

class FixedAnaFactor(AnaFactor):
    """修正Ana factor，将因子频率改为月度并增加validate函数"""
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate analyst coverage factor."""
        try:
            import os
            # Load analyst report data
            report_file = DATADIR.parent / "tushare" / "report_rc.csv"
            if not report_file.exists():
                print(f"Warning: Analyst report file not found: {report_file}")
                return pd.DataFrame(columns=['code', 'year', 'month', 'Ana'])

            df = pd.read_csv(report_file)
            df = df.rename(columns={'ts_code': 'code', 'report_date': 'date'})
            df = df[['code', 'date', 'author_name']].drop_duplicates()

            # Standardize code
            df['code'] = df['code'].apply(standardize_ticker)

            # Extract year and month
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
            df = df.dropna(subset=['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Count unique analysts per stock per year
            df['coverage'] = 1
            df_Ana = df.groupby(['code', 'year', 'month'])['coverage'].sum().reset_index()
            df_Ana = df_Ana.rename(columns={'coverage': 'Ana'})

            return df_Ana

        except Exception as e:
            print(f"Warning: Could not calculate Ana factor: {e}")
            return pd.DataFrame(columns=['code', 'year', 'month', 'Ana'])

    def validate(self) -> bool:
        """Validate Ana factor calculation."""
        df = self.calculate()
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df = self.calculate()
        print(f"Shape: {df.shape}")
        print(f"Non-NaN values: {df['Ana'].notna().sum()}")
        print(f"Statistics:\n{df['Ana'].describe()}")
        print(f"Sample:\n{df.head(20)}")

        print(f"Validation complete for {self.abbr}!")
        return True