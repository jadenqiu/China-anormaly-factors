import sys
from pathlib import Path
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from more_itertools.recipes import factor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from china_anomalies_factors import (
    A1MomentumFactorBase, A2ValueFactorBase, A3InvestmentFactorBase, A5IntangiblesFactorBase, A6TradingFrictionsFactorBase,
    load_fina_indicator_data, get_annual_shift_data, get_monthly_mv, load_cashflow_data, load_wind_daily_data,
    load_balancesheet_data, load_income_data, get_quartly_be, date_transfer, get_monthly_share, load_daily_basic_data,
    RESSET_DIR, TUSHARE_DIR, _data_dir, plot_date_counts, plotVarTrend
)

class RsFactor(A1MomentumFactorBase):
    """
    A.1.7 Rs - Revenue Surprises

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
    A.1.8 Tes - Tax Expense Surprises

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

        # Exclude tax<=0
        df = df[df['taxes_payable'] > 0]

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
    A.1.10 Nei - Number of Quarters with Consecutive Earnings Increase

    Counts the number of consecutive quarters with earnings growth compared to the same quarter in the prior year.
    Capped at maximum of 8 consecutive quarters.

    Formula:
        Nei = Count of consecutive quarters where NI_t > NI_{t-4} (max 8)

    where:
        NI_t: net income in quarter t
        NI_{t-4}: net income in the same quarter of the previous year
    """

    @property
    def factor_id(self) -> str:
        return "A.1.10"

    @property
    def abbr(self) -> str:
        return "nei"

    def _calculate_year_over_year_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate year-over-year growth and count consecutive growth quarters.

        Args:
            df: DataFrame with net income data sorted by code and end_date

        Returns:
            DataFrame with added 'growth_flag' and 'nei' columns
        """
        # Sort by code and date to ensure proper shifting
        df = df.sort_values(['code', 'end_date'])

        # Calculate year-over-year comparison (shift by 4 quarters for same quarter last year)
        df['n_income_lag4'] = df.groupby('code')['n_income'].shift(4)

        # Create growth flag: 1 if current quarter > same quarter last year, else 0
        df['growth_flag'] = (df['n_income'] > df['n_income_lag4']).astype(int)

        # For the first 4 quarters of each stock, we can't calculate YoY growth
        df.loc[df['n_income_lag4'].isna(), 'growth_flag'] = 0

        return df

    def _count_consecutive_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count consecutive quarters with year-over-year growth.

        Args:
            df: DataFrame with 'growth_flag' column indicating growth (1) or not (0)

        Returns:
            DataFrame with 'nei' column counting consecutive growth quarters
        """
        # Sort to ensure proper grouping
        df = df.sort_values(['code', 'end_date'])

        # Identify groups of consecutive growth quarters
        def calculate_sequence(series):
            result = []
            count = 0

            for val in series:
                if val == 1:
                    count += 1
                    result.append(count)
                else:
                    count = 0
                    result.append(count)

            return pd.Series(result, index=series.index)

        # 对每个公司分别计算
        df['consecutive_count'] = df.groupby('code')['growth_flag'].transform(calculate_sequence)

        # Cap at 8 consecutive quarters
        df['nei'] = df['consecutive_count'].clip(upper=8)

        return df

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate Nei factor - consecutive quarters with year-over-year earnings growth.

        Returns:
            DataFrame with columns: code, end_date, ann_date, nei
        """
        # Load net income data
        df = load_income_data(['quarterly_n_income_fillna'])

        # Use the fillna version for robustness
        df = df[['code', 'quarterly_n_income_fillna', 'end_date', 'ann_date']]
        df = df.rename(columns={'quarterly_n_income_fillna': 'n_income'})

        # Ensure proper date sorting
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df.sort_values(['code', 'end_date'])

        # Remove any rows with missing net income
        df = df.dropna(subset=['n_income'])

        # Step 1: Calculate year-over-year growth flags
        df = self._calculate_year_over_year_growth(df)

        # Step 2: Count consecutive growth quarters
        df = self._count_consecutive_growth(df)

        return df[['code', 'end_date', 'ann_date', 'nei']].dropna()

    def validate(self) -> bool:
        """Run validation tests for Nei factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_nei = self.calculate()
        print(f"Shape: {df_nei.shape}")
        print(f"Non-NaN values: {df_nei['nei'].notna().sum()}")
        print(f"Statistics:\n{df_nei['nei'].describe()}")

        # Check the distribution of Nei values
        nei_counts = df_nei['nei'].value_counts().sort_index()
        print(f"Nei value distribution:\n{nei_counts}")

        # Verify year-over-year logic with a sample stock
        sample_code = df_nei['code'].iloc[0] if len(df_nei) > 0 else None
        if sample_code:
            sample_data = df_nei[df_nei['code'] == sample_code].sort_values('end_date').head(12)
            print(f"Sample data for {sample_code}:\n{sample_data[['end_date', 'nei']]}")

        print(f"Sample of fulln{df_nei.head(10)}")

        # Convert to factor frame
        ft_nei = self.as_factor_frame(df_nei)
        print(f"Factor frame shape: {ft_nei.shape}")
        print(f"Factor frame sample:\n{ft_nei.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


# =============================================================================
# SECTION: A2 VALUE FACTOR IMPLEMENTATIONS
# =============================================================================

class BmjFactor(A2ValueFactorBase):
    """
    A.2.2 Bmj - Book-to-June-end Market Equity

    At the end of June of each year t, we sort stocks into deciles based on Bmj,
    which is book equity per share for the fiscal year ending in calendar year t-1
    divided by share price(from CRSP) at the end of June of t.

    Formula:
        Bmj = BookEquity_per_share_{t-1} / Price_June_t

    Data source:
        - Book equity: calculated from balance sheet data
        - Share price: from daily basic data

    Implementation:
        - Uses fiscal year ending in t-1
        - Uses June end price of year t
        - Adjusts for stock splits between fiscal year end and June
    """

    @property
    def factor_id(self) -> str:
        return "A.2.2"

    @property
    def abbr(self) -> str:
        return "Bmj"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate Bmj factor - book-to-June-end market equity.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Bmj']
        """

        # Load book equity data (balance sheet)
        be_data = load_balancesheet_data(
            ['quarterly_total_share_fillna', 'quarterly_total_hldr_eqy_inc_min_int_fillna']
        )

        be_data.rename(columns={'quarterly_total_hldr_eqy_inc_min_int_fillna': 'quarterly_book_equity'}, inplace=True)

        # Filter for December fiscal year-end only
        be_data = be_data[pd.to_datetime(be_data['end_date']).dt.month == 12]
        be_data = be_data[['code', 'end_date', 'quarterly_book_equity', 'quarterly_total_share_fillna']]

        # Calculate book equity per share
        be_data['be_per_shr'] = (
                be_data['quarterly_book_equity'] / be_data['quarterly_total_share_fillna']
        )
        # Handle division errors
        be_data = be_data.replace([np.inf, -np.inf], np.nan).dropna()

        # 2. Load market price data (June prices)
        price_data = load_daily_basic_data(['close'])
        # Filter for June month-end only
        price_data = price_data[pd.to_datetime(price_data['date']).dt.month == 6]
        price_data = price_data[['code', 'date', 'close']]

        # 3. Align dates: December t-1 with June t
        # Extract years for alignment
        be_data['year'] = pd.to_datetime(be_data['end_date']).dt.year
        price_data['year'] = pd.to_datetime(price_data['date']).dt.year

        # December of year t-1 corresponds to June of year t
        be_data['june_year'] = be_data['year'] + 1

        # Merge book equity (Dec t-1) with prices (Jun t)
        result = pd.merge(
            be_data[['code', 'end_date', 'be_per_shr', 'june_year']],
            price_data[['code', 'year', 'close']].rename(columns={'year': 'june_year'}),
            on=['code', 'june_year'],
            how='inner'  # Inner join to avoid NaN values
        )

        # 4. Calculate Bmj factor
        result['Bmj'] = result['be_per_shr'] / result['close']
        # Clean invalid values
        result = result.replace([np.inf, -np.inf], np.nan)

        return result[['code', 'end_date', 'Bmj']]

    def validate(self) -> bool:
        """Validate Bmj factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_bmj = self.calculate()
        print(f"Shape: {df_bmj.shape}")
        print(f"Non-NaN values: {df_bmj['Bmj'].notna().sum()}")
        print(f"Statistics:\n{df_bmj['Bmj'].describe()}")
        print(f"Sample:\n{df_bmj.head(10)}")

        # Check for common issues
        n_inf = (df_bmj['Bmj'] == np.inf).sum()
        n_neg = (df_bmj['Bmj'] < 0).sum()
        print(f"Inf values: {n_inf}")
        print(f"Negative values: {n_neg}")

        # Convert to factor frame
        ft_bmj = self.as_factor_frame(df_bmj)
        print(f"Factor frame shape: {ft_bmj.shape}")
        print(f"Factor frame sample:\n{ft_bmj.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class BmqFactor(A2ValueFactorBase):
    """
    A.2.3 Bmq1, Bmq6, and Bmq12 - Quarterly Book-to-Market Equity

    At the beginning of each month t, we split stocks into deciles based on Bmq,
    which is the book equity for the latest fiscal quarter ending at least four
    months ago divided by the market equity at the end of month t-1.

    Formula:
        Bmq = BookEquity_{quarter_t-4} / MarketEquity_{month_t-1}

    Data source:
        - Book equity: quarterly balance sheet data
        - Market equity: monthly market value data

    Implementation:
        - Uses quarterly data with 4-month lag
        - For firms with multiple share classes, merges market equity
        - Calculates returns for different holding periods
    """

    @property
    def factor_id(self) -> str:
        return "A.2.3"

    @property
    def abbr(self) -> str:
        return "Bmq"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate quarterly book-to-market equity factor.

        Args:
            shift_month: Number of months to shift quarterly data (default: 4)

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Bmq']
        """
        # 1. Load book equity data (quarterly balance sheet)
        be_data = load_balancesheet_data(
            ['quarterly_total_hldr_eqy_inc_min_int_fillna']
        )

        be_data.rename(columns={'quarterly_total_hldr_eqy_inc_min_int_fillna': 'quarterly_book_equity'}, inplace=True)

        # Select only necessary columns
        be_data = be_data[['code', 'end_date', 'quarterly_book_equity']]

        # Ensure end_date is datetime
        be_data['end_date'] = pd.to_datetime(be_data['end_date'], errors='coerce')
        be_data = be_data.sort_values(['code', 'end_date'])
        be_data = be_data.replace([np.inf, -np.inf], np.nan).dropna()

        # 2. Load market value data (monthly)
        mv_data = get_monthly_mv()
        mv_data = mv_data[['code', 'date', 'monthly_mv_end']]
        mv_data['date'] = pd.to_datetime(mv_data['date'], errors='coerce')

        # 3. Align dates: quarter t-4 with month t-1
        # Extract year and month for alignment
        be_data['year'] = be_data['end_date'].dt.year
        be_data['month'] = be_data['end_date'].dt.month
        be_data['quarter'] = (be_data['month'] - 1) // 3 + 1

        mv_data['year'] = mv_data['date'].dt.year
        mv_data['month'] = mv_data['date'].dt.month

        # Shift book equity by 4 quarters (1 year)
        be_data = be_data.sort_values(['code', 'year', 'quarter'])
        be_data['quarterly_book_equity_lag4'] = be_data.groupby('code')['quarterly_book_equity'].shift(4)
        be_data['end_date_lag4'] = be_data.groupby('code')['end_date'].shift(4)

        # Shift market value by 1 month
        mv_data = mv_data.sort_values(['code', 'year', 'month'])
        mv_data['monthly_mv_end_lag1'] = mv_data.groupby('code')['monthly_mv_end'].shift(1)
        mv_data['date_lag1'] = mv_data.groupby('code')['date'].shift(1)

        # 4. Merge data on aligned dates
        # Use the lagged dates for merging
        result = pd.merge(
            be_data[['code', 'end_date_lag4', 'quarterly_book_equity_lag4']].rename(
                columns={'end_date_lag4': 'book_date'}),
            mv_data[['code', 'date_lag1', 'monthly_mv_end_lag1']].rename(
                columns={'date_lag1': 'market_date'}),
            left_on=['code', 'book_date'],
            right_on=['code', 'market_date'],
            how='inner'
        )

        # 5. Calculate Bmq factor
        result['Bmq'] = result['quarterly_book_equity_lag4'] / result['monthly_mv_end_lag1']
        result = result.replace([np.inf, -np.inf], np.nan).dropna()

        # Use the market value date as the reference date
        result = result.rename(columns={'market_date': 'end_date'})

        return result[['code', 'end_date', 'Bmq']]

    def validate(self) -> bool:
        """Validate Bmq factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor with default parameters
        df_bmq = self.calculate(shift_month=4)
        print(f"Shape: {df_bmq.shape}")
        print(f"Non-NaN values: {df_bmq['Bmq'].notna().sum()}")
        print(f"Statistics:\n{df_bmq['Bmq'].describe()}")
        print(f"Sample:\n{df_bmq.head(10)}")

        # Check date alignment
        df_bmq['year'] = df_bmq['end_date'].dt.year
        df_bmq['month'] = df_bmq['end_date'].dt.month
        print(f"Date range: {df_bmq['end_date'].min()} to {df_bmq['end_date'].max()}")
        print(f"Year coverage: {sorted(df_bmq['year'].unique())}")

        # Convert to factor frame
        ft_bmq = self.as_factor_frame(df_bmq)
        print(f"Factor frame shape: {ft_bmq.shape}")
        print(f"Factor frame sample:\n{ft_bmq.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class SrFactor(A2ValueFactorBase):
    """
    A.2.18 Sr - 5-Year Sales Growth Rank

    Following Lakonishok, Shleifer, and Vishny(1994), we measure 5-year sales
    growth rank, Sr, in June of year t as the weighted average of the annual
    sales growth ranks for the prior five years.

    Formula:
        Sr = Σ_{j=1}^5 (6-j) × Rank(SalesGrowth_{t-j})

    where SalesGrowth for year t-j is the growth rate in sales from fiscal year
    ending in t-j-1 to fiscal year ending in t-j.

    Data source:
        - Sales data: from cash flow statement
    """

    @property
    def factor_id(self) -> str:
        return "A.2.18"

    @property
    def abbr(self) -> str:
        return "Sr"

    def _linearly_weighted_average(self, x: pd.Series) -> float:
        """
        Calculate linearly weighted average with declining weights.

        Args:
            x: Series of values to weight

        Returns:
            Weighted average
        """
        weights = pd.Series(range(5, 0, -1))  # [5, 4, 3, 2, 1]
        return (x * weights).sum() / weights.sum()

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate 5-year sales growth rank factor.

        Returns:
            DataFrame with columns: ['code', 'year', 'Sr']
        """
        # Use cash flow sales data as proxy for SALE
        data = load_cashflow_data(['c_fr_sale_sg'], RESSET_DIR)
        data = data[['code', 'end_date', 'c_fr_sale_sg']]
        data = date_transfer(data, 'end_date')

        # Keep only year-end data (December)
        data = data[data['month'] == 12]

        # Get all unique codes and years
        all_years = data['year'].sort_values().unique().tolist()
        all_codes = data['code'].sort_values().unique().tolist()

        # Create full panel to ensure we have all year-code combinations
        data = data.set_index(['code', 'year']).reindex(
            pd.MultiIndex.from_product([all_codes, all_years], names=['code', 'year'])
        ).sort_index().reset_index(drop=False)

        # Calculate year-over-year growth rate
        data['growth'] = data.groupby(['code'])['c_fr_sale_sg'].pct_change()
        data = data.sort_values(['code', 'year'])
        data['valid_growth_count'] = data.groupby('code')['growth'].transform(
            lambda x: x.rolling(5, min_periods=1).count()
        )

        # Calculate conditional normalized rank
        def conditional_rank(group):
            """Calculate rank only for groups with >=5 valid data points"""
            has_sufficient_data = group['valid_growth_count'] >= 5
            result = pd.Series([0.5] * len(group), index=group.index)

            if has_sufficient_data.any():
                eligible = group[has_sufficient_data]
                ranks = eligible['growth'].rank(method='average')
                result[eligible.index] = ranks / len(eligible)

            return result

        data['normalized_rank'] = data.groupby('year').apply(conditional_rank).reset_index(level=0, drop=True)
        data['normalized_rank'] = data['normalized_rank'].fillna(0.5)

        # Calculate final Sr factor
        data['Sr'] = data.groupby('code')['normalized_rank'].transform(
            lambda x: x.rolling(5, min_periods=1).apply(self._linearly_weighted_average, raw=False)
        )

        data['year'] = data['year'].astype(int)

        return data[['code', 'year', 'Sr']].dropna()

    def validate(self) -> bool:
        """Validate Sr factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_sr = self.calculate()
        print(f"Shape: {df_sr.shape}")
        print(f"Non-NaN values: {df_sr['Sr'].notna().sum()}")
        print(f"Statistics:\n{df_sr['Sr'].describe()}")
        print(f"Sample:\n{df_sr.head(15)}")

        # Check the weighting mechanism
        sample_stock = df_sr['code'].iloc[0]
        sample_data = df_sr[df_sr['code'] == sample_stock].head(10)
        print(f"Sample stock {sample_stock}n{sample_data}")

        # Convert to factor frame
        ft_sr = self.as_factor_frame(df_sr, date_col='year')
        print(f"Factor frame shape: {ft_sr.shape}")
        print(f"Factor frame sample:\n{ft_sr.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class SgFactor(A2ValueFactorBase):
    """
    A.2.19 Sg - Sales Growth

    At the end of June of each year t, we assign stocks into deciles based on Sg,
    which is the growth in annual sales from the fiscal year ending in calendar
    year t-2 to the fiscal year ending in t-1.

    Formula:
        Sg = (Sales_{t-1} - Sales_{t-2}) / Sales_{t-2}

    Data source:
        - Sales data: from cash flow statement

    Implementation:
        - Uses annual data (December fiscal year-end)
        - Excludes firms with nonpositive sales
    """

    @property
    def factor_id(self) -> str:
        return "A.2.19"

    @property
    def abbr(self) -> str:
        return "Sg"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate sales growth factor.

        Returns:
            DataFrame with columns: ['code', 'year', 'Sg']
        """
        # Use cash flow sales data as proxy for SALE
        data = load_cashflow_data(['c_fr_sale_sg'], RESSET_DIR)

        # Keep only year-end data (December)
        data = date_transfer(data, 'end_date')
        data = data[data['month'] == 12]

        # Calculate year-over-year growth rate
        data['growth'] = data.groupby(['code'])['c_fr_sale_sg'].pct_change()
        data['Sg'] = data['growth']

        # Exclude firms with nonpositive sales
        data = data[data['c_fr_sale_sg'] > 0]

        data['year'] = data['year'].astype(int)

        return data[['code', 'year', 'Sg']].dropna()

    def validate(self) -> bool:
        """Validate Sg factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_sg = self.calculate()
        print(f"Shape: {df_sg.shape}")
        print(f"Non-NaN values: {df_sg['Sg'].notna().sum()}")
        print(f"Statistics:\n{df_sg['Sg'].describe()}")
        print(f"Sample:\n{df_sg.head(10)}")

        # Check growth rate calculation
        sample_stock = df_sg['code'].iloc[0] if len(df_sg) > 0 else None
        if sample_stock:
            sample_data = df_sg[df_sg['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} growth rates:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_sg['Sg'] > 10).sum()
        extreme_low = (df_sg['Sg'] < -0.9).sum()
        print(f"Extreme high growth (>1000%): {extreme_high}")
        print(f"Extreme negative growth (<-90%): {extreme_low}")

        # Convert to factor frame
        ft_sg = self.as_factor_frame(df_sg, date_col='year')
        print(f"Factor frame shape: {ft_sg.shape}")
        print(f"Factor frame sample:\n{ft_sg.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


# =============================================================================
# SECTION: A3 INVESTMENT FACTOR
# =============================================================================
class CeiFactor(A3InvestmentFactorBase):
    """
    A.3.12 Cei - Composite Equity Issuance

    Composite equity issuance measures the growth in market equity not attributable
    to stock returns. It captures the net effect of equity issuance and repurchases.

    Formula:
        Cei = log(Met/Met−5) - r(t−5,t)

    where:
        - Met: market equity at the end of June in year t
        - Met−5: market equity at the end of June in year t−5
        - r(t−5,t): cumulative log stock return from June t−5 to June t

    Data Sources:
        - Market value: total_mv from daily_basic_resset.csv
        - Stock prices: close from Wind daily return data

    Implementation:
        - Uses June month-end data for alignment with fiscal year
        - Calculates 5-year cumulative returns and market equity growth
        - Composite issuance = market equity growth - stock return
    """

    @property
    def factor_id(self) -> str:
        return "A.3.12"

    @property
    def abbr(self) -> str:
        return "cei"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate composite equity issuance factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'cei']
        """
        # Load daily basic data for market value
        df = load_daily_basic_data(['total_mv', 'close'], RESSET_DIR)

        # Convert date and filter to June month-end
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Keep only June observations
        df_june = df[df['month'] == 6].copy()
        df_june = df_june.sort_values(['code', 'date'])

        # Calculate 5-year cumulative return and market equity growth
        df_june['close_lag5'] = df_june.groupby('code')['close'].shift(5)
        df_june['total_mv_lag5'] = df_june.groupby('code')['total_mv'].shift(5)

        # Calculate cumulative return (r(t-5,t))
        df_june['cumulative_return'] = np.log(df_june['close'] / df_june['close_lag5'])

        # Calculate market equity growth (log(Met/Met-5))
        df_june['me_growth'] = np.log(df_june['total_mv'] / df_june['total_mv_lag5'])

        # Calculate composite equity issuance
        df_june['cei'] = df_june['me_growth'] - df_june['cumulative_return']

        # Use date as end_date for output consistency
        df_june = df_june.rename(columns={'date': 'end_date'})

        return df_june[['code', 'end_date', 'cei']].dropna()

    def validate(self) -> bool:
        """Validate Cei factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_cei = self.calculate()
        print(f"Shape: {df_cei.shape}")
        print(f"Non-NaN values: {df_cei['cei'].notna().sum()}")
        print(f"Statistics:\n{df_cei['cei'].describe()}")
        print(f"Sample:\n{df_cei.head(10)}")

        # Check composite equity issuance calculation
        sample_stock = df_cei['code'].iloc[0] if len(df_cei) > 0 else None
        if sample_stock:
            sample_data = df_cei[df_cei['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} composite equity issuance:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_cei['cei'] > 5).sum()
        extreme_low = (df_cei['cei'] < -5).sum()
        print(f"Extreme high CEI (>5): {extreme_high}")
        print(f"Extreme low CEI (<-5): {extreme_low}")

        # Check June month filtering
        june_months = pd.to_datetime(df_cei['end_date']).dt.month
        print(f"June observations: {(june_months == 6).sum()}")
        print(f"Non-June observations: {(june_months != 6).sum()}")

        # Convert to factor frame
        ft_cei = self.as_factor_frame(df_cei)
        print(f"Factor frame shape: {ft_cei.shape}")
        print(f"Factor frame sample:\n{ft_cei.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class CdiFactor(A3InvestmentFactorBase):
    """
    A.3.13 Cdi - Composite Debt Issuance

    Composite debt issuance measures the growth in the book value of debt over
    a 6-year period. It captures the net effect of debt issuance and repayments.

    Formula:
        Cdi = log(Total Debtt−1 / Total Debtt−6)

    where Total Debt = Short-term debt (DLC) + Long-term debt (DLTT)

    Data Sources:
        - Balance sheet: st_borr, st_bonds_payable, non_cur_liab_due_1y,
                         bond_payable, lt_payable, lt_borr from quarterly_balancesheet_cleaned.csv

    Implementation:
        - Uses annual fiscal year-end data (December)
        - Calculates 6-year log growth in total debt
        - Debt components include all interest-bearing liabilities
    """

    @property
    def factor_id(self) -> str:
        return "A.3.13"

    @property
    def abbr(self) -> str:
        return "cdi"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate composite debt issuance factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'cdi']
        """
        # Load debt-related balance sheet items
        debt_cols = [
            'quarterly_st_borr_fillna', 'quarterly_st_bonds_payable_fillna',
            'quarterly_non_cur_liab_due_1y_fillna', 'quarterly_bond_payable_fillna',
            'quarterly_lt_payable_fillna', 'quarterly_lt_borr_fillna'
        ]
        df = load_balancesheet_data(debt_cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Calculate total debt (sum of all interest-bearing liabilities)
        df['total_debt'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna'] +
                df['quarterly_bond_payable_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_lt_borr_fillna']
        )

        # Calculate 6-year log growth in total debt
        df['total_debt_lag6'] = df.groupby('code')['total_debt'].shift(6)

        # Calculate Cdi with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            df['cdi'] = np.log(df['total_debt'] / df['total_debt_lag6'])

        # Replace inf/-inf with NaN
        df['cdi'] = df['cdi'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'cdi']].dropna()

    def validate(self) -> bool:
        """Validate Cdi factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_cdi = self.calculate()
        print(f"Shape: {df_cdi.shape}")
        print(f"Non-NaN values: {df_cdi['cdi'].notna().sum()}")
        print(f"Statistics:\n{df_cdi['cdi'].describe()}")
        print(f"Sample:\n{df_cdi.head(10)}")

        # Check composite debt issuance calculation
        sample_stock = df_cdi['code'].iloc[0] if len(df_cdi) > 0 else None
        if sample_stock:
            sample_data = df_cdi[df_cdi['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} composite debt issuance:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_cdi['cdi'] > 3).sum()
        extreme_low = (df_cdi['cdi'] < -3).sum()
        print(f"Extreme high CDI (>3): {extreme_high}")
        print(f"Extreme low CDI (<-3): {extreme_low}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_cdi['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check total debt components
        if 'total_debt' in df_cdi.columns:
            zero_debt = (df_cdi['total_debt'] == 0).sum()
            print(f"Stocks with zero total debt: {zero_debt}")

        # Convert to factor frame
        ft_cdi = self.as_factor_frame(df_cdi)
        print(f"Factor frame shape: {ft_cdi.shape}")
        print(f"Factor frame sample:\n{ft_cdi.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class IvgFactor(A3InvestmentFactorBase):
    """
    A.3.14 Ivg - Inventory Growth

    Inventory growth measures the annual growth rate in inventory levels.
    High inventory growth may indicate overproduction or slowing sales.

    Formula:
        Ivg = (Inventoryt−1 - Inventoryt−2) / Inventoryt−2

    Data Sources:
        - Inventory: quarterly_inventories_fillna from quarterly_balancesheet_cleaned.csv

    Implementation:
        - Uses annual fiscal year-end data (December)
        - Calculates year-over-year inventory growth
        - Simple percentage change calculation
    """

    @property
    def factor_id(self) -> str:
        return "A.3.14"

    @property
    def abbr(self) -> str:
        return "ivg"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate inventory growth factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'ivg']
        """
        # Load inventory data
        df = load_balancesheet_data(['quarterly_inventories_fillna'], RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        df = df[df['quarterly_inventories_fillna'] > 0]

        # Calculate year-over-year inventory growth
        df['inventory_lag1'] = df.groupby('code')['quarterly_inventories_fillna'].shift(1)
        df['inventory_lag2'] = df.groupby('code')['quarterly_inventories_fillna'].shift(2)

        # Calculate Ivg with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ivg'] = (df['inventory_lag1'] - df['inventory_lag2']) / df['inventory_lag2']

        # Replace inf/-inf with NaN
        df['ivg'] = df['ivg'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'ivg']].dropna()

    def validate(self) -> bool:
        """Validate Ivg factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ivg = self.calculate()
        print(f"Shape: {df_ivg.shape}")
        print(f"Non-NaN values: {df_ivg['ivg'].notna().sum()}")
        print(f"Statistics:\n{df_ivg['ivg'].describe()}")
        print(f"Sample:\n{df_ivg.head(10)}")

        # Check inventory growth calculation
        sample_stock = df_ivg['code'].iloc[0] if len(df_ivg) > 0 else None
        if sample_stock:
            sample_data = df_ivg[df_ivg['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} inventory growth:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_ivg['ivg'] > 10).sum()
        extreme_low = (df_ivg['ivg'] < -0.9).sum()
        infinite_values = np.isinf(df_ivg['ivg']).sum()
        print(f"Extreme high growth (>1000%): {extreme_high}")
        print(f"Extreme negative growth (<-90%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_ivg['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check inventory data quality
        if 'quarterly_inventories_fillna' in df_ivg.columns:
            zero_inventory = (df_ivg['quarterly_inventories_fillna'] == 0).sum()
            print(f"Stocks with zero inventory: {zero_inventory}")

        # Convert to factor frame
        ft_ivg = self.as_factor_frame(df_ivg)
        print(f"Factor frame shape: {ft_ivg.shape}")
        print(f"Factor frame sample:\n{ft_ivg.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class IvcFactor(A3InvestmentFactorBase):
    """
    A.3.15 Ivc - Inventory Changes

    Inventory changes measure the annual change in inventory scaled by the
    average of total assets over the past two years.

    Formula:
        Ivc = (Inventoryt−1 - Inventoryt−2) / Average(Total Assetst−2, Total Assetst−1)

    Data Sources:
        - Inventory: quarterly_inventories_fillna from quarterly_balancesheet_cleaned.csv
        - Total assets: quarterly_total_assets_fillna from quarterly_balancesheet_cleaned.csv

    Implementation:
        - Uses annual fiscal year-end data (December)
        - Excludes firms with no inventory for the past two years
        - Uses 2-year average total assets as scaling factor
    """

    @property
    def factor_id(self) -> str:
        return "A.3.15"

    @property
    def abbr(self) -> str:
        return "ivc"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate inventory changes factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'ivc']
        """
        # Load inventory and total assets data
        cols = ['quarterly_inventories_fillna', 'quarterly_total_assets_fillna']
        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Calculate inventory change
        df['inventory_lag1'] = df.groupby('code')['quarterly_inventories_fillna'].shift(1)
        df['inventory_lag2'] = df.groupby('code')['quarterly_inventories_fillna'].shift(2)

        # Calculate 2-year average total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df['avg_total_assets'] = (df['total_assets_lag1'] + df['total_assets_lag2']) / 2

        # Calculate Ivc with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ivc'] = (df['inventory_lag1'] - df['inventory_lag2']) / df['avg_total_assets']

        # Replace inf/-inf with NaN and filter out zero inventory firms
        df['ivc'] = df['ivc'].replace([np.inf, -np.inf], np.nan)
        df = df[(df['inventory_lag1'] > 0) & (df['inventory_lag2'] > 0)]

        return df[['code', 'end_date', 'ivc']].dropna()

    def validate(self) -> bool:
        """Validate Ivc factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ivc = self.calculate()
        print(f"Shape: {df_ivc.shape}")
        print(f"Non-NaN values: {df_ivc['ivc'].notna().sum()}")
        print(f"Statistics:\n{df_ivc['ivc'].describe()}")
        print(f"Sample:\n{df_ivc.head(10)}")

        # Check inventory changes calculation
        sample_stock = df_ivc['code'].iloc[0] if len(df_ivc) > 0 else None
        if sample_stock:
            sample_data = df_ivc[df_ivc['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} inventory changes:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_ivc['ivc'] > 1).sum()
        extreme_low = (df_ivc['ivc'] < -1).sum()
        infinite_values = np.isinf(df_ivc['ivc']).sum()
        print(f"Extreme high changes (>100%): {extreme_high}")
        print(f"Extreme negative changes (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_ivc['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        if 'avg_total_assets' in df_ivc.columns:
            zero_assets = (df_ivc['avg_total_assets'] == 0).sum()
            print(f"Stocks with zero average assets: {zero_assets}")

        # Convert to factor frame
        ft_ivc = self.as_factor_frame(df_ivc)
        print(f"Factor frame shape: {ft_ivc.shape}")
        print(f"Factor frame sample:\n{ft_ivc.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class OaFactor(A3InvestmentFactorBase):
    """
    A.3.16 Oa - Operating Accruals

    Operating accruals measure the difference between accounting earnings
    and cash flows from operations. Two methods are used depending on data availability:

    Pre-1988 (Balance Sheet Approach):
        Oa = (ΔCA - ΔCASH) - (ΔCL - ΔSTD - ΔTP) - DP

    Post-1988 (Cash Flow Approach):
        Oa = (Net Income - Operating Cash Flow) / Lagged Total Assets

    Data Sources:
        - Balance sheet: total_cur_assets, money_cap, total_cur_liab, taxes_payable
        - Cash flow: depr_fa_coga_dpba1_fillna (depreciation)
        - Income statement: n_income, quarterly_n_cashflow_act_fillna

    Implementation:
        - Uses the cash flow approach (post-1988 method) as primary
        - Scaled by 1-year lagged total assets
        - Annual fiscal year-end data (December)
    """

    @property
    def factor_id(self) -> str:
        return "A.3.16"

    @property
    def abbr(self) -> str:
        return "oa"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate operating accruals factor using cash flow approach.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'oa']
        """
        # Load required data from multiple sources
        # Net income from income statement
        df_income = load_income_data(['quarterly_n_income_fillna'], RESSET_DIR)
        df_income = df_income.rename(columns={'quarterly_n_income_fillna': 'n_income'})

        # Operating cash flow from cash flow statement
        df_cashflow = load_cashflow_data(['quarterly_n_cashflow_act_fillna'], RESSET_DIR)
        df_cashflow = df_cashflow.rename(columns={'quarterly_n_cashflow_act_fillna': 'operating_cf'})

        # Total assets from balance sheet (for scaling)
        df_assets = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        df_assets = df_assets.rename(columns={'quarterly_total_assets_fillna': 'total_assets'})

        # Merge all data
        df = pd.merge(df_income[['code', 'end_date', 'n_income']],
                      df_cashflow[['code', 'end_date', 'operating_cf']],
                      on=['code', 'end_date'], how='inner')
        df = pd.merge(df, df_assets[['code', 'end_date', 'total_assets']],
                      on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Calculate lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['total_assets'].shift(1)

        # Calculate operating accruals (cash flow approach)
        df['oa'] = (df['n_income'] - df['operating_cf']) / df['total_assets_lag1']

        return df[['code', 'end_date', 'oa']].dropna()

    def validate(self) -> bool:
        """Validate Oa factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_oa = self.calculate()
        print(f"Shape: {df_oa.shape}")
        print(f"Non-NaN values: {df_oa['oa'].notna().sum()}")
        print(f"Statistics:\n{df_oa['oa'].describe()}")
        print(f"Sample:\n{df_oa.head(10)}")

        # Check operating accruals calculation
        sample_stock = df_oa['code'].iloc[0] if len(df_oa) > 0 else None
        if sample_stock:
            sample_data = df_oa[df_oa['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} operating accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_oa['oa'] > 1).sum()
        extreme_low = (df_oa['oa'] < -1).sum()
        infinite_values = np.isinf(df_oa['oa']).sum()
        print(f"Extreme high OA (>100%): {extreme_high}")
        print(f"Extreme negative OA (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_oa['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data merge quality
        required_cols = ['n_income', 'operating_cf', 'total_assets']
        missing_cols = [col for col in required_cols if col not in df_oa.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
        else:
            zero_income = (df_oa['n_income'] == 0).sum()
            zero_cf = (df_oa['operating_cf'] == 0).sum()
            print(f"Stocks with zero net income: {zero_income}")
            print(f"Stocks with zero operating cash flow: {zero_cf}")

        # Convert to factor frame
        ft_oa = self.as_factor_frame(df_oa)
        print(f"Factor frame shape: {ft_oa.shape}")
        print(f"Factor frame sample:\n{ft_oa.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class TaFactor(A3InvestmentFactorBase):
    """
    A.3.17 Ta - Total Accruals

    Total accruals measured using the balance sheet approach (Richardson et al., 2005).

    Formula:
        Ta = dWc + dNco + dFin

    Where:
        - dWc: change in net noncash working capital
        - dNco: change in net noncurrent operating assets
        - dFin: change in net financial assets

    Components:
        - Coa = Current Assets - Cash & Short-term Investments
        - Col = Current Liabilities - Debt in Current Liabilities
        - Nca = Total Assets - Current Assets - Long-term Investments
        - Ncl = Total Liabilities - Current Liabilities - Long-term Debt
        - Fna = Short-term Investments + Long-term Investments
        - Fnl = Long-term Debt + Debt in Current Liabilities + Preferred Stocks

    Data Source:
        - Balance Sheet: quarterly_balancesheet_cleaned.csv

    Implementation:
        - Annual fiscal year-end data (December only)
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.17"

    @property
    def abbr(self) -> str:
        return "Ta"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate total accruals using balance sheet approach.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ta']
        """
        # Load required balance sheet columns (same as in document 1)
        cols = [
            'quarterly_total_cur_assets_fillna', 'quarterly_money_cap_fillna',
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna', 'quarterly_total_cur_liab_fillna',
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_total_assets_fillna', 'quarterly_htm_invest_fillna',
            'quarterly_fa_avail_for_sale_fillna', 'quarterly_lt_eqt_invest_fillna',
            'quarterly_invest_real_estate_fillna', 'quarterly_lt_rec_fillna',
            'quarterly_total_liab_fillna', 'quarterly_lt_borr_fillna',
            'quarterly_lt_payable_fillna', 'quarterly_bond_payable_fillna',
            'quarterly_oth_eqt_tools_p_shr_fillna', 'quarterly_total_hldr_eqy_inc_min_int_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation (as in document 1)
        df = df.fillna(0)

        # Calculate components following document 1 logic
        # Cash and short-term investment
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

        # Short-term investments
        df['short_term_inv'] = df['cash_st_inv'] - df['quarterly_money_cap_fillna']

        # Long-term investments
        df['long_term_inv'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        # Long-term debt
        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        # Debt in current liabilities
        df['debt_curr_liab'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        # Calculate accounting components
        # Current operating assets
        df['Coa'] = df['quarterly_total_cur_assets_fillna'] - df['cash_st_inv']

        # Current operating liabilities
        df['Col'] = df['quarterly_total_cur_liab_fillna'] - df['debt_curr_liab']

        # Noncurrent operating assets
        df['Nca'] = (
                df['quarterly_total_assets_fillna'] -
                df['quarterly_total_cur_assets_fillna'] -
                df['long_term_inv']
        )

        # Noncurrent operating liabilities
        df['Ncl'] = (
                df['quarterly_total_liab_fillna'] -
                df['quarterly_total_cur_liab_fillna'] -
                df['long_term_debt']
        )

        # Financial assets
        df['Fna'] = df['short_term_inv'] + df['long_term_inv']

        # Financial liabilities
        df['Fnl'] = df['long_term_debt'] + df['debt_curr_liab'] + df['quarterly_oth_eqt_tools_p_shr_fillna']

        # Calculate changes (year-over-year)
        df['dWc'] = (df['Coa'] - df['Col']) - df.groupby('code')['Coa'].shift(1) + df.groupby('code')['Col'].shift(1)
        df['dNco'] = (df['Nca'] - df['Ncl']) - df.groupby('code')['Nca'].shift(1) + df.groupby('code')['Ncl'].shift(1)
        df['dFin'] = (df['Fna'] - df['Fnl']) - df.groupby('code')['Fna'].shift(1) + df.groupby('code')['Fnl'].shift(1)

        # Calculate total accruals
        df['Ta'] = df['dWc'] + df['dNco'] + df['dFin']

        # Scale by 1-year lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df = df[df['total_assets_lag1'] > 0]
        df['Ta'] = df['Ta'] / df['total_assets_lag1']

        return df[['code', 'end_date', 'Ta']].dropna()

    def validate(self) -> bool:
        """Validate Ta factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ta = self.calculate()
        print(f"Shape: {df_ta.shape}")
        print(f"Non-NaN values: {df_ta['Ta'].notna().sum()}")
        print(f"Statistics:\n{df_ta['Ta'].describe()}")
        print(f"Sample:\n{df_ta.head(10)}")

        # Check total accruals calculation
        sample_stock = df_ta['code'].iloc[0] if len(df_ta) > 0 else None
        if sample_stock:
            sample_data = df_ta[df_ta['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_ta['Ta'] > 1).sum()
        extreme_low = (df_ta['Ta'] < -1).sum()
        infinite_values = np.isinf(df_ta['Ta']).sum()
        print(f"Extreme high Ta (>100%): {extreme_high}")
        print(f"Extreme negative Ta (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_ta['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_ta['Ta'].isna()).sum()
        print(f"Missing Ta values: {zero_assets}")

        # Convert to factor frame
        ft_ta = self.as_factor_frame(df_ta)
        print(f"Factor frame shape: {ft_ta.shape}")
        print(f"Factor frame sample:\n{ft_ta.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DWcFactor(A3InvestmentFactorBase):
    """
    A.3.18 dWc - Changes in Net Non-cash Working Capital

    Change in net noncash working capital, which is current operating assets
    minus current operating liabilities.

    Formula:
        dWc = Δ(Coa - Col)

    Where:
        - Coa = Current Assets - Cash & Short-term Investments
        - Col = Current Liabilities - Debt in Current Liabilities

    Data Source:
        - Balance Sheet: quarterly_balancesheet_cleaned.csv

    Implementation:
        - Annual fiscal year-end data (December only)
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.18"

    @property
    def abbr(self) -> str:
        return "dWc"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in net non-cash working capital.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dWc']
        """
        # Load required balance sheet columns
        cols = [
            'quarterly_total_cur_assets_fillna', 'quarterly_money_cap_fillna',
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna', 'quarterly_total_cur_liab_fillna',
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_total_assets_fillna'
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

        # Debt in current liabilities
        df['debt_curr_liab'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        # Calculate current operating assets and liabilities
        df['Coa'] = df['quarterly_total_cur_assets_fillna'] - df['cash_st_inv']
        df['Col'] = df['quarterly_total_cur_liab_fillna'] - df['debt_curr_liab']

        # Calculate net working capital and its change
        df['Wc'] = df['Coa'] - df['Col']
        df['Wc_lag1'] = df.groupby('code')['Wc'].shift(1)
        df['dWc'] = df['Wc'] - df['Wc_lag1']

        # Scale by 1-year lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df = df[df['total_assets_lag1'] > 0]
        df['dWc'] = df['dWc'] / df['total_assets_lag1']

        return df[['code', 'end_date', 'dWc']].dropna()

    def validate(self) -> bool:
        """Validate dWc factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dWc = self.calculate()
        print(f"Shape: {df_dWc.shape}")
        print(f"Non-NaN values: {df_dWc['dWc'].notna().sum()}")
        print(f"SdWctistics:\n{df_dWc['dWc'].describe()}")
        print(f"Sample:\n{df_dWc.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dWc['code'].iloc[0] if len(df_dWc) > 0 else None
        if sample_stock:
            sample_data = df_dWc[df_dWc['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dWc['dWc'] > 1).sum()
        extreme_low = (df_dWc['dWc'] < -1).sum()
        infinite_values = np.isinf(df_dWc['dWc']).sum()
        print(f"Extreme high dWc (>100%): {extreme_high}")
        print(f"Extreme negative dWc (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dWc['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dWc['dWc'].isna()).sum()
        print(f"Missing dWc values: {zero_assets}")

        # Convert to factor frame
        ft_dWc = self.as_factor_frame(df_dWc)
        print(f"Factor frame shape: {ft_dWc.shape}")
        print(f"Factor frame sample:\n{ft_dWc.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DCoaFactor(A3InvestmentFactorBase):
    """
    A.3.18 dCoa - Changes in Current Operating Assets

    Change in current operating assets.

    Formula:
        dCoa = ΔCoa

    Where:
        - Coa = Current Assets - Cash & Short-term Investments

    Data Source:
        - Balance Sheet: quarterly_balancesheet_cleaned.csv

    Implementation:
        - Annual fiscal year-end data (December only)
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.18"

    @property
    def abbr(self) -> str:
        return "dCoa"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in current operating assets.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dCoa']
        """
        # Load required balance sheet columns
        cols = [
            'quarterly_total_cur_assets_fillna', 'quarterly_money_cap_fillna',
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna', 'quarterly_total_assets_fillna'
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

        # Calculate current operating assets
        df['Coa'] = df['quarterly_total_cur_assets_fillna'] - df['cash_st_inv']

        # Calculate change in current operating assets
        df['Coa_lag1'] = df.groupby('code')['Coa'].shift(1)
        df['dCoa'] = df['Coa'] - df['Coa_lag1']

        # Scale by 1-year lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df = df[df['total_assets_lag1'] > 0]
        df['dCoa'] = df['dCoa'] / df['total_assets_lag1']

        return df[['code', 'end_date', 'dCoa']].dropna()

    def validate(self) -> bool:
        """Validate dCoa factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dCoa = self.calculate()
        print(f"Shape: {df_dCoa.shape}")
        print(f"Non-NaN values: {df_dCoa['dCoa'].notna().sum()}")
        print(f"SdCoatistics:\n{df_dCoa['dCoa'].describe()}")
        print(f"Sample:\n{df_dCoa.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dCoa['code'].iloc[0] if len(df_dCoa) > 0 else None
        if sample_stock:
            sample_data = df_dCoa[df_dCoa['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dCoa['dCoa'] > 1).sum()
        extreme_low = (df_dCoa['dCoa'] < -1).sum()
        infinite_values = np.isinf(df_dCoa['dCoa']).sum()
        print(f"Extreme high dCoa (>100%): {extreme_high}")
        print(f"Extreme negative dCoa (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dCoa['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dCoa['dCoa'].isna()).sum()
        print(f"Missing dCoa values: {zero_assets}")

        # Convert to factor frame
        ft_dCoa = self.as_factor_frame(df_dCoa)
        print(f"Factor frame shape: {ft_dCoa.shape}")
        print(f"Factor frame sample:\n{ft_dCoa.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DColFactor(A3InvestmentFactorBase):
    """
    A.3.18 dCol - Changes in Current Operating Liabilities

    Change in current operating liabilities.

    Formula:
        dCol = ΔCol

    Where:
        - Col = Current Liabilities - Debt in Current Liabilities

    Data Source:
        - Balance Sheet: quarterly_balancesheet_cleaned.csv

    Implementation:
        - Annual fiscal year-end data (December only)
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.18"

    @property
    def abbr(self) -> str:
        return "dCol"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in current operating liabilities.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dCol']
        """
        # Load required balance sheet columns
        cols = [
            'quarterly_total_cur_liab_fillna', 'quarterly_st_borr_fillna',
            'quarterly_notes_payable_fillna', 'quarterly_st_bonds_payable_fillna',
            'quarterly_non_cur_liab_due_1y_fillna', 'quarterly_total_assets_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate debt in current liabilities
        df['debt_curr_liab'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        # Calculate current operating liabilities
        df['Col'] = df['quarterly_total_cur_liab_fillna'] - df['debt_curr_liab']

        # Calculate change in current operating liabilities
        df['Col_lag1'] = df.groupby('code')['Col'].shift(1)
        df['dCol'] = df['Col'] - df['Col_lag1']

        # Scale by 1-year lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df = df[df['total_assets_lag1'] > 0]
        df['dCol'] = df['dCol'] / df['total_assets_lag1']

        return df[['code', 'end_date', 'dCol']].dropna()

    def validate(self) -> bool:
        """Validate dCol factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dCol = self.calculate()
        print(f"Shape: {df_dCol.shape}")
        print(f"Non-NaN values: {df_dCol['dCol'].notna().sum()}")
        print(f"SdColtistics:\n{df_dCol['dCol'].describe()}")
        print(f"Sample:\n{df_dCol.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dCol['code'].iloc[0] if len(df_dCol) > 0 else None
        if sample_stock:
            sample_data = df_dCol[df_dCol['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dCol['dCol'] > 1).sum()
        extreme_low = (df_dCol['dCol'] < -1).sum()
        infinite_values = np.isinf(df_dCol['dCol']).sum()
        print(f"Extreme high dCol (>100%): {extreme_high}")
        print(f"Extreme negative dCol (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dCol['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dCol['dCol'].isna()).sum()
        print(f"Missing dCol values: {zero_assets}")

        # Convert to factor frame
        ft_dCol = self.as_factor_frame(df_dCol)
        print(f"Factor frame shape: {ft_dCol.shape}")
        print(f"Factor frame sample:\n{ft_dCol.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DNcoFactor(A3InvestmentFactorBase):
    """
    A.3.19 dNco - Changes in Net Non-current Operating Assets

    Change in net non-current operating assets, which are non-current operating
    assets minus non-current operating liabilities.

    Formula:
        dNco = Δ(Nca - Ncl)

    Where:
        - Nca = Total Assets - Current Assets - Long-term Investments
        - Ncl = Total Liabilities - Current Liabilities - Long-term Debt

    Data Source:
        - Balance Sheet: quarterly_balancesheet_cleaned.csv

    Implementation:
        - Annual fiscal year-end data (December only)
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.19"

    @property
    def abbr(self) -> str:
        return "dNco"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in net non-current operating assets.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dNco']
        """
        # Load required balance sheet columns
        cols = [
            'quarterly_total_assets_fillna', 'quarterly_total_cur_assets_fillna',
            'quarterly_htm_invest_fillna', 'quarterly_fa_avail_for_sale_fillna',
            'quarterly_lt_eqt_invest_fillna', 'quarterly_invest_real_estate_fillna',
            'quarterly_lt_rec_fillna', 'quarterly_total_liab_fillna',
            'quarterly_total_cur_liab_fillna', 'quarterly_lt_borr_fillna',
            'quarterly_lt_payable_fillna', 'quarterly_bond_payable_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate long-term investments
        df['long_term_inv'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        # Calculate long-term debt
        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        # Calculate non-current operating assets and liabilities
        df['Nca'] = (
                df['quarterly_total_assets_fillna'] -
                df['quarterly_total_cur_assets_fillna'] -
                df['long_term_inv']
        )

        df['Ncl'] = (
                df['quarterly_total_liab_fillna'] -
                df['quarterly_total_cur_liab_fillna'] -
                df['long_term_debt']
        )

        # Calculate net non-current operating assets and its change
        df['Nco'] = df['Nca'] - df['Ncl']
        df['Nco_lag1'] = df.groupby('code')['Nco'].shift(1)
        df['dNco'] = df['Nco'] - df['Nco_lag1']

        # Scale by 1-year lagged total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df = df[df['total_assets_lag1'] > 0]
        df['dNco'] = df['dNco'] / df['total_assets_lag1']

        return df[['code', 'end_date', 'dNco']].dropna()

    def validate(self) -> bool:
        """Validate dNco factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dNco = self.calculate()
        print(f"Shape: {df_dNco.shape}")
        print(f"Non-NaN values: {df_dNco['dNco'].notna().sum()}")
        print(f"SdNcotistics:\n{df_dNco['dNco'].describe()}")
        print(f"Sample:\n{df_dNco.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dNco['code'].iloc[0] if len(df_dNco) > 0 else None
        if sample_stock:
            sample_data = df_dNco[df_dNco['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dNco['dNco'] > 1).sum()
        extreme_low = (df_dNco['dNco'] < -1).sum()
        infinite_values = np.isinf(df_dNco['dNco']).sum()
        print(f"Extreme high dNco (>100%): {extreme_high}")
        print(f"Extreme negative dNco (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dNco['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dNco['dNco'].isna()).sum()
        print(f"Missing dNco values: {zero_assets}")

        # Convert to factor frame
        ft_dNco = self.as_factor_frame(df_dNco)
        print(f"Factor frame shape: {ft_dNco.shape}")
        print(f"Factor frame sample:\n{ft_dNco.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DncaFactor(A3InvestmentFactorBase):
    """
    A.3.19 dNca - Changes in Non-current Operating Assets

    dNca is the change in noncurrent operating assets.
    Noncurrent operating assets are total assets minus current assets
    minus long-term investments.

    Formula:
        dNca = Nca_t - Nca_{t-1}
        where Nca = total_assets - total_cur_assets - long_term_investments

    Scaled by total assets for the fiscal year ending in calendar year t-2.
    Annual fiscal year-end data (December).

    Data Source:
        - Balance Sheet: total_assets, total_cur_assets, long_term_investments
    """

    @property
    def factor_id(self) -> str:
        return "A.3.19"

    @property
    def abbr(self) -> str:
        return "dNca"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in non-current operating assets.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dNca']
        """
        # Load required balance sheet columns (same as in document1)
        cols = [
            'quarterly_total_assets_fillna',
            'quarterly_total_cur_assets_fillna',
            'quarterly_htm_invest_fillna', 'quarterly_fa_avail_for_sale_fillna',
            'quarterly_lt_eqt_invest_fillna', 'quarterly_invest_real_estate_fillna',
            'quarterly_lt_rec_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate long-term investments (same calculation as document1)
        df['long_term_investments'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        # Calculate Nca (noncurrent operating assets)
        df['Nca'] = (
                df['quarterly_total_assets_fillna'] -
                df['quarterly_total_cur_assets_fillna'] -
                df['long_term_investments']
        )

        # Calculate dNca (change in Nca)
        df['Nca_lag1'] = df.groupby('code')['Nca'].shift(1)
        df['dNca'] = df['Nca'] - df['Nca_lag1']

        # Scale by lagged total assets (t-2)
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df = df[df['total_assets_lag2'] > 0]
        df['dNca'] = df['dNca'] / df['total_assets_lag2']

        return df[['code', 'end_date', 'dNca']].dropna()

    def validate(self) -> bool:
        """Validate dNca factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dNca = self.calculate()
        print(f"Shape: {df_dNca.shape}")
        print(f"Non-NaN values: {df_dNca['dNca'].notna().sum()}")
        print(f"SdNcatistics:\n{df_dNca['dNca'].describe()}")
        print(f"Sample:\n{df_dNca.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dNca['code'].iloc[0] if len(df_dNca) > 0 else None
        if sample_stock:
            sample_data = df_dNca[df_dNca['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dNca['dNca'] > 1).sum()
        extreme_low = (df_dNca['dNca'] < -1).sum()
        infinite_values = np.isinf(df_dNca['dNca']).sum()
        print(f"Extreme high dNca (>100%): {extreme_high}")
        print(f"Extreme negative dNca (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dNca['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dNca['dNca'].isna()).sum()
        print(f"Missing dNca values: {zero_assets}")

        # Convert to factor frame
        ft_dNca = self.as_factor_frame(df_dNca)
        print(f"Factor frame shape: {ft_dNca.shape}")
        print(f"Factor frame sample:\n{ft_dNca.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DnclFactor(A3InvestmentFactorBase):
    """
    A.3.19 dNcl - Changes in Non-current Operating Liabilities

    dNcl is the change in noncurrent operating liabilities.
    Noncurrent operating liabilities are total liabilities minus current liabilities
    minus long-term debt.

    Formula:
        dNcl = Ncl_t - Ncl_{t-1}
        where Ncl = total_liab - total_cur_liab - long_term_debt

    Scaled by total assets for the fiscal year ending in calendar year t-2.
    Annual fiscal year-end data (December).

    Data Source:
        - Balance Sheet: total_liab, total_cur_liab, long_term_debt
    """

    @property
    def factor_id(self) -> str:
        return "A.3.19"

    @property
    def abbr(self) -> str:
        return "dNcl"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in non-current operating liabilities.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dNcl']
        """
        # Load required balance sheet columns (same as in document1)
        cols = [
            'quarterly_total_liab_fillna',
            'quarterly_total_cur_liab_fillna',
            'quarterly_lt_borr_fillna', 'quarterly_lt_payable_fillna',
            'quarterly_bond_payable_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate long-term debt (same calculation as document1)
        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        # Calculate Ncl (noncurrent operating liabilities)
        df['Ncl'] = (
                df['quarterly_total_liab_fillna'] -
                df['quarterly_total_cur_liab_fillna'] -
                df['long_term_debt']
        )

        # Calculate dNcl (change in Ncl)
        df['Ncl_lag1'] = df.groupby('code')['Ncl'].shift(1)
        df['dNcl'] = df['Ncl'] - df['Ncl_lag1']

        # Scale by lagged total assets (t-2)
        # Need to load total assets for scaling
        df_assets = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        df_assets = df_assets[pd.to_datetime(df_assets['end_date']).dt.month == 12].copy()
        df_assets = df_assets[['code', 'end_date', 'quarterly_total_assets_fillna']]

        df = pd.merge(df, df_assets, on=['code', 'end_date'], how='left')
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df = df[df['total_assets_lag2'] > 0]
        df['dNcl'] = df['dNcl'] / df['total_assets_lag2']

        return df[['code', 'end_date', 'dNcl']].dropna()

    def validate(self) -> bool:
        """Validate dNcl factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dNcl = self.calculate()
        print(f"Shape: {df_dNcl.shape}")
        print(f"Non-NaN values: {df_dNcl['dNcl'].notna().sum()}")
        print(f"SdNcltistics:\n{df_dNcl['dNcl'].describe()}")
        print(f"Sample:\n{df_dNcl.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dNcl['code'].iloc[0] if len(df_dNcl) > 0 else None
        if sample_stock:
            sample_data = df_dNcl[df_dNcl['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dNcl['dNcl'] > 1).sum()
        extreme_low = (df_dNcl['dNcl'] < -1).sum()
        infinite_values = np.isinf(df_dNcl['dNcl']).sum()
        print(f"Extreme high dNcl (>100%): {extreme_high}")
        print(f"Extreme negative dNcl (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dNcl['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dNcl['dNcl'].isna()).sum()
        print(f"Missing dNcl values: {zero_assets}")

        # Convert to factor frame
        ft_dNcl = self.as_factor_frame(df_dNcl)
        print(f"Factor frame shape: {ft_dNcl.shape}")
        print(f"Factor frame sample:\n{ft_dNcl.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DfinFactor(A3InvestmentFactorBase):
    """
    A.3.20 dFin - Changes in Net Financial Assets

    dFin is the change in net financial assets.
    Net financial assets are financial assets minus financial liabilities.

    Formula:
        dFin = Fin_t - Fin_{t-1}
        where Fin = (short_term_investments + long_term_investments) -
                   (long_term_debt + current_debt + preferred_stock)

    Scaled by total assets for the fiscal year ending in calendar year t-2.
    Annual fiscal year-end data (December).

    Data Source:
        - Balance Sheet: short_term_investments, long_term_investments,
                       long_term_debt, current_debt, preferred_stock
    """

    @property
    def factor_id(self) -> str:
        return "A.3.20"

    @property
    def abbr(self) -> str:
        return "dFin"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in net financial assets.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dFin']
        """
        # Load required balance sheet columns (same as in document1)
        cols = [
            # Financial assets components
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna', 'quarterly_htm_invest_fillna',
            'quarterly_fa_avail_for_sale_fillna', 'quarterly_lt_eqt_invest_fillna',
            'quarterly_invest_real_estate_fillna', 'quarterly_lt_rec_fillna',
            # Financial liabilities components
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_lt_borr_fillna', 'quarterly_lt_payable_fillna',
            'quarterly_bond_payable_fillna', 'quarterly_oth_eqt_tools_p_shr_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate financial assets (Fna) - same as document1
        df['short_term_investments'] = (
                df['quarterly_trad_asset_fillna'] +
                df['quarterly_accounts_receiv_bill_fillna'] +
                df['quarterly_int_receiv_fillna'] +
                df['quarterly_pur_resale_fa_fillna'] +
                df['quarterly_loanto_oth_bank_fi_fillna'] +
                df['quarterly_time_deposits_fillna'] +
                df['quarterly_refund_depos_fillna']
        )

        df['long_term_investments'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        df['financial_assets'] = df['short_term_investments'] + df['long_term_investments']

        # Calculate financial liabilities (Fnl) - same as document1
        df['current_debt'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        df['preferred_stock'] = df['quarterly_oth_eqt_tools_p_shr_fillna']

        df['financial_liabilities'] = (
                df['current_debt'] + df['long_term_debt'] + df['preferred_stock']
        )

        # Calculate net financial assets (Fin)
        df['Fin'] = df['financial_assets'] - df['financial_liabilities']

        # Calculate dFin (change in Fin)
        df['Fin_lag1'] = df.groupby('code')['Fin'].shift(1)
        df['dFin'] = df['Fin'] - df['Fin_lag1']

        # Scale by lagged total assets (t-2)
        df_assets = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        df_assets = df_assets[pd.to_datetime(df_assets['end_date']).dt.month == 12].copy()
        df_assets = df_assets[['code', 'end_date', 'quarterly_total_assets_fillna']]

        df = pd.merge(df, df_assets, on=['code', 'end_date'], how='left')
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df['dFin'] = df['dFin'] / df['total_assets_lag2']

        return df[['code', 'end_date', 'dFin']].dropna()

    def validate(self) -> bool:
        """Validate dFin factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dFin = self.calculate()
        print(f"Shape: {df_dFin.shape}")
        print(f"Non-NaN values: {df_dFin['dFin'].notna().sum()}")
        print(f"SdFintistics:\n{df_dFin['dFin'].describe()}")
        print(f"Sample:\n{df_dFin.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dFin['code'].iloc[0] if len(df_dFin) > 0 else None
        if sample_stock:
            sample_data = df_dFin[df_dFin['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dFin['dFin'] > 1).sum()
        extreme_low = (df_dFin['dFin'] < -1).sum()
        infinite_values = np.isinf(df_dFin['dFin']).sum()
        print(f"Extreme high dFin (>100%): {extreme_high}")
        print(f"Extreme negative dFin (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dFin['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dFin['dFin'].isna()).sum()
        print(f"Missing dFin values: {zero_assets}")

        # Convert to factor frame
        ft_dFin = self.as_factor_frame(df_dFin)
        print(f"Factor frame shape: {ft_dFin.shape}")
        print(f"Factor frame sample:\n{ft_dFin.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DstiFactor(A3InvestmentFactorBase):
    """
    A.3.20 dSti - Changes in Short-term Investments

    dSti is the change in short-term investments.
    Short-term investments include trading assets, accounts receivable bills,
    interest receivable, purchase resale financial assets, loans to other banks,
    time deposits, and refund deposits.

    Formula:
        dSti = Sti_t - Sti_{t-1}

    Scaled by total assets for the fiscal year ending in calendar year t-2.
    Annual fiscal year-end data (December).
    Firms that do not have short-term investments in the past two fiscal years are excluded.

    Data Source:
        - Balance Sheet: trad_asset, accounts_receiv_bill, int_receiv, pur_resale_fa,
                        loanto_oth_bank_fi, time_deposits, refund_depos
    """

    @property
    def factor_id(self) -> str:
        return "A.3.20"

    @property
    def abbr(self) -> str:
        return "dSti"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in short-term investments.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dSti']
        """
        # Load required balance sheet columns for short-term investments
        cols = [
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate short-term investments (same calculation as document1)
        df['short_term_investments'] = (
                df['quarterly_trad_asset_fillna'] +
                df['quarterly_accounts_receiv_bill_fillna'] +
                df['quarterly_int_receiv_fillna'] +
                df['quarterly_pur_resale_fa_fillna'] +
                df['quarterly_loanto_oth_bank_fi_fillna'] +
                df['quarterly_time_deposits_fillna'] +
                df['quarterly_refund_depos_fillna']
        )

        # Calculate dSti (change in short-term investments)
        df['short_term_investments_lag1'] = df.groupby('code')['short_term_investments'].shift(1)
        df['dSti'] = df['short_term_investments'] - df['short_term_investments_lag1']

        # Scale by lagged total assets (t-2)
        df_assets = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        df_assets = df_assets[pd.to_datetime(df_assets['end_date']).dt.month == 12].copy()
        df_assets = df_assets[['code', 'end_date', 'quarterly_total_assets_fillna']]

        df = pd.merge(df, df_assets, on=['code', 'end_date'], how='left')
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df['dSti'] = df['dSti'] / df['total_assets_lag2']

        # Exclude firms that do not have short-term investments in the past two years
        df = df[(df['short_term_investments'] > 0) |
                (df['short_term_investments_lag1'] > 0)]

        return df[['code', 'end_date', 'dSti']].dropna()

    def validate(self) -> bool:
        """Validate dSti factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dSti = self.calculate()
        print(f"Shape: {df_dSti.shape}")
        print(f"Non-NaN values: {df_dSti['dSti'].notna().sum()}")
        print(f"SdStitistics:\n{df_dSti['dSti'].describe()}")
        print(f"Sample:\n{df_dSti.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dSti['code'].iloc[0] if len(df_dSti) > 0 else None
        if sample_stock:
            sample_data = df_dSti[df_dSti['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dSti['dSti'] > 1).sum()
        extreme_low = (df_dSti['dSti'] < -1).sum()
        infinite_values = np.isinf(df_dSti['dSti']).sum()
        print(f"Extreme high dSti (>100%): {extreme_high}")
        print(f"Extreme negative dSti (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dSti['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dSti['dSti'].isna()).sum()
        print(f"Missing dSti values: {zero_assets}")

        # Convert to factor frame
        ft_dSti = self.as_factor_frame(df_dSti)
        print(f"Factor frame shape: {ft_dSti.shape}")
        print(f"Factor frame sample:\n{ft_dSti.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DltiFactor(A3InvestmentFactorBase):
    """
    A.3.20 dLti - Changes in Long-term Investments

    dLti is the change in long-term investments.
    Long-term investments include held-to-maturity investments, available-for-sale
    financial assets, long-term equity investments, investment real estate,
    and long-term receivables.

    Formula:
        dLti = Lti_t - Lti_{t-1}

    Scaled by total assets for the fiscal year ending in calendar year t-2.
    Annual fiscal year-end data (December).
    Firms that do not have long-term investments in the past two fiscal years are excluded.

    Data Source:
        - Balance Sheet: htm_invest, fa_avail_for_sale, lt_eqt_invest,
                        invest_real_estate, lt_rec
    """

    @property
    def factor_id(self) -> str:
        return "A.3.20"

    @property
    def abbr(self) -> str:
        return "dLti"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in long-term investments.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dLti']
        """
        # Load required balance sheet columns for long-term investments
        cols = [
            'quarterly_htm_invest_fillna', 'quarterly_fa_avail_for_sale_fillna',
            'quarterly_lt_eqt_invest_fillna', 'quarterly_invest_real_estate_fillna',
            'quarterly_lt_rec_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate long-term investments (same calculation as document1)
        df['long_term_investments'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        # Calculate dLti (change in long-term investments)
        df['long_term_investments_lag1'] = df.groupby('code')['long_term_investments'].shift(1)
        df['dLti'] = df['long_term_investments'] - df['long_term_investments_lag1']

        # Scale by lagged total assets (t-2)
        df_assets = load_balancesheet_data(['quarterly_total_assets_fillna'], RESSET_DIR)
        df_assets = df_assets[pd.to_datetime(df_assets['end_date']).dt.month == 12].copy()
        df_assets = df_assets[['code', 'end_date', 'quarterly_total_assets_fillna']]

        df = pd.merge(df, df_assets, on=['code', 'end_date'], how='left')
        df['total_assets_lag2'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(2)
        df['dLti'] = df['dLti'] / df['total_assets_lag2']

        # Exclude firms that do not have long-term investments in the past two years
        df = df[(df['long_term_investments'] > 0) |
                (df['long_term_investments_lag1'] > 0)]

        return df[['code', 'end_date', 'dLti']].dropna()

    def validate(self) -> bool:
        """Validate dLti factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dLti = self.calculate()
        print(f"Shape: {df_dLti.shape}")
        print(f"Non-NaN values: {df_dLti['dLti'].notna().sum()}")
        print(f"SdLtitistics:\n{df_dLti['dLti'].describe()}")
        print(f"Sample:\n{df_dLti.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dLti['code'].iloc[0] if len(df_dLti) > 0 else None
        if sample_stock:
            sample_data = df_dLti[df_dLti['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dLti['dLti'] > 1).sum()
        extreme_low = (df_dLti['dLti'] < -1).sum()
        infinite_values = np.isinf(df_dLti['dLti']).sum()
        print(f"Extreme high dLti (>100%): {extreme_high}")
        print(f"Extreme negative dLti (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dLti['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dLti['dLti'].isna()).sum()
        print(f"Missing dLti values: {zero_assets}")

        # Convert to factor frame
        ft_dLti = self.as_factor_frame(df_dLti)
        print(f"Factor frame shape: {ft_dLti.shape}")
        print(f"Factor frame sample:\n{ft_dLti.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DfnlFactor(A3InvestmentFactorBase):
    """
    A.3.20 dFnl - Changes in Financial Liabilities

    dFnl measures the annual change in financial liabilities scaled by
    lagged total assets. Financial liabilities include long-term debt,
    debt in current liabilities, and preferred stocks.

    Formula:
        dFnl = (Fnl_t - Fnl_{t-1}) / Total Assets_{t-1}
        where Fnl = DLTT + DLC + PSTK (preferred stock)

    Data Sources:
        - Balance Sheet: long-term debt, short-term debt, preferred stock
        - Uses annual fiscal year-end data (December)

    Implementation:
        - Financial liabilities = long-term debt + debt in current liabilities + preferred stock
        - Change calculated year-over-year
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.20"

    @property
    def abbr(self) -> str:
        return "dFnl"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in financial liabilities.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dFnl']
        """
        # Load required balance sheet columns
        cols = [
            'quarterly_lt_borr_fillna', 'quarterly_lt_payable_fillna',
            'quarterly_bond_payable_fillna', 'quarterly_st_borr_fillna',
            'quarterly_notes_payable_fillna', 'quarterly_st_bonds_payable_fillna',
            'quarterly_non_cur_liab_due_1y_fillna', 'quarterly_oth_eqt_tools_p_shr_fillna',
            'quarterly_total_assets_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate financial liabilities components
        # Long-term debt
        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        # Debt in current liabilities
        df['current_debt'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        # Preferred stock
        df['preferred_stock'] = df['quarterly_oth_eqt_tools_p_shr_fillna']

        # Total financial liabilities
        df['fnl'] = df['long_term_debt'] + df['current_debt'] + df['preferred_stock']

        # Calculate change in financial liabilities
        df['fnl_lag1'] = df.groupby('code')['fnl'].shift(1)
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)

        # Calculate dFnl
        df['dFnl'] = (df['fnl'] - df['fnl_lag1']) / df['total_assets_lag1']

        # Replace inf/-inf with NaN
        df['dFnl'] = df['dFnl'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'dFnl']].dropna()

    def validate(self) -> bool:
        """Validate dFnl factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dFnl = self.calculate()
        print(f"Shape: {df_dFnl.shape}")
        print(f"Non-NaN values: {df_dFnl['dFnl'].notna().sum()}")
        print(f"Statistics:\n{df_dFnl['dFnl'].describe()}")
        print(f"Sample:\n{df_dFnl.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dFnl['code'].iloc[0] if len(df_dFnl) > 0 else None
        if sample_stock:
            sample_data = df_dFnl[df_dFnl['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dFnl['dFnl'] > 1).sum()
        extreme_low = (df_dFnl['dFnl'] < -1).sum()
        infinite_values = np.isinf(df_dFnl['dFnl']).sum()
        print(f"Extreme high dFnl (>100%): {extreme_high}")
        print(f"Extreme negative dFnl (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dFnl['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dFnl['dFnl'].isna()).sum()
        print(f"Missing dFnl values: {zero_assets}")

        # Convert to factor frame
        ft_dFnl = self.as_factor_frame(df_dFnl)
        print(f"Factor frame shape: {ft_dFnl.shape}")
        print(f"Factor frame sample:\n{ft_dFnl.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DbeFactor(A3InvestmentFactorBase):
    """
    A.3.20 dBe - Changes in Book Equity

    dBe measures the annual change in book equity scaled by lagged total assets.
    Book equity represents the shareholders' equity in the company.

    Formula:
        dBe = (Book Equity_t - Book Equity_{t-1}) / Total Assets_{t-1}

    Data Sources:
        - Balance Sheet: total stockholders' equity (including minority interest)
        - Uses annual fiscal year-end data (December)

    Implementation:
        - Book equity = total stockholders' equity (including minority interest)
        - Change calculated year-over-year
        - Scaled by 1-year lagged total assets
    """

    @property
    def factor_id(self) -> str:
        return "A.3.20"

    @property
    def abbr(self) -> str:
        return "dBe"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate changes in book equity.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dBe']
        """
        # Load book equity and total assets data
        cols = [
            'quarterly_total_hldr_eqy_inc_min_int_fillna',
            'quarterly_total_assets_fillna'
        ]

        df = load_balancesheet_data(cols, RESSET_DIR)

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate book equity change
        df['book_equity_lag1'] = df.groupby('code')['quarterly_total_hldr_eqy_inc_min_int_fillna'].shift(1)
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)

        # Calculate dBe
        df['dBe'] = (
                (df['quarterly_total_hldr_eqy_inc_min_int_fillna'] - df['book_equity_lag1']) /
                df['total_assets_lag1']
        )

        # Replace inf/-inf with NaN
        df['dBe'] = df['dBe'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'dBe']].dropna()

    def validate(self) -> bool:
        """Validate dBe factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dBe = self.calculate()
        print(f"Shape: {df_dBe.shape}")
        print(f"Non-NaN values: {df_dBe['dBe'].notna().sum()}")
        print(f"Statistics:\n{df_dBe['dBe'].describe()}")
        print(f"Sample:\n{df_dBe.head(10)}")

        # Check total accruals calculation
        sample_stock = df_dBe['code'].iloc[0] if len(df_dBe) > 0 else None
        if sample_stock:
            sample_data = df_dBe[df_dBe['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_dBe['dBe'] > 1).sum()
        extreme_low = (df_dBe['dBe'] < -1).sum()
        infinite_values = np.isinf(df_dBe['dBe']).sum()
        print(f"Extreme high dBe (>100%): {extreme_high}")
        print(f"Extreme negative dBe (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_dBe['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_dBe['dBe'].isna()).sum()
        print(f"Missing dBe values: {zero_assets}")

        # Convert to factor frame
        ft_dBe = self.as_factor_frame(df_dBe)
        print(f"Factor frame shape: {ft_dBe.shape}")
        print(f"Factor frame sample:\n{ft_dBe.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class PoaFactor(A3InvestmentFactorBase):
    """
    A.3.22 Poa - Percent Operating Accruals

    Poa measures operating accruals scaled by the absolute value of net income.
    This provides an alternative scaling method to traditional asset-based scaling.

    Formula:
        Poa = Operating Accruals / |Net Income|

    Data Sources:
        - Balance Sheet: current assets, cash, current liabilities, debt, taxes payable
        - Cash Flow: depreciation and amortization
        - Income Statement: net income
        - Uses annual fiscal year-end data (December)

    Implementation:
        - Operating accruals calculated using balance sheet approach
        - Uses absolute value of net income as denominator
        - Similar to traditional operating accruals but with different scaling
    """

    @property
    def factor_id(self) -> str:
        return "A.3.22"

    @property
    def abbr(self) -> str:
        return "Poa"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate percent operating accruals.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Poa']
        """
        # Load balance sheet data for operating accruals calculation
        bs_cols = [
            'quarterly_total_cur_assets_fillna', 'quarterly_money_cap_fillna',
            'quarterly_total_cur_liab_fillna', 'quarterly_st_borr_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_taxes_payable_fillna'
        ]
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)

        # Load cash flow data for depreciation
        cf_cols = ['quarterly_depr_fa_coga_dpba1_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)

        # Load income data for net income
        income_cols = ['quarterly_n_income_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)

        # Merge all data
        df = pd.merge(df_bs, df_cf, on=['code', 'end_date'], how='inner')
        df = pd.merge(df, df_income, on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate operating accruals components (using 1-year lagged values)
        df = df.sort_values(['code', 'end_date'])

        # Calculate changes from previous year
        df['dCA'] = df['quarterly_total_cur_assets_fillna'] - df.groupby('code')[
            'quarterly_total_cur_assets_fillna'].shift(1)
        df['dCASH'] = df['quarterly_money_cap_fillna'] - df.groupby('code')['quarterly_money_cap_fillna'].shift(1)
        df['dCL'] = df['quarterly_total_cur_liab_fillna'] - df.groupby('code')['quarterly_total_cur_liab_fillna'].shift(
            1)

        # Calculate change in short-term debt
        df['dSTD'] = (
                (df['quarterly_st_borr_fillna'] + df['quarterly_st_bonds_payable_fillna'] +
                 df['quarterly_non_cur_liab_due_1y_fillna']) -
                (df.groupby('code')['quarterly_st_borr_fillna'].shift(1) +
                 df.groupby('code')['quarterly_st_bonds_payable_fillna'].shift(1) +
                 df.groupby('code')['quarterly_non_cur_liab_due_1y_fillna'].shift(1))
        )

        df['dTP'] = df['quarterly_taxes_payable_fillna'] - df.groupby('code')['quarterly_taxes_payable_fillna'].shift(1)

        # Calculate operating accruals
        df['operating_accruals'] = (
                (df['dCA'] - df['dCASH']) -
                (df['dCL'] - df['dSTD'] - df['dTP']) -
                df['quarterly_depr_fa_coga_dpba1_fillna']
        )

        # Calculate Poa = operating accruals / |net income|
        df['abs_net_income'] = df['quarterly_n_income_fillna'].abs()

        # Avoid division by zero
        df['Poa'] = np.where(
            df['abs_net_income'] > 0,
            df['operating_accruals'] / df['abs_net_income'],
            np.nan
        )

        # Replace inf/-inf with NaN
        df['Poa'] = df['Poa'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'Poa']].dropna()

    def validate(self) -> bool:
        """Validate Poa factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_Poa = self.calculate()
        print(f"Shape: {df_Poa.shape}")
        print(f"Non-NaN values: {df_Poa['Poa'].notna().sum()}")
        print(f"Statistics:\n{df_Poa['Poa'].describe()}")
        print(f"Sample:\n{df_Poa.head(10)}")

        # Check total accruals calculation
        sample_stock = df_Poa['code'].iloc[0] if len(df_Poa) > 0 else None
        if sample_stock:
            sample_data = df_Poa[df_Poa['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_Poa['Poa'] > 1).sum()
        extreme_low = (df_Poa['Poa'] < -1).sum()
        infinite_values = np.isinf(df_Poa['Poa']).sum()
        print(f"Extreme high Poa (>100%): {extreme_high}")
        print(f"Extreme negative Poa (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_Poa['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_Poa['Poa'].isna()).sum()
        print(f"Missing Poa values: {zero_assets}")

        # Convert to factor frame
        ft_Poa = self.as_factor_frame(df_Poa)
        print(f"Factor frame shape: {ft_Poa.shape}")
        print(f"Factor frame sample:\n{ft_Poa.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class PtaFactor(A3InvestmentFactorBase):
    """
    A.3.23 Pta - Percent Total Accruals

    Pta measures total accruals scaled by the absolute value of net income.
    This scaling method is more effective for selecting firms with extreme
    differences between sophisticated and naive earnings forecasts.

    Formula:
        Pta = Total Accruals / |Net Income|

    Data Sources:
        - Balance Sheet: various components for total accruals calculation
        - Income Statement: net income
        - Uses annual fiscal year-end data (December)

    Implementation:
        - Total accruals calculated using comprehensive approach
        - Uses absolute value of net income as denominator
        - Provides alternative scaling to traditional asset-based methods
    """

    @property
    def factor_id(self) -> str:
        return "A.3.23"

    @property
    def abbr(self) -> str:
        return "Pta"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate percent total accruals.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Pta']
        """
        # Load comprehensive balance sheet data for total accruals calculation
        cols = [
            'quarterly_total_cur_assets_fillna', 'quarterly_money_cap_fillna',
            'quarterly_trad_asset_fillna', 'quarterly_accounts_receiv_bill_fillna',
            'quarterly_int_receiv_fillna', 'quarterly_pur_resale_fa_fillna',
            'quarterly_loanto_oth_bank_fi_fillna', 'quarterly_time_deposits_fillna',
            'quarterly_refund_depos_fillna', 'quarterly_total_cur_liab_fillna',
            'quarterly_st_borr_fillna', 'quarterly_notes_payable_fillna',
            'quarterly_st_bonds_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
            'quarterly_total_assets_fillna', 'quarterly_htm_invest_fillna',
            'quarterly_fa_avail_for_sale_fillna', 'quarterly_lt_eqt_invest_fillna',
            'quarterly_invest_real_estate_fillna', 'quarterly_lt_rec_fillna',
            'quarterly_total_liab_fillna', 'quarterly_lt_borr_fillna',
            'quarterly_lt_payable_fillna', 'quarterly_bond_payable_fillna',
            'quarterly_oth_eqt_tools_p_shr_fillna', 'quarterly_total_hldr_eqy_inc_min_int_fillna'
        ]

        df_bs = load_balancesheet_data(cols, RESSET_DIR)

        # Load income data for net income
        income_cols = ['quarterly_n_income_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)

        # Merge data
        df = pd.merge(df_bs, df_income, on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df = df[pd.to_datetime(df['end_date']).dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NA with 0 for calculation
        df = df.fillna(0)

        # Calculate components for total accruals
        # Cash and short-term investments
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

        # Current operating assets (Coa)
        df['Coa'] = df['quarterly_total_cur_assets_fillna'] - df['cash_st_inv']

        # Debt in current liabilities (DLC)
        df['DLC'] = (
                df['quarterly_st_borr_fillna'] +
                df['quarterly_notes_payable_fillna'] +
                df['quarterly_st_bonds_payable_fillna'] +
                df['quarterly_non_cur_liab_due_1y_fillna']
        )

        # Current operating liabilities (Col)
        df['Col'] = df['quarterly_total_cur_liab_fillna'] - df['DLC']

        # Long-term investments
        df['long_term_inv'] = (
                df['quarterly_htm_invest_fillna'] +
                df['quarterly_fa_avail_for_sale_fillna'] +
                df['quarterly_lt_eqt_invest_fillna'] +
                df['quarterly_invest_real_estate_fillna'] +
                df['quarterly_lt_rec_fillna']
        )

        # Noncurrent operating assets (Nca)
        df['Nca'] = (
                df['quarterly_total_assets_fillna'] -
                df['quarterly_total_cur_assets_fillna'] -
                df['long_term_inv']
        )

        # Long-term debt
        df['long_term_debt'] = (
                df['quarterly_lt_borr_fillna'] +
                df['quarterly_lt_payable_fillna'] +
                df['quarterly_bond_payable_fillna']
        )

        # Noncurrent operating liabilities (Ncl)
        df['Ncl'] = (
                df['quarterly_total_liab_fillna'] -
                df['quarterly_total_cur_liab_fillna'] -
                df['long_term_debt']
        )

        # Financial assets (Fna) - short-term investments + long-term investments
        df['Fna'] = (df['cash_st_inv'] - df['quarterly_money_cap_fillna']) + df['long_term_inv']

        # Financial liabilities (Fnl)
        df['Fnl'] = df['long_term_debt'] + df['DLC'] + df['quarterly_oth_eqt_tools_p_shr_fillna']

        # Calculate changes (year-over-year)
        df = df.sort_values(['code', 'end_date'])

        df['dWc'] = (
                (df['Coa'] - df['Col']) -
                (df.groupby('code')['Coa'].shift(1) - df.groupby('code')['Col'].shift(1))
        )

        df['dNco'] = (
                (df['Nca'] - df['Ncl']) -
                (df.groupby('code')['Nca'].shift(1) - df.groupby('code')['Ncl'].shift(1))
        )

        df['dFin'] = (
                (df['Fna'] - df['Fnl']) -
                (df.groupby('code')['Fna'].shift(1) - df.groupby('code')['Fnl'].shift(1))
        )

        # Total accruals (Ta)
        df['Ta'] = df['dWc'] + df['dNco'] + df['dFin']

        # Calculate Pta = total accruals / |net income|
        df['abs_net_income'] = df['quarterly_n_income_fillna'].abs()

        # Avoid division by zero
        df['Pta'] = np.where(
            df['abs_net_income'] > 0,
            df['Ta'] / df['abs_net_income'],
            np.nan
        )

        # Replace inf/-inf with NaN
        df['Pta'] = df['Pta'].replace([np.inf, -np.inf], np.nan)

        return df[['code', 'end_date', 'Pta']].dropna()

    def validate(self) -> bool:
        """Validate Pta factor calculation."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_Pta = self.calculate()
        print(f"Shape: {df_Pta.shape}")
        print(f"Non-NaN values: {df_Pta['Pta'].notna().sum()}")
        print(f"Statistics:\n{df_Pta['Pta'].describe()}")
        print(f"Sample:\n{df_Pta.head(10)}")

        # Check total accruals calculation
        sample_stock = df_Pta['code'].iloc[0] if len(df_Pta) > 0 else None
        if sample_stock:
            sample_data = df_Pta[df_Pta['code'] == sample_stock].head(5)
            print(f"Sample stock {sample_stock} total accruals:\n{sample_data}")

        # Check for extreme values
        extreme_high = (df_Pta['Pta'] > 1).sum()
        extreme_low = (df_Pta['Pta'] < -1).sum()
        infinite_values = np.isinf(df_Pta['Pta']).sum()
        print(f"Extreme high Pta (>100%): {extreme_high}")
        print(f"Extreme negative Pta (<-100%): {extreme_low}")
        print(f"Infinite values: {infinite_values}")

        # Check December month filtering
        dec_months = pd.to_datetime(df_Pta['end_date']).dt.month
        print(f"December observations: {(dec_months == 12).sum()}")
        print(f"Non-December observations: {(dec_months != 12).sum()}")

        # Check data quality
        zero_assets = (df_Pta['Pta'].isna()).sum()
        print(f"Missing Pta values: {zero_assets}")

        # Convert to factor frame
        ft_Pta = self.as_factor_frame(df_Pta)
        print(f"Factor frame shape: {ft_Pta.shape}")
        print(f"Factor frame sample:\n{ft_Pta.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class NxfFactor(A3InvestmentFactorBase):
    """
    A.3.25 Nxf - Net External Financing

    Net external financing, Nxf, is the sum of net equity financing, Nef, and net debt financing, Ndf.

    Formula:
        Nxf = Nef + Ndf

    where:
        Nef = Δ(股本+资本公积+其他权益工具) + (息税前利润 - 利润总额) - 分配股利、利润或偿付利息支付的现金
        Ndf = 发行债券收到的现金 + 取得借款收到的现金 - 偿还债务支付的现金 + Δ(短期有息负债)

    At the end of June of each year t, stocks are split into deciles based on Nxf for the fiscal
    year ending in calendar year t−1 scaled by the average of total assets for fiscal years ending
    in t−2 and t−1.

    Data Sources:
        - Balance Sheet: total_share, cap_rese, oth_eqt_tools_p_shr, total_assets
        - Income Statement: total_profit, n_income, income_tax, int_exp
        - Cash Flow: proc_issue_bonds, c_recp_borrow, c_prepay_amt_borr, c_pay_dist_dpcp_int_exp
        - Balance Sheet: st_borr, st_bonds_payable, notes_payable, non_cur_liab_due_1y

    Implementation notes:
        - Annual fiscal year-end data (December only)
        - Uses 2-year average total assets for scaling
        - EBIT = Net Income + Interest Expense + Income Tax
    """

    @property
    def factor_id(self) -> str:
        return "A.3.25"

    @property
    def abbr(self) -> str:
        return "Nxf"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate net external financing factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Nxf']
        """
        # Load required data from balance sheet, income statement, and cash flow
        bs_cols = ['quarterly_total_share_fillna', 'quarterly_cap_rese_fillna',
                   'quarterly_oth_eqt_tools_p_shr_fillna', 'quarterly_total_assets_fillna',
                   'quarterly_st_borr_fillna', 'quarterly_st_bonds_payable_fillna',
                   'quarterly_notes_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna']
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)

        income_cols = ['quarterly_total_profit_fillna', 'quarterly_n_income_fillna',
                       'quarterly_income_tax_fillna', 'quarterly_int_exp_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)

        cf_cols = ['quarterly_proc_issue_bonds_fillna', 'quarterly_c_recp_borrow_fillna',
                   'quarterly_c_prepay_amt_borr_fillna', 'quarterly_c_pay_dist_dpcp_int_exp_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)

        # Merge all data
        df = pd.merge(df_bs, df_income, on=['code', 'end_date'], how='inner')
        df = pd.merge(df, df_cf, on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NaN with 0 for calculation
        df = df.fillna(0)

        # Calculate EBIT (息税前利润)
        df['ebit'] = df['quarterly_n_income_fillna'] + df['quarterly_income_tax_fillna'] + df[
            'quarterly_int_exp_fillna']

        # Calculate Nef (Net Equity Financing)
        df['equity_components'] = (df['quarterly_total_share_fillna'] +
                                   df['quarterly_cap_rese_fillna'] +
                                   df['quarterly_oth_eqt_tools_p_shr_fillna'])
        df['equity_components_lag1'] = df.groupby('code')['equity_components'].shift(1)
        df['nef_numerator'] = ((df['equity_components'] - df['equity_components_lag1']) +
                               (df['ebit'] - df['quarterly_total_profit_fillna']) -
                               df['quarterly_c_pay_dist_dpcp_int_exp_fillna'])

        # Calculate Ndf (Net Debt Financing)
        df['short_term_debt'] = (df['quarterly_st_borr_fillna'] +
                                 df['quarterly_st_bonds_payable_fillna'] +
                                 df['quarterly_notes_payable_fillna'] +
                                 df['quarterly_non_cur_liab_due_1y_fillna'])
        df['short_term_debt_lag1'] = df.groupby('code')['short_term_debt'].shift(1)
        df['ndf_numerator'] = (df['quarterly_proc_issue_bonds_fillna'] +
                               df['quarterly_c_recp_borrow_fillna'] -
                               df['quarterly_c_prepay_amt_borr_fillna'] +
                               (df['short_term_debt'] - df['short_term_debt_lag1']))

        # Calculate 2-year average total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['avg_total_assets'] = (df['quarterly_total_assets_fillna'] + df['total_assets_lag1']) / 2

        # Calculate Nxf = (Nef + Ndf) / avg_total_assets
        df['Nxf'] = (df['nef_numerator'] + df['ndf_numerator']) / df['avg_total_assets']

        return df[['code', 'end_date', 'Nxf']].dropna()

    def validate(self) -> bool:
        """Run validation tests for Nxf factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_nxf = self.calculate()
        print(f"Shape: {df_nxf.shape}")
        print(f"Non-NaN values: {df_nxf['Nxf'].notna().sum()}")
        print(f"Statistics:\n{df_nxf['Nxf'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_nxf['Nxf'] > 1.0).sum()
        n_le_zero = (df_nxf['Nxf'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_nxf.head(20)}")

        # Convert to factor frame
        ft_nxf = self.as_factor_frame(df_nxf)
        print(f"Factor frame shape: {ft_nxf.shape}")
        print(f"Factor frame sample:\n{ft_nxf.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class NefFactor(A3InvestmentFactorBase):
    """
    A.3.25 Nef - Net Equity Financing

    Net equity financing, Nef, is calculated as:
        Nef = Δ(股本+资本公积+其他权益工具) + (息税前利润 - 利润总额) - 分配股利、利润或偿付利息支付的现金

    At the end of June of each year t, stocks are split into deciles based on Nef for the fiscal
    year ending in calendar year t−1 scaled by the average of total assets for fiscal years ending
    in t−2 and t−1.

    Data Sources:
        - Balance Sheet: total_share, cap_rese, oth_eqt_tools_p_shr, total_assets
        - Income Statement: total_profit, n_income, income_tax, int_exp
        - Cash Flow: c_pay_dist_dpcp_int_exp

    Implementation notes:
        - Annual fiscal year-end data (December only)
        - Uses 2-year average total assets for scaling
        - EBIT = Net Income + Interest Expense + Income Tax
    """

    @property
    def factor_id(self) -> str:
        return "A.3.25"

    @property
    def abbr(self) -> str:
        return "Nef"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate net equity financing factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Nef']
        """
        # Load required data
        bs_cols = ['quarterly_total_share_fillna', 'quarterly_cap_rese_fillna',
                   'quarterly_oth_eqt_tools_p_shr_fillna', 'quarterly_total_assets_fillna']
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)

        income_cols = ['quarterly_total_profit_fillna', 'quarterly_n_income_fillna',
                       'quarterly_income_tax_fillna', 'quarterly_int_exp_fillna']
        df_income = load_income_data(income_cols, RESSET_DIR)

        cf_cols = ['quarterly_c_pay_dist_dpcp_int_exp_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)

        # Merge all data
        df = pd.merge(df_bs, df_income, on=['code', 'end_date'], how='inner')
        df = pd.merge(df, df_cf, on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NaN with 0 for calculation
        df = df.fillna(0)

        # Calculate EBIT (息税前利润)
        df['ebit'] = (df['quarterly_n_income_fillna'] + df['quarterly_income_tax_fillna'] +
                      df['quarterly_int_exp_fillna'])

        # Calculate equity components
        df['equity_components'] = (df['quarterly_total_share_fillna'] +
                                   df['quarterly_cap_rese_fillna'] +
                                   df['quarterly_oth_eqt_tools_p_shr_fillna'])
        df['equity_components_lag1'] = df.groupby('code')['equity_components'].shift(1)

        # Calculate Nef numerator
        df['nef_numerator'] = ((df['equity_components'] - df['equity_components_lag1']) +
                               (df['ebit'] - df['quarterly_total_profit_fillna']) -
                               df['quarterly_c_pay_dist_dpcp_int_exp_fillna'])

        # Calculate 2-year average total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['avg_total_assets'] = (df['quarterly_total_assets_fillna'] + df['total_assets_lag1']) / 2

        # Calculate Nef = numerator / avg_total_assets
        df['Nef'] = df['nef_numerator'] / df['avg_total_assets']

        return df[['code', 'end_date', 'Nef']].dropna()

    def validate(self) -> bool:
        """Run validation tests for Nef factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_nef = self.calculate()
        print(f"Shape: {df_nef.shape}")
        print(f"Non-NaN values: {df_nef['Nef'].notna().sum()}")
        print(f"Statistics:\n{df_nef['Nef'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_nef['Nef'] > 1.0).sum()
        n_le_zero = (df_nef['Nef'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_nef.head(20)}")

        # Convert to factor frame
        ft_nef = self.as_factor_frame(df_nef)
        print(f"Factor frame shape: {ft_nef.shape}")
        print(f"Factor frame sample:\n{ft_nef.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class NdfFactor(A3InvestmentFactorBase):
    """
    A.3.25 Ndf - Net Debt Financing

    Net debt financing, Ndf, is calculated as:
        Ndf = 发行债券收到的现金 + 取得借款收到的现金 - 偿还债务支付的现金 + Δ(短期有息负债)

    At the end of June of each year t, stocks are split into deciles based on Ndf for the fiscal
    year ending in calendar year t−1 scaled by the average of total assets for fiscal years ending
    in t−2 and t−1.

    Data Sources:
        - Cash Flow: proc_issue_bonds, c_recp_borrow, c_prepay_amt_borr
        - Balance Sheet: st_borr, st_bonds_payable, notes_payable, non_cur_liab_due_1y, total_assets

    Implementation notes:
        - Annual fiscal year-end data (December only)
        - Uses 2-year average total assets for scaling
        - Short-term debt includes: st_borr, st_bonds_payable, notes_payable, non_cur_liab_due_1y
    """

    @property
    def factor_id(self) -> str:
        return "A.3.25"

    @property
    def abbr(self) -> str:
        return "Ndf"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate net debt financing factor.

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ndf']
        """
        # Load required data
        cf_cols = ['quarterly_proc_issue_bonds_fillna', 'quarterly_c_recp_borrow_fillna',
                   'quarterly_c_prepay_amt_borr_fillna']
        df_cf = load_cashflow_data(cf_cols, path=RESSET_DIR)

        bs_cols = ['quarterly_st_borr_fillna', 'quarterly_st_bonds_payable_fillna',
                   'quarterly_notes_payable_fillna', 'quarterly_non_cur_liab_due_1y_fillna',
                   'quarterly_total_assets_fillna']
        df_bs = load_balancesheet_data(bs_cols, RESSET_DIR)

        # Merge data
        df = pd.merge(df_cf, df_bs, on=['code', 'end_date'], how='inner')

        # Filter to December only (fiscal year-end)
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df = df[df['end_date'].dt.month == 12].copy()
        df = df.sort_values(['code', 'end_date'])

        # Fill NaN with 0 for calculation
        df = df.fillna(0)

        # Calculate short-term debt
        df['short_term_debt'] = (df['quarterly_st_borr_fillna'] +
                                 df['quarterly_st_bonds_payable_fillna'] +
                                 df['quarterly_notes_payable_fillna'] +
                                 df['quarterly_non_cur_liab_due_1y_fillna'])
        df['short_term_debt_lag1'] = df.groupby('code')['short_term_debt'].shift(1)

        # Calculate Ndf numerator
        df['ndf_numerator'] = (df['quarterly_proc_issue_bonds_fillna'] +
                               df['quarterly_c_recp_borrow_fillna'] -
                               df['quarterly_c_prepay_amt_borr_fillna'] +
                               (df['short_term_debt'] - df['short_term_debt_lag1']))

        # Calculate 2-year average total assets
        df['total_assets_lag1'] = df.groupby('code')['quarterly_total_assets_fillna'].shift(1)
        df['avg_total_assets'] = (df['quarterly_total_assets_fillna'] + df['total_assets_lag1']) / 2

        # Calculate Ndf = numerator / avg_total_assets
        df['Ndf'] = df['ndf_numerator'] / df['avg_total_assets']

        return df[['code', 'end_date', 'Ndf']].dropna()

    def validate(self) -> bool:
        """Run validation tests for Ndf factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ndf = self.calculate()
        print(f"Shape: {df_ndf.shape}")
        print(f"Non-NaN values: {df_ndf['Ndf'].notna().sum()}")
        print(f"Statistics:\n{df_ndf['Ndf'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_ndf['Ndf'] > 1.0).sum()
        n_le_zero = (df_ndf['Ndf'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_ndf.head(20)}")

        # Convert to factor frame
        ft_ndf = self.as_factor_frame(df_ndf)
        print(f"Factor frame shape: {ft_ndf.shape}")
        print(f"Factor frame sample:\n{ft_ndf.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


# =============================================================================
# SECTION 6: A6 TRADING FRICTIONS FACTOR IMPLEMENTATIONS
# =============================================================================

class MeFactor(A6TradingFrictionsFactorBase):
    """
    A.6.1 Me - Market Equity (monthly)

    Me denotes market capitalization, calculated as price times shares outstanding.

    Formula:
        Me = price * shares_outstanding

    Data source:
        - close price from Wind dailystockreturn_wind.csv
        - float_share (流通股本) from daily_basic_resset.csv

    Implementation:
        - Uses circ_mv (流通市值) directly from RESSET daily_basic data
        - Gets month-end market value (last trading day each month)
    """

    @property
    def factor_id(self) -> str:
        return "A.6.1"

    @property
    def abbr(self) -> str:
        return "me"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate monthly market equity factor.

        Returns:
            DataFrame with columns: ['code', 'date', 'me']
        """
        # Load daily basic data for market value
        #df = load_daily_basic_data(['circ_mv'], RESSET_DIR)
        df = pd.read_csv(RESSET_DIR / "daily_basic_resset.csv")
        df = df[['code', 'date', 'circ_mv']]

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Get month-end values (last trading day each month)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df = df.sort_values(['code', 'date'])
        df_monthly = df.groupby(['code', 'year', 'month']).tail(1)

        # Use circ_mv as market equity (already in correct units)
        df_monthly = df_monthly.rename(columns={'circ_mv': 'me'})
        df_monthly = df_monthly.rename(columns={'date': 'end_date'})

        return df_monthly[['code', 'end_date', 'me']].dropna()

    def validate(self) -> bool:
        """Run validation tests for me factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_me = self.calculate()
        print(f"Shape: {df_me.shape}")
        print(f"Non-NaN values: {df_me['me'].notna().sum()}")
        print(f"Statistics:\n{df_me['me'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_me['me'] > 1.0).sum()
        n_le_zero = (df_me['me'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_me.head(20)}")

        # Convert to factor frame
        ft_me = self.as_factor_frame(df_me)
        print(f"Factor frame shape: {ft_me.shape}")
        print(f"Factor frame sample:\n{ft_me.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class IvffFactor(A6TradingFrictionsFactorBase):
    """
    A.6.2 Ivff1, Ivff6, Ivff12 - Idiosyncratic Volatility (Fama-French 3-Factor)

    Ivff is the residual volatility from regressing stock excess returns on the
    Fama-French 3-factor model over prior 1, 6, or 12 months.

    Formula:
        R_it - R_ft = α + β₁(R_Mt - R_ft) + β₂SMB_t + β₃HML_t + ε_it
        Ivff = √N × std(ε_it)

    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv
        - Rmrf_tmv, Smb_tmv, Hml_tmv from RESSET_FamaFrenchDaily.csv
        - riskfree rate from RESSET_RISKFREE.csv
    """

    @property
    def factor_id(self) -> str:
        return "A.6.2"

    @property
    def abbr(self) -> str:
        return "ivff"

    def _get_IV(self, df: pd.DataFrame, horizon: int) -> float:
        """
        Calculate idiosyncratic volatility for a given time horizon.

        Args:
            df: DataFrame with daily return data
            horizon: Lookback period in months (1, 6, or 12)

        Returns:
            IV value or np.nan if insufficient data
        """
        if df.empty or len(df) <= 1:
            return np.nan

        try:
            # Extract data for the horizon
            end_date = df['date'].max()
            start_date = end_date - pd.DateOffset(months=horizon)
            reg_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].dropna()

            if len(reg_df) < 15:  # Minimum observations requirement
                return np.nan

            # Prepare regression variables
            X = reg_df[['Rmrf_tmv', 'Smb_tmv', 'Hml_tmv']].values
            y = (reg_df['day_return'] - reg_df['riskfree']).values

            if len(X) < 10:  # Need sufficient data for regression
                return np.nan

            # Perform linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate residuals and IV
            predictions = model.predict(X)
            residuals = y - predictions
            N = len(residuals)
            IV = np.std(residuals, ddof=1) * np.sqrt(N)

            return IV

        except Exception as e:
            print(f"Warning: IV calculation failed: {e}")
            return np.nan

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate FF3 idiosyncratic volatility factors.

        Returns:
            DataFrame with columns: ['code', 'date', 'ivff1', 'ivff6', 'ivff12']
        """
        # Load and prepare main data (replicating document1's prepare_A6_main_data)
        try:
            # Load Wind daily data
            wind_data = load_wind_daily_data(['close', 'preclose', 'volume', 'amount'],
                                             _data_dir / "wind")

            # Load risk-free rate
            riskfree_path = RESSET_DIR / 'RESSET_RISKFREE.csv'
            riskfree = pd.read_csv(riskfree_path, names=['date', 'riskfree'], skiprows=1)
            riskfree['date'] = pd.to_datetime(riskfree['date'], format='%Y-%m-%d')
            riskfree['date'] = riskfree['date'].dt.strftime('%Y%m%d').astype('int64')

            # Load FF3 factors
            ff3_path = RESSET_DIR / 'RESSET_FamaFrenchDaily.csv'
            ff3 = pd.read_csv(ff3_path)
            ff3 = ff3.rename(columns={
                '日期_Date': 'date',
                '市场溢酬因子__流通市值加权_Rmrf_tmv': 'Rmrf_tmv',
                '市值因子__流通市值加权_Smb_tmv': 'Smb_tmv',
                '账面市值比因子__流通市值加权_Hml_tmv': 'Hml_tmv'
            })
            ff3 = ff3[['date', 'Rmrf_tmv', 'Smb_tmv', 'Hml_tmv']]
            ff3['date'] = pd.to_datetime(ff3['date'], format='%Y-%m-%d')
            ff3['date'] = ff3['date'].dt.strftime('%Y%m%d').astype('int64')

            # Merge data
            df = wind_data.merge(riskfree, on='date', how='left')
            df = df.merge(ff3, on='date', how='left')

            # Filter to post-2010 data as in document1
            df = df[df['date'] >= 20100104]

        except Exception as e:
            print(f"Error loading data for Ivff: {e}")
            return pd.DataFrame(columns=['code', 'date', 'ivff1', 'ivff6', 'ivff12'])

        # Calculate daily returns
        df['day_return'] = (df['close'] - df['preclose']) / df['preclose']
        df = df[['code', 'date', 'day_return', 'Rmrf_tmv', 'Hml_tmv', 'Smb_tmv', 'riskfree']]

        # Convert date format
        df = date_transfer(df, 'date')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Apply minimum observation filter (15 days per month)
        df['count'] = df.groupby(['code', 'year', 'month'])['code'].transform('count')
        df = df[df['count'] >= 15]

        # Calculate market excess return
        #df['Market_No_risk'] = df['Rmrf_tmv'] - df['riskfree']

        # Calculate IV for different horizons
        results = []
        for (code, year, month), group in df.groupby(['code', 'year', 'month']):
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

            iv1 = self._get_IV(group, 1)
            iv6 = self._get_IV(group, 6)
            iv12 = self._get_IV(group, 12)

            results.append({
                'code': code,
                'date': month_end,
                'ivff1': iv1,
                'ivff6': iv6,
                'ivff12': iv12
            })

        result_df = pd.DataFrame(results)
        return result_df.dropna(subset=['ivff1', 'ivff6', 'ivff12'], how='all')

    def validate(self) -> bool:
        """Validate Ivff factor calculation."""
        df = self.calculate()
        if df.empty:
            print("❌ Ivff factor calculation failed - empty DataFrame")
            return False

        print(f"✅ Ivff factor validation passed")
        print(f"   Shape: {df.shape}")
        print(f"   Non-null values - ivff1: {df['ivff1'].notna().sum()}, "
              f"ivff6: {df['ivff6'].notna().sum()}, ivff12: {df['ivff12'].notna().sum()}")
        return True


class IvcFactor(A6TradingFrictionsFactorBase):
    """
    A.6.4 Ivc1, Ivc6, Ivc12 - Idiosyncratic Volatility (CAPM)

    Ivc is the residual volatility from regressing stock excess returns on
    the market excess return (CAPM) over prior 1, 6, or 12 months.

    Formula:
        R_it - R_ft = α + β(R_Mt - R_ft) + ε_it
        Ivc = √N × std(ε_it)

    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv
        - Rmrf_tmv from RESSET_FamaFrenchDaily.csv (market factor)
        - riskfree rate from RESSET_RISKFREE.csv
    """

    @property
    def factor_id(self) -> str:
        return "A.6.4"

    @property
    def abbr(self) -> str:
        return "ivc"

    def _get_IVC(self, df: pd.DataFrame, horizon: int) -> float:
        """
        Calculate CAPM idiosyncratic volatility for a given time horizon.

        Args:
            df: DataFrame with daily return data
            horizon: Lookback period in months (1, 6, or 12)

        Returns:
            IV value or np.nan if insufficient data
        """
        if df.empty or len(df) <= 1:
            return np.nan

        try:
            # Extract data for the horizon
            end_date = df['date'].max()
            start_date = end_date - pd.DateOffset(months=horizon)
            reg_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].dropna()

            if len(reg_df) < 15:  # Minimum observations requirement
                return np.nan

            # Prepare regression variables (CAPM - only market factor)
            X = reg_df[['Rmrf_tmv']].values
            y = (reg_df['day_return'] - reg_df['riskfree']).values

            if len(X) < 10:  # Need sufficient data for regression
                return np.nan

            # Perform linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate residuals and IV
            predictions = model.predict(X)
            residuals = y - predictions
            N = len(residuals)
            IV = np.std(residuals, ddof=1) * np.sqrt(N)

            return IV

        except Exception as e:
            print(f"Warning: IVC calculation failed: {e}")
            return np.nan

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        Calculate CAPM idiosyncratic volatility factors.

        Returns:
            DataFrame with columns: ['code', 'date', 'ivc1', 'ivc6', 'ivc12']
        """
        # Reuse the data loading logic from IvffFactor
        ivff_factor = IvffFactor()

        try:
            # Load Wind daily data
            wind_data = load_wind_daily_data(['close', 'preclose'], _data_dir / "wind")

            # Load risk-free rate
            riskfree_path = RESSET_DIR / 'RESSET_RISKFREE.csv'
            riskfree = pd.read_csv(riskfree_path, names=['date', 'riskfree'], skiprows=1)
            riskfree['date'] = pd.to_datetime(riskfree['date'], format='%Y-%m-%d')
            riskfree['date'] = riskfree['date'].dt.strftime('%Y%m%d').astype('int64')

            # Load market factor only
            ff3_path = RESSET_DIR / 'RESSET_FamaFrenchDaily.csv'
            ff3 = pd.read_csv(ff3_path)
            ff3 = ff3.rename(columns={
                '日期_Date': 'date',
                '市场溢酬因子__流通市值加权_Rmrf_tmv': 'Rmrf_tmv'
            })
            ff3 = ff3[['date', 'Rmrf_tmv']]
            ff3['date'] = pd.to_datetime(ff3['date'], format='%Y-%m-%d')
            ff3['date'] = ff3['date'].dt.strftime('%Y%m%d').astype('int64')

            # Merge data
            df = wind_data.merge(riskfree, on='date', how='left')
            df = df.merge(ff3, on='date', how='left')
            df = df[df['date'] >= 20100104]

        except Exception as e:
            print(f"Error loading data for Ivc: {e}")
            return pd.DataFrame(columns=['code', 'date', 'ivc1', 'ivc6', 'ivc12'])

        # Calculate daily returns
        df['day_return'] = (df['close'] - df['preclose']) / df['preclose']
        df = df[['code', 'date', 'day_return', 'Rmrf_tmv', 'riskfree']]

        # Convert date format
        df = date_transfer(df, 'date')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Apply minimum observation filter
        df['count'] = df.groupby(['code', 'year', 'month'])['code'].transform('count')
        df = df[df['count'] >= 15]

        # Calculate market excess return
        #df['Market_No_risk'] = df['Rmrf_tmv'] - df['riskfree']

        # Calculate IV for different horizons
        results = []
        for (code, year, month), group in df.groupby(['code', 'year', 'month']):
            month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

            iv1 = self._get_IVC(group, 1)
            iv6 = self._get_IVC(group, 6)
            iv12 = self._get_IVC(group, 12)

            results.append({
                'code': code,
                'date': month_end,
                'ivc1': iv1,
                'ivc6': iv6,
                'ivc12': iv12
            })

        result_df = pd.DataFrame(results)
        return result_df.dropna(subset=['ivc1', 'ivc6', 'ivc12'], how='all')

    def validate(self) -> bool:
        """Validate Ivc factor calculation."""
        df = self.calculate()
        if df.empty:
            print("❌ Ivc factor calculation failed - empty DataFrame")
            return False

        print(f"✅ Ivc factor validation passed")
        print(f"   Shape: {df.shape}")
        print(f"   Non-null values - ivc1: {df['ivc1'].notna().sum()}, "
              f"ivc6: {df['ivc6'].notna().sum()}, ivc12: {df['ivc12'].notna().sum()}")
        return True


class TurFactor(A6TradingFrictionsFactorBase):
    """
    A.6.11 Tur1, Tur6, Tur12 - Share Turnover

    Tur is the average daily share turnover over prior 1, 6, or 12 months.
    Requires minimum of 50 observations for 6-month and 12-month factors.

    Formula:
        Tur = average(daily_turnover_rate) over specified period
        where daily_turnover_rate = shares_traded / shares_outstanding

    Data source:
        - turnover_rate_f (换手率-自由流通股) from daily_basic_resset.csv
    """

    @property
    def factor_id(self) -> str:
        return "A.6.11"

    @property
    def abbr(self) -> str:
        return "tur"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：过去6个月的平均日换手率。
        每月最后一个交易日输出一个数值，代表对该股票下个月初排序的依据。
        """
        # 1. 加载日频换手率数据
        df_daily = load_daily_basic_data(['turnover_rate_f'], RESSET_DIR)
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily.rename(columns={'turnover_rate_f': 'daily_turnover'}).dropna()

        # 2. 转换为月频：计算每个股票每月的平均日换手率及交易日计数
        df_daily['year_month'] = df_daily['date'].dt.to_period('M')
        monthly_stats = df_daily.groupby(['code', 'year_month']).agg(
            monthly_turnover=('daily_turnover', 'mean'),
            trade_days=('daily_turnover', 'count')
        ).reset_index()

        # 3. 计算过去6个月的滚动平均（核心排序因子）
        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])
        # 计算滚动6个月的总交易天数
        monthly_stats['roll_6m_days'] = monthly_stats.groupby('code')['trade_days'].transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
        # 计算滚动6个月的换手率总和（平均日换手率 * 当月交易天数）
        monthly_stats['roll_6m_turnover_sum'] = (
                    monthly_stats['monthly_turnover'] * monthly_stats['trade_days']).groupby(
            monthly_stats['code']).transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
        # 计算过去6个月平均日换手率 = 总换手率 / 总交易天数，并应用50天过滤
        monthly_stats['tur'] = monthly_stats['roll_6m_turnover_sum'] / monthly_stats['roll_6m_days']
        monthly_stats.loc[monthly_stats['roll_6m_days'] < 50, 'tur'] = np.nan

        # 4. 整理输出，将月末时点作为因子值生效的日期
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')  # 月末
        return monthly_stats[['code', 'end_date', 'tur']].dropna()

    def validate(self) -> bool:
        """Run validation tests for tur factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_tur = self.calculate()
        print(f"Shape: {df_tur.shape}")
        print(f"Non-NaN values: {df_tur['tur'].notna().sum()}")
        print(f"Statistics:\n{df_tur['tur'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_tur['tur'] > 1.0).sum()
        n_le_zero = (df_tur['tur'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_tur.head(20)}")

        # Convert to factor frame
        ft_tur = self.as_factor_frame(df_tur)
        print(f"Factor frame shape: {ft_tur.shape}")
        print(f"Factor frame sample:\n{ft_tur.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class DtvFactor(A6TradingFrictionsFactorBase):
    """
    A.6.13 Dtv1, Dtv6, Dtv12 - Dollar Trading Volume

    Dtv is the average daily dollar trading volume over prior 1, 6, or 12 months.
    Dollar trading volume = share price × number of shares traded.

    Formula:
        Dtv = average(amount) over specified period
        where amount = trading amount in thousand yuan

    Data source:
        - amount (成交金额-千元) from Wind dailystockreturn_wind.csv
    """

    @property
    def factor_id(self) -> str:
        return "A.6.13"

    @property
    def abbr(self) -> str:
        return "dtv"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：过去6个月的平均日美元交易额。

        要求：至少50个交易日观测值。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'dtv']
        """
        # 1. 加载日频美元交易额数据
        # 注意：Wind数据中的amount是成交额（人民币元），即股价×成交量
        df = load_wind_daily_data(['amount'], _data_dir / "wind")
        df = df.rename(columns={'amount': 'dollar_volume'})

        # 数据清洗
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'dollar_volume'])
        df = df.sort_values(['code', 'date'])

        # 2. 转换为月频数据：计算每月平均日交易额和交易日数
        df['year_month'] = df['date'].dt.to_period('M')

        monthly_stats = df.groupby(['code', 'year_month']).agg(
            monthly_dollar_volume=('dollar_volume', 'mean'),
            trade_days=('dollar_volume', 'count')
        ).reset_index()

        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])

        # 3. 计算过去6个月的平均日交易额
        # 计算滚动6个月的总交易额（平均日交易额 × 当月交易日数）
        monthly_stats['roll_6m_volume_sum'] = (
                monthly_stats['monthly_dollar_volume'] * monthly_stats['trade_days']
        ).groupby(monthly_stats['code']).transform(
            lambda x: x.rolling(6, min_periods=1).sum()
        )

        # 计算滚动6个月的总交易日数
        monthly_stats['roll_6m_days'] = monthly_stats.groupby('code')['trade_days'].transform(
            lambda x: x.rolling(6, min_periods=1).sum()
        )

        # 计算过去6个月平均日交易额
        monthly_stats['dtv'] = (
                monthly_stats['roll_6m_volume_sum'] / monthly_stats['roll_6m_days']
        )

        # 4. 应用50个交易日要求
        monthly_stats.loc[monthly_stats['roll_6m_days'] < 50, 'dtv'] = np.nan

        # 5. 准备输出（以月末为时点）
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')

        return monthly_stats[['code', 'end_date', 'dtv']].dropna()

    def validate(self) -> bool:
        """Run validation tests for dtv factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_dtv = self.calculate()
        print(f"Shape: {df_dtv.shape}")
        print(f"Non-NaN values: {df_dtv['dtv'].notna().sum()}")
        print(f"Statistics:\n{df_dtv['dtv'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_dtv['dtv'] > 1.0).sum()
        n_le_zero = (df_dtv['dtv'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_dtv.head(20)}")

        # Convert to factor frame
        ft_dtv = self.as_factor_frame(df_dtv)
        print(f"Factor frame shape: {ft_dtv.shape}")
        print(f"Factor frame sample:\n{ft_dtv.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class PpsFactor(A6TradingFrictionsFactorBase):
    """
    A.6.15 Pps1, Pps6, and Pps12 - Share Price

    Pps is the share price at the end of the month.
    We use the closing price from Wind daily stock return data.

    Formula:
        Pps = Close price at month-end

    Data source:
        - close (收盘价) from Wind dailystockreturn_wind.csv

    Implementation:
        - Uses month-end closing price (last trading day each month)
        - For multi-period factors (Pps6, Pps12), uses rolling averages of past 6 and 12 months
        - Requires minimum 50 observations in past 6 months
    """

    @property
    def factor_id(self) -> str:
        return "A.6.15"

    @property
    def abbr(self) -> str:
        return "pps"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：每月末的收盘价。

        原文要求：使用每月最后一个交易日的收盘价。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'pps']
        """
        # 1. 加载日频收盘价数据
        #df = load_daily_basic_data(['close'], RESSET_DIR)
        df = pd.read_csv(project_root/RESSET_DIR/"daily_basic_resset.csv")
        df = df[['code', 'date', 'close']]
        df = df.rename(columns={'close': 'price'})

        # 数据清洗
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'price'])
        df = df.sort_values(['code', 'date'])

        # 2. 转换为月频：获取每月最后一个交易日的收盘价
        df['year_month'] = df['date'].dt.to_period('M')

        # 获取每月最后一条记录（月末收盘价）
        monthly_end_prices = df.groupby(['code', 'year_month']).tail(1).copy()
        monthly_end_prices = monthly_end_prices[['code', 'year_month', 'date', 'price']]

        # 3. 计算因子值：月末收盘价
        monthly_end_prices['pps'] = monthly_end_prices['price']

        # 4. 准备输出
        monthly_end_prices['end_date'] = monthly_end_prices['year_month'].dt.to_timestamp('M')

        return monthly_end_prices[['code', 'end_date', 'pps']].dropna()

    def validate(self) -> bool:
        """Run validation tests for pps factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_pps = self.calculate()
        print(f"Shape: {df_pps.shape}")
        print(f"Non-NaN values: {df_pps['pps'].notna().sum()}")
        print(f"Statistics:\n{df_pps['pps'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_pps['pps'] > 1.0).sum()
        n_le_zero = (df_pps['pps'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_pps.head(20)}")

        # Convert to factor frame
        ft_pps = self.as_factor_frame(df_pps)
        print(f"Factor frame shape: {ft_pps.shape}")
        print(f"Factor frame sample:\n{ft_pps.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class AmiFactor(A6TradingFrictionsFactorBase):
    """
    A.6.16 Ami1, Ami6, and Ami12 - Absolute Return-to-Volume

    Ami is the ratio of absolute daily stock return to daily dollar trading volume,
    averaged over the prior one, six, or twelve months.

    Formula:
        Ami = average(|Return| / Amount) over specified period

    where:
        - Return = daily return calculated from close and preclose
        - Amount = trading amount in thousand yuan

    Data source:
        - close, preclose, amount from Wind dailystockreturn_wind.csv

    Implementation:
        - Calculates daily ratio of absolute return to dollar volume
        - Averages over 6 months
        - Uses weighted rolling averages for multi-period factors
    """

    @property
    def factor_id(self) -> str:
        return "A.6.16"

    @property
    def abbr(self) -> str:
        return "ami"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：过去6个月的Amihud指标平均值。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'ami']
        """
        # 1. 加载日频数据
        try:
            df = load_wind_daily_data(['close', 'preclose', 'amount'], _data_dir / "wind")
        except:
            # 如果wind数据加载失败，尝试其他数据源
            df = load_daily_basic_data(['close', 'pre_close', 'amount'], RESSET_DIR)
            df = df.rename(columns={'pre_close': 'preclose'})

        # 数据清洗
        df = date_transfer(df, 'date')
        df = df.dropna(subset=['date', 'close', 'preclose', 'amount'])
        df = df.sort_values(['code', 'date'])

        # 2. 计算每日Amihud指标
        # 日收益率
        df['daily_return'] = (df['close'] - df['preclose']) / df['preclose']

        # 绝对收益率除以成交金额
        # 注意：避免除零，当amount为0时设为NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            df['amihud_daily'] = df['daily_return'].abs() / df['amount']

        # 处理异常值
        df['amihud_daily'] = df['amihud_daily'].replace([np.inf, -np.inf], np.nan)

        # 3. 转换为月频：计算每月平均Amihud指标
        df['year_month'] = df['date'].dt.to_period('M')

        monthly_stats = df.groupby(['code', 'year_month']).agg(
            monthly_amihud=('amihud_daily', 'mean'),
            trade_days=('amihud_daily', 'count')
        ).reset_index()

        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])

        # 4. 计算过去6个月的Amihud指标平均值
        # 计算滚动6个月的总Amihud（月度平均 * 当月交易日数）
        monthly_stats['roll_6m_amihud_sum'] = (
                monthly_stats['monthly_amihud'] * monthly_stats['trade_days']
        ).groupby(monthly_stats['code']).transform(
            lambda x: x.rolling(6, min_periods=1).sum()
        )

        # 计算滚动6个月的总交易日数
        monthly_stats['roll_6m_days'] = monthly_stats.groupby('code')['trade_days'].transform(
            lambda x: x.rolling(6, min_periods=1).sum()
        )

        # 计算过去6个月平均Amihud指标
        monthly_stats['ami'] = (
                monthly_stats['roll_6m_amihud_sum'] / monthly_stats['roll_6m_days']
        )

        # 5. 应用50个交易日要求
        monthly_stats.loc[monthly_stats['roll_6m_days'] < 50, 'ami'] = np.nan

        # 6. 准备输出
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')

        return monthly_stats[['code', 'end_date', 'ami']].dropna()

    def validate(self) -> bool:
        """Run validation tests for ami factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ami = self.calculate()
        print(f"Shape: {df_ami.shape}")
        print(f"Non-NaN values: {df_ami['ami'].notna().sum()}")
        print(f"Statistics:\n{df_ami['ami'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_ami['ami'] > 1.0).sum()
        n_le_zero = (df_ami['ami'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_ami.head(20)}")

        # Convert to factor frame
        ft_ami = self.as_factor_frame(df_ami)
        print(f"Factor frame shape: {ft_ami.shape}")
        print(f"Factor frame sample:\n{ft_ami.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class MdrFactor(A6TradingFrictionsFactorBase):
    """
    A.6.18 Mdr1, Mdr6, and Mdr12 - Maximum Daily Return

    Mdr is the maximum daily return over the prior one, six, or twelve months.

    Formula:
        Mdr = max(daily return) over specified period

    where daily return is calculated from close and preclose prices.

    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv

    Implementation:
        - Calculates daily returns
        - Takes maximum over 1, 6, and 12 month periods
        - Uses rolling window maximum for multi-period factors
    """

    @property
    def factor_id(self) -> str:
        return "A.6.18"

    @property
    def abbr(self) -> str:
        return "mdr"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：上个月的最大日收益率。

        注意：原文中使用的是t-1月的最大日收益率，在t月初进行排序。
        因此，我们计算每月末的最大日收益率，在回测框架中作为下个月初的排序依据。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'mdr']
        """
        # 1. 加载日频数据
        try:
            df = load_wind_daily_data(['close', 'preclose'], _data_dir / "wind")
        except:
            # 如果wind数据加载失败，尝试其他数据源
            df = load_daily_basic_data(['close', 'pre_close'], RESSET_DIR)
            df = df.rename(columns={'pre_close': 'preclose'})

        # 数据清洗
        df = date_transfer(df, 'date')
        df = df.dropna(subset=['date', 'close', 'preclose'])
        df = df.sort_values(['code', 'date'])

        # 2. 计算日收益率
        df['daily_return'] = (df['close'] - df['preclose']) / df['preclose']

        # 处理异常值
        df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan)

        # 3. 转换为月频：计算每月最大日收益率
        df['year_month'] = df['date'].dt.to_period('M')

        monthly_stats = df.groupby(['code', 'year_month']).agg(
            max_daily_return=('daily_return', 'max'),  # 月度最大日收益率
            trade_days=('daily_return', 'count')  # 交易日数
        ).reset_index()

        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])

        # 4. 应用15个交易日要求
        monthly_stats['mdr'] = np.where(
            monthly_stats['trade_days'] >= 15,
            monthly_stats['max_daily_return'],
            np.nan
        )

        # 5. 准备输出
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')

        return monthly_stats[['code', 'end_date', 'mdr']].dropna()

    def validate(self) -> bool:
        """Run validation tests for mdr factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_mdr = self.calculate()
        print(f"Shape: {df_mdr.shape}")
        print(f"Non-NaN values: {df_mdr['mdr'].notna().sum()}")
        print(f"Statistics:\n{df_mdr['mdr'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_mdr['mdr'] > 1.0).sum()
        n_le_zero = (df_mdr['mdr'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_mdr.head(20)}")

        # Convert to factor frame
        ft_mdr = self.as_factor_frame(df_mdr)
        print(f"Factor frame shape: {ft_mdr.shape}")
        print(f"Factor frame sample:\n{ft_mdr.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class TsFactor(A6TradingFrictionsFactorBase):
    """
    A.6.19 Ts1, Ts6, and Ts12 - Total Skewness

    Ts is the skewness of daily returns over the prior one month.

    Formula:
        Ts = skewness(daily returns) over specified period

    where skewness is calculated as the third standardized moment.

    Data source:
        - close, preclose from Wind dailystockreturn_wind.csv

    Implementation:
        - Calculates daily returns
        - Calculates skewness for 1 month period
        - Uses rolling window skewness for multi-period factors
    """

    @property
    def factor_id(self) -> str:
        return "A.6.19"

    @property
    def abbr(self) -> str:
        return "Ts"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：上个月的日收益率总偏度。

        注意：原文中使用的是t-1月的日收益率计算偏度，在t月初进行排序。
        因此，我们计算每月末的偏度值，在回测框架中作为下个月初的排序依据。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'Ts']
        """
        # 1. 加载日频数据
        try:
            df = load_wind_daily_data(['close', 'preclose'], _data_dir / "wind")
        except:
            # 如果wind数据加载失败，尝试其他数据源
            df = load_daily_basic_data(['close', 'pre_close'], RESSET_DIR)
            df = df.rename(columns={'pre_close': 'preclose'})

        # 数据清洗
        df = date_transfer(df, 'date')
        df = df.dropna(subset=['date', 'close', 'preclose'])
        df = df.sort_values(['code', 'date'])

        # 2. 计算日收益率
        df['daily_return'] = (df['close'] - df['preclose']) / df['preclose']

        # 处理异常值
        df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan)

        # 3. 转换为月频：计算每月日收益率的偏度
        df['year_month'] = df['date'].dt.to_period('M')

        # 定义计算偏度的函数
        def calculate_skewness(group):
            if len(group) < 3:  # 偏度计算至少需要3个数据点
                return np.nan
            return group.skew()

        monthly_stats = df.groupby(['code', 'year_month']).agg(
            skewness=('daily_return', calculate_skewness),  # 月度偏度
            trade_days=('daily_return', 'count')  # 交易日数
        ).reset_index()

        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])

        # 4. 应用15个交易日要求
        monthly_stats['Ts'] = np.where(
            monthly_stats['trade_days'] >= 15,
            monthly_stats['skewness'],
            np.nan
        )

        # 5. 准备输出
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')

        return monthly_stats[['code', 'end_date', 'Ts']].dropna()

    def validate(self) -> bool:
        """Run validation tests for ts factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_ts = self.calculate()
        print(f"Shape: {df_ts.shape}")
        print(f"Non-NaN values: {df_ts['Ts'].notna().sum()}")
        print(f"Statistics:\n{df_ts['Ts'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_ts['Ts'] > 1.0).sum()
        n_le_zero = (df_ts['Ts'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_ts.head(20)}")

        # Convert to factor frame
        ft_ts = self.as_factor_frame(df_ts)
        print(f"Factor frame shape: {ft_ts.shape}")
        print(f"Factor frame sample:\n{ft_ts.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class SrevFactor(A6TradingFrictionsFactorBase):
    """
    A.6.26 Srev - Short-term Reversal

    Srev is the minimum monthly return in the prior month.

    Formula:
        Srev = min(monthly return) in month t-1

    This factor aims to capture the short-term reversal effect where stocks
    with poor recent performance tend to rebound.

    Data source:
        - close from Wind dailystockreturn_wind.csv

    Implementation:
        - Calculates monthly returns using month-end closing prices
        - Takes minimum return over the month
        - Uses lagged monthly return for reversal signal
    """

    @property
    def factor_id(self) -> str:
        return "A.6.26"

    @property
    def abbr(self) -> str:
        return "srev"

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：t-1月的月度收益率。

        注意：原文中使用t-1月的收益率在t月初进行排序。
        我们需要确保：
        1. t-2月末有有效价格
        2. t-1月有有效收益率

        Returns:
            DataFrame with columns: ['code', 'end_date', 'srev']
        """
        # 1. 加载日频收盘价数据
        try:
            df = load_wind_daily_data(['close'], _data_dir / "wind")
        except:
            # 如果wind数据加载失败，尝试其他数据源
            df = load_daily_basic_data(['close'], RESSET_DIR)

        # 数据清洗
        df = date_transfer(df, 'date')
        df = df.dropna(subset=['date', 'close'])
        df = df.sort_values(['code', 'date'])

        # 2. 转换为月频：获取每月最后一个交易日的收盘价
        df['year_month'] = df['date'].dt.to_period('M')

        # 获取每月最后一个交易日的收盘价
        monthly_prices = df.groupby(['code', 'year_month']).tail(1).copy()
        monthly_prices = monthly_prices[['code', 'year_month', 'date', 'close']]

        # 3. 计算月度收益率
        # 月度收益率 = (当月月末收盘价 - 上月月末收盘价) / 上月月末收盘价
        monthly_prices = monthly_prices.sort_values(['code', 'year_month'])
        monthly_prices['prev_close'] = monthly_prices.groupby('code')['close'].shift(1)

        # 计算月度收益率
        monthly_prices['monthly_return'] = (
                (monthly_prices['close'] - monthly_prices['prev_close']) /
                monthly_prices['prev_close']
        )

        # 4. 构建因子值
        # 对于t月，我们使用t-1月的月度收益率作为因子值
        # 所以需要将收益率向前移动一个月
        monthly_prices['srev'] = monthly_prices.groupby('code')['monthly_return'].shift(1)

        # 5. 准备输出
        # 使用月末日期作为因子值的时间戳
        monthly_prices['end_date'] = monthly_prices['year_month'].dt.to_timestamp('M')

        return monthly_prices[['code', 'end_date', 'srev']].dropna()

    def validate(self) -> bool:
        """Run validation tests for Srev factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_srev = self.calculate()
        print(f"Shape: {df_srev.shape}")
        print(f"Non-NaN values: {df_srev['srev'].notna().sum()}")
        print(f"Statistics:\n{df_srev['srev'].describe()}")

        # Check for extreme reversal values
        if not df_srev.empty:
            extreme_negative = (df_srev['srev'] < -0.5).sum()  # < -50%
            print(f"Extreme negative reversal (<-50%): {extreme_negative}")

            # Check if most values are negative (as expected for reversal)
            negative_count = (df_srev['srev'] < 0).sum()
            negative_pct = negative_count / len(df_srev) * 100
            print(f"Negative values: {negative_count} ({negative_pct:.1f}%)")

        print(f"Sample:\n{df_srev.head(10)}")

        # Convert to factor frame
        ft_srev = self.as_factor_frame(df_srev)
        print(f"Factor frame shape: {ft_srev.shape}")
        print(f"Factor frame sample:\n{ft_srev.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


class ShlFactor(A6TradingFrictionsFactorBase):
    """
    A.6.28 Shl1, Shl6, and Shl12 - High-low Bid-ask Spread Estimator

    Shl is the monthly Corwin and Schultz (2012) stock-level high-low bid-ask spread estimator.

    Formula:
        Shl = 2*(e^α - 1) / (1 + e^α)

    where:
        α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
        β = Σ_{j=0}^1 [log(H_{t+j}/L_{t+j})]²
        γ = [log(H_{t,t+1}/L_{t,t+1})]²

    H_t(L_t) is the highest(lowest) price on day t, and H_{t,t+1}(L_{t,t+1})
    is the highest(lowest) price on days t and t+1.

    Data source:
        - high, low from Wind dailystockreturn_wind.csv

    Implementation:
        - Calculates Corwin-Schultz high-low spread estimator
        - Averages over 1, 6, and 12 months
        - Uses rolling averages for multi-period factors
    """

    @property
    def factor_id(self) -> str:
        return "A.6.28"

    @property
    def abbr(self) -> str:
        return "shl"

    def _calculate_corwin_schultz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Corwin-Schultz(2012)买卖价差估计值

        公式：
        S = 2*(exp(α)-1)/(1+exp(α))
        其中：
        β = [ln(H_{t,t+1}/L_{t,t+1})]^2 + [ln(H_t/L_t)]^2
        γ = [ln(H_{t,t+1}/L_{t,t+1})]^2
        α = (√(2β) - √β)/(3-2√2) - √(γ/(3-2√2))

        H_{t,t+1}: 两天内的最高价
        L_{t,t+1}: 两天内的最低价
        H_t: 第t天的最高价
        L_t: 第t天的最低价
        """
        df = df.copy()

        # 确保按时间和股票排序
        df = df.sort_values(['code', 'date'])

        # 获取下一天的最高最低价
        df['high_next'] = df.groupby('code')['high'].shift(-1)
        df['low_next'] = df.groupby('code')['low'].shift(-1)

        # 计算两天窗口内的最高最低价
        df['high_2day'] = df[['high', 'high_next']].max(axis=1)
        df['low_2day'] = df[['low', 'low_next']].min(axis=1)

        # 计算γ和β
        # 避免log(0)的情况
        df['high_2day'] = df['high_2day'].replace(0, np.nan)
        df['low_2day'] = df['low_2day'].replace(0, np.nan)
        df['high'] = df['high'].replace(0, np.nan)
        df['low'] = df['low'].replace(0, np.nan)

        # 计算自然对数
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_hl_2day'] = np.log(df['high_2day'] / df['low_2day'])
            df['log_hl_1day'] = np.log(df['high'] / df['low'])

        # 计算γ和β
        df['gamma'] = df['log_hl_2day'] ** 2
        df['beta'] = df['log_hl_2day'] ** 2 + df['log_hl_1day'] ** 2

        # 计算α（处理负数）
        df['sqrt_2beta'] = np.sqrt(2 * df['beta'])
        df['sqrt_beta'] = np.sqrt(df['beta'])
        df['sqrt_gamma'] = np.sqrt(df['gamma'])

        # 计算α值
        denominator = 3 - 2 * np.sqrt(2)
        df['alpha'] = (df['sqrt_2beta'] - df['sqrt_beta']) / denominator - df['sqrt_gamma'] / np.sqrt(denominator)

        # 计算Shl（买卖价差估计值）
        df['shl_daily'] = 2 * (np.exp(df['alpha']) - 1) / (1 + np.exp(df['alpha']))

        # 限制在合理范围内（0到1之间）
        df['shl_daily'] = df['shl_daily'].clip(lower=0, upper=1)

        return df

    def calculate(self, **kwargs) -> pd.DataFrame:
        """
        计算用于排序的因子：上个月的Corwin-Schultz买卖价差估计值。

        Returns:
            DataFrame with columns: ['code', 'end_date', 'shl']
        """
        # 1. 加载日频最高最低价数据
        try:
            df = load_wind_daily_data(['high', 'low'], _data_dir / "wind")
        except:
            # 如果wind数据加载失败，尝试其他数据源
            df = load_daily_basic_data(['high', 'low'], RESSET_DIR)

        # 数据清洗
        df = date_transfer(df, 'date')
        df = df.dropna(subset=['date', 'high', 'low'])
        df = df.sort_values(['code', 'date'])

        # 2. 计算每日Corwin-Schultz估计值
        df = self._calculate_corwin_schultz(df)

        # 3. 转换为月频：计算每月平均Shl
        df['year_month'] = df['date'].dt.to_period('M')

        monthly_stats = df.groupby(['code', 'year_month']).agg(
            shl_monthly=('shl_daily', 'mean'),
            trade_days=('shl_daily', 'count')
        ).reset_index()

        monthly_stats = monthly_stats.sort_values(['code', 'year_month'])

        # 4. 应用最小交易日要求（原文未明确要求，但合理假设需要一定数据量）
        # 设置最小交易日数为15天
        min_days = kwargs.get('min_trade_days', 15)
        monthly_stats['shl'] = np.where(
            monthly_stats['trade_days'] >= min_days,
            monthly_stats['shl_monthly'],
            np.nan
        )

        # 5. 准备输出（使用月末作为时点）
        monthly_stats['end_date'] = monthly_stats['year_month'].dt.to_timestamp('M')

        return monthly_stats[['code', 'end_date', 'shl']].dropna()

    def validate(self) -> bool:
        """Run validation tests for shl factor."""
        print(f"Validating {self.factor_id} {self.abbr}...")

        # Calculate factor
        df_shl = self.calculate()
        print(f"Shape: {df_shl.shape}")
        print(f"Non-NaN values: {df_shl['shl'].notna().sum()}")
        print(f"Statistics:\n{df_shl['shl'].describe()}")

        # Check for issues
        n_greater_than_1 = (df_shl['shl'] > 1.0).sum()
        n_le_zero = (df_shl['shl'] <= 0).sum()
        print(f"Values > 1.0: {n_greater_than_1}")
        print(f"Values <= 0: {n_le_zero}")

        print(f"Sample:\n{df_shl.head(20)}")

        # Convert to factor frame
        ft_shl = self.as_factor_frame(df_shl)
        print(f"Factor frame shape: {ft_shl.shape}")
        print(f"Factor frame sample:\n{ft_shl.head()}")

        print(f"Validation complete for {self.abbr}!")
        return True


if __name__ == "__main__":
    # Example for saving factors
    factor1 = RsFactor()
    print("Rs因子验证...")
    #factor1.validate()
    factor1.save(factor1.as_factor_frame(factor1.calculate()))
    # Example for validating factors
    factor2 = TesFactor()
    print("Tes因子验证...")
    factor2.validate()

    factor3 = NeiFactor()
    print("Nei因子验证...")
    factor3.validate()

    factor4 = BmjFactor()
    print("Bmj因子验证...")
    factor4.validate()

    factor5 = BmqFactor()
    print("Bmq因子验证...")
    factor5.validate()

    factor6 = SrFactor()
    print("Sr因子验证...")
    factor6.validate()

    factor7 = SgFactor()
    print("Sg因子验证...")
    factor7.validate()
    
    factor8 = CeiFactor()
    print("Cei因子验证...")
    factor8.validate()

    factor9 = CdiFactor()
    print("Cdi因子验证...")
    factor9.validate()

    factor10 = IvgFactor()
    print("Ivg因子验证...")
    factor10.validate()

    factor11 = IvcFactor()
    print("Ivc因子验证...")
    factor11.validate()
    
    factor12 = OaFactor()
    print("Oa因子验证...")
    factor12.validate()

    factor13 = TaFactor()
    print("Ta因子验证...")
    factor13.validate()
    
    factor14 = DWcFactor()
    print("DWc因子验证...")
    factor14.validate()

    factor15 = DCoaFactor()
    print("DCoa因子验证...")
    factor15.validate()

    factor16 = DColFactor()
    print("DCol因子验证...")
    factor16.validate()

    factor17 = DNcoFactor()
    print("DNco因子验证...")
    factor17.validate()

    factor18 = DncaFactor()
    print("Dnca因子验证...")
    factor18.validate()
    
    factor19 = DnclFactor()
    print("Dncl因子验证...")
    factor19.validate()
    
    factor20 = DfinFactor()
    print("Dfin因子验证...")
    factor20.validate()
    
    factor21 = DstiFactor()
    print("Dsti因子验证...")
    factor21.validate()
    
    factor22 = DltiFactor()
    print("Dlti因子验证...")
    factor22.validate()

    factor23 = DfnlFactor()
    print("Dfnl因子验证...")
    factor23.validate()
        
    factor24 = DbeFactor()
    print("Dbe因子验证...")
    factor24.validate()
    
    factor25 = PoaFactor()
    print("Poa因子验证...")
    factor25.validate()
    
    factor26 = PtaFactor()
    print("Pta因子验证...")
    factor26.validate()

    factor27 = NxfFactor()
    print("Nxf因子验证...")
    factor27.validate()
    
    factor28 = NefFactor()
    print("Nef因子验证...")
    factor28.validate()
    
    factor29 = NdfFactor()
    print("Ndf因子验证...")
    factor29.validate()

    factor30 = MeFactor()
    print("Me因子验证...")
    factor30.validate()

    factor31 = IvffFactor()
    print("Ivff因子验证...")
    factor31.validate()

    factor32 = IvcFactor()
    print("Ivc因子验证...")
    factor32.validate()
    
    factor33 = TurFactor()
    print("Tur因子验证...")
    factor33.validate()

    factor34 = DtvFactor()
    print("Dtv因子验证...")
    factor34.validate()
    
    factor35 = PpsFactor()
    print("Pps因子验证...")
    factor35.validate()

    factor36 = AmiFactor()
    print("Ami因子验证...")
    factor36.validate()

    factor37 = MdrFactor()
    print("Mdr因子验证...")
    factor37.validate()

    factor38 = TsFactor()
    print("Ts因子验证...")
    factor38.validate()

    factor39 = SrevFactor()
    print("Srev因子验证...")
    factor39.validate()

    factor40 = ShlFactor()
    print("Shl因子验证...")
    factor40.validate()