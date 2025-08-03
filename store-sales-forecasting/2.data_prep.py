# data_prep.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class StoreSalesDataPrep:
    def __init__(self, base_path='store-sales-time-series-forecasting'):
        self.base_path = os.path.join(base_path, '')
        self.le_family = None
        print(f"[INFO] StoreSalesDataPrep initialized with base_path: {self.base_path}")
    
    def set_seed(self, seed=42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        print(f"[INFO] Random seed set to {seed} for reproducibility.")

    def load_raw_data(self):
        print("[INFO] Loading raw CSV files...")
        train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.base_path, 'test.csv'))
        stores_df = pd.read_csv(os.path.join(self.base_path, 'stores.csv'))
        transactions_df = pd.read_csv(os.path.join(self.base_path, 'transactions.csv'))
        oil_df = pd.read_csv(os.path.join(self.base_path, 'oil.csv'))
        holidays_events_df = pd.read_csv(os.path.join(self.base_path, 'holidays_events.csv'))
        print("[INFO] Raw data loaded successfully.")
        return train_df, test_df, stores_df, transactions_df, oil_df, holidays_events_df

    def preprocess_df_dates(self, *dfs, date_cols=None):
        if date_cols is None:
            date_cols = ['date']
        processed = []
        for i, df in enumerate(dfs):
            df = df.copy()
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            processed.append(df)
            print(f"[INFO] Converted date columns in dataframe {i+1}")
        return processed

    def fill_oil_prices(self, oil_df):
        oil_df['dcoilwtico'] = oil_df['dcoilwtico'].ffill().bfill()
        print("[INFO] Missing oil prices filled (forward/backward fill).")
        return oil_df

    def merge_external_data(self, df, holidays_events_df, oil_df, transactions_df, stores_df):
        holidays_filtered = holidays_events_df[
            (holidays_events_df['locale'] == 'National') &
            (holidays_events_df['transferred'] == False)
        ].drop_duplicates('date').rename(columns={
            'type': 'holiday_type',
            'locale': 'holiday_locale',
            'locale_name': 'holiday_locale_name',
            'description': 'holiday_description',
            'transferred': 'holiday_transferred'
        })

        transactions_df = transactions_df.drop_duplicates(subset=['date', 'store_nbr'])
        stores_df = stores_df.rename(columns={'type': 'store_type'})

        df = df.merge(holidays_filtered, on='date', how='left')
        df = df.merge(oil_df[['date', 'dcoilwtico']], on='date', how='left')
        df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')
        df = df.merge(stores_df, on='store_nbr', how='left')

        df = df.sort_values(['store_nbr', 'date'])
        df['transactions'] = df['transactions'].ffill().bfill()
        df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

        return df

    def feature_engineering(self, df, holidays_events_df):
        print("[INFO] Starting feature engineering...")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        holiday_dates = holidays_events_df['date'].unique()
        df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)

        if 'family' in df.columns:
            self.le_family = LabelEncoder()
            df['family_enc'] = self.le_family.fit_transform(df['family'])
            print("[INFO] Label encoding applied to 'family' column.")

        if 'sales' in df.columns and 'id' in df.columns:
            df = df.sort_values(['id', 'date'])
            df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
            df['sales_roll_mean_7'] = df.groupby('id')['sales'].transform(
                lambda x: x.shift(1).rolling(window=7).mean()
            )
            df['sales_lag_7'] = df['sales_lag_7'].fillna(0)
            df['sales_roll_mean_7'] = df['sales_roll_mean_7'].fillna(0)
            print("[INFO] Lag and rolling features created for sales.")

        print("[INFO] Feature engineering complete.")
        return df

    def preprocess_train(self, train_df, holidays_events_df, oil_df, transactions_df, stores_df):
        print("[INFO] Preprocessing training data...")
        train_df, holidays_events_df, oil_df, transactions_df = self.preprocess_df_dates(
            train_df, holidays_events_df, oil_df, transactions_df
        )
        oil_df = self.fill_oil_prices(oil_df)
        train_df = self.merge_external_data(train_df, holidays_events_df, oil_df, transactions_df, stores_df)
        train_df = self.feature_engineering(train_df, holidays_events_df)
        print("[INFO] Training data preprocessing complete.")
        return train_df

    def preprocess_test(self, test_df, holidays_events_df, oil_df, transactions_df, stores_df):
        print("[INFO] Preprocessing test data...")
        test_df, holidays_events_df, oil_df, transactions_df = self.preprocess_df_dates(
            test_df, holidays_events_df, oil_df, transactions_df
        )
        oil_df = self.fill_oil_prices(oil_df)
        test_df = self.merge_external_data(test_df, holidays_events_df, oil_df, transactions_df, stores_df)

        test_df['year'] = test_df['date'].dt.year
        test_df['month'] = test_df['date'].dt.month
        test_df['day'] = test_df['date'].dt.day
        test_df['dayofweek'] = test_df['date'].dt.dayofweek
        test_df['weekofyear'] = test_df['date'].dt.isocalendar().week.astype(int)
        test_df['is_weekend'] = test_df['dayofweek'].isin([5, 6]).astype(int)

        holiday_dates = holidays_events_df['date'].unique()
        test_df['is_holiday'] = test_df['date'].isin(holiday_dates).astype(int)

        if 'family' in test_df.columns and self.le_family:
            test_df['family_enc'] = self.le_family.transform(test_df['family'])
        else:
            test_df['family_enc'] = -1  # Fallback for missing encoder or column

        print("[INFO] Test data preprocessing complete.")
        return test_df
