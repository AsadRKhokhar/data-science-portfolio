# data_prep.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class StoreSalesDataPrep:
    def __init__(self, base_path='store-sales-time-series-forecasting/'):
        self.base_path = base_path
        self.le_family = None
        print(f"[INFO] StoreSalesDataPrep initialized with base_path: {self.base_path}")
    
    def set_seed(seed=42):
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        print(f"[INFO] Random seed set to {seed} for reproducibility.")

    
    def load_raw_data(self):
        print("[INFO] Loading raw CSV files...")
        train_df = pd.read_csv(self.base_path + 'train.csv')
        test_df = pd.read_csv(self.base_path + 'test.csv')
        stores_df = pd.read_csv(self.base_path + 'stores.csv')
        transactions_df = pd.read_csv(self.base_path + 'transactions.csv')
        oil_df = pd.read_csv(self.base_path + 'oil.csv')
        holidays_events_df = pd.read_csv(self.base_path + 'holidays_events.csv')
        print("[INFO] Raw data loaded successfully.")
        return train_df, test_df, stores_df, transactions_df, oil_df, holidays_events_df

    def preprocess_df_dates(self, *dfs, date_cols=None):
        if date_cols is None:
            date_cols = ['date']
        for i, df in enumerate(dfs):
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            print(f"[INFO] Converted date columns in dataframe {i+1}")
    
    def fill_oil_prices(self, oil_df):
        oil_df['dcoilwtico'] = oil_df['dcoilwtico'].ffill()
        oil_df['dcoilwtico'] = oil_df['dcoilwtico'].bfill()
        print("[INFO] Missing oil prices filled (forward/backward fill).")
        return oil_df

    def merge_and_process(self, df, holidays_events_df, oil_df, transactions_df, stores_df):
        print("[INFO] Merging external datasets...")
        df = df.merge(
            holidays_events_df[['date', 'type', 'locale', 'locale_name', 'description', 'transferred']],
            on='date', how='left'
        )
        df = df.merge(oil_df[['date', 'dcoilwtico']], on='date', how='left')
        df = df.merge(transactions_df, on=['date', 'store_nbr'], how='left')
        df = df.merge(stores_df, on='store_nbr', how='left')

        df['transactions'] = df['transactions'].fillna(0)
        df['dcoilwtico'] = df['dcoilwtico'].ffill()
        df['dcoilwtico'] = df['dcoilwtico'].bfill()

        cat_cols_to_fill = ['type', 'locale', 'locale_name', 'description', 'city', 'state', 'type_y']
        for col in cat_cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        categorical_cols = ['type', 'locale', 'locale_name', 'description', 'store_nbr', 'city', 'state', 'type_y']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        print("[INFO] Merge and preprocessing complete.")
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

        self.le_family = LabelEncoder()
        df['family_enc'] = self.le_family.fit_transform(df['family'])
        print("[INFO] Label encoding applied to 'family' column.")

        if 'sales' in df.columns and 'id' in df.columns:
            df = df.sort_values(['id', 'date'])
            df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
            df['sales_roll_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())
            df['sales_lag_7'] = df['sales_lag_7'].fillna(0)
            df['sales_roll_mean_7'] = df['sales_roll_mean_7'].fillna(0)
            print("[INFO] Lag and rolling features created for sales.")

        print("[INFO] Feature engineering complete.")
        return df

    def preprocess_train(self, train_df, holidays_events_df, oil_df, transactions_df, stores_df):
        print("[INFO] Preprocessing training data...")
        self.preprocess_df_dates(train_df, holidays_events_df, oil_df, transactions_df)
        oil_df = self.fill_oil_prices(oil_df)
        train_df = self.merge_and_process(train_df, holidays_events_df, oil_df, transactions_df, stores_df)
        train_df = self.feature_engineering(train_df, holidays_events_df)
        print("[INFO] Training data preprocessing complete.")
        return train_df

    def preprocess_test(self, test_df, holidays_events_df, oil_df, transactions_df, stores_df):
        print("[INFO] Preprocessing test data...")
        self.preprocess_df_dates(test_df, holidays_events_df, oil_df, transactions_df)
        oil_df = self.fill_oil_prices(oil_df)
        test_df = self.merge_and_process(test_df, holidays_events_df, oil_df, transactions_df, stores_df)

        test_df['year'] = test_df['date'].dt.year
        test_df['month'] = test_df['date'].dt.month
        test_df['day'] = test_df['date'].dt.day
        test_df['dayofweek'] = test_df['date'].dt.dayofweek
        test_df['weekofyear'] = test_df['date'].dt.isocalendar().week.astype(int)
        test_df['is_weekend'] = test_df['dayofweek'].isin([5, 6]).astype(int)

        holiday_dates = holidays_events_df['date'].unique()
        test_df['is_holiday'] = test_df['date'].isin(holiday_dates).astype(int)

        test_df['family_enc'] = self.le_family.transform(test_df['family'])
        print("[INFO] Test data preprocessing complete.")
        return test_df
