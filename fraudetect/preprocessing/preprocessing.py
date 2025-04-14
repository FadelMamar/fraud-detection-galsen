from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self,windows_size_in_days:list[int]=[1,7,30],
                 uid_cols:list=None,
                 session_gap_minutes:int=30,
                 n_clusters:int=8,
                 ):
        self.account_stats = None
        self.product_fraud_rate = None
        self.customer_stats = None
        self.windows_size_in_days=windows_size_in_days
        self.uid_cols = None
        self.behavioral_drift_cols = ['AccountId',]
        self.session_gap_minutes = session_gap_minutes
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self, df):
        self.account_stats = df.groupby('AccountId')['TX_AMOUNT'].agg(['mean', 'std']).rename(columns={'mean': 'AccountMeanAmt', 
                                                                                                       'std': 'AccountStdAmt'})
        self.customer_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(['mean', 'std']).rename(columns={'mean': 'CustomerMeanAmt', 
                                                                                                          'std': 'CustomerStdAmt'})
        
        self.product_fraud_rate = df.groupby('ProductId')['TX_FRAUD'].mean().rename('ProductFraudRate')
        self.provider_fraud_rate = df.groupby('ProviderId')['TX_FRAUD'].mean().rename('ProviderFraudRate')
        self.channel_fraud_rate = df.groupby('ChannelId')['TX_FRAUD'].mean().rename('ChannelIdFraudRate')
        
        #-- Compute clusters
        cluster_data = df.groupby('CUSTOMER_ID').agg({
            'Amount': 'mean',
            'ChannelId': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        }).fillna(0)

        # Encode channel as numeric for clustering
        cluster_data['ChannelId'] = cluster_data['ChannelId'].astype('category').cat.codes

        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=41,max_iter=500,batch_size=1024)
        self.kmeans.fit(cluster_data)

        self.customer_cluster_labels = pd.DataFrame({
            'CUSTOMER_ID': cluster_data.index,
            'CustomerCluster': self.kmeans.labels_
        })
        
    def transform(self, df):
        df = df.copy()
        df = df.sort_values(by=['AccountId', 'TX_DATETIME'])
        
        if self.uid_cols is not None:
            df = self._create_unique_identifier(df)

        df = self._add_temporal_features(df)
        df = self._add_account_stats(df)
        df = self.__add_customer_stats(df)
        df = self._compute_behavioral_drift(df)
        df = self._compute_batch_gap_features(df)
        df = self._compute_avg_txn_features(df)
        df = self._add_categorical_cross_features(df)
        df = self._add_temporal_identity_interactions(df)
        df = self._add_frequency_features(df)
        df = self._add_fraud_rate_features(df)
        df = self._cleanup(df)

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    # ---------- Private Helper Methods ----------
    def _create_unique_identifier(self,df:pd.DataFrame):
        
        df["Customer_UID"] = df[self.uid_cols].apply(
            lambda x: "+".join(x), axis=1, raw=False, result_type="reduce"
        )
        
        return df
        
    def _add_temporal_features(self, df):
        df['Hour'] = df['TX_DATETIME'].dt.hour
        df['DayOfWeek'] = df['TX_DATETIME'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['IsNight'] = df['Hour'].between(0, 6).astype(int)
        
        for col in ['AccountId','CustomerId']:

            df[col+'_TimeSinceLastTxn'] = df.groupby(col)['TX_DATETIME'].diff().dt.total_seconds() / 60
    
            df[col+'_Txn1hCount'] = (
                df.set_index('TX_DATETIME')
                  .groupby(col)['TRANSACTION_ID']
                  .rolling('1h').count()
                  .reset_index(level=0, drop=True)
            )
            
            for day in self.windows_size_in_days:
                df[f'{col}_AvgAmount_{day}day'] = (
                    df.groupby(col)['TX_AMOUNT']
                      .rolling(window=day, min_periods=1).mean()
                      .reset_index(level=0, drop=True)
                )
        
        
        return df

    def _add_account_stats(self, df):
        df = df.merge(self.account_stats, on='AccountId', how='left')
        df['AccountAmountZScore'] = (df['TX_AMOUNT'] - df['AccountMeanAmt']) / df['AccountStdAmt'].replace(0, 1)
        df['AccountAmountOverAvg'] = df['TX_AMOUNT'] / df['AccountMeanAmt'].replace(0, 1)
        return df
    
    def _add_customer_stats(self, df):
        df = df.merge(self.account_stats, on='CUSTOMER_ID', how='left')
        df['CustomerAmountZScore'] = (df['TX_AMOUNT'] - df['CustomerMeanAmt']) / df['CustomerStdAmt'].replace(0, 1)
        df['CustomerAmountOverAvg'] = df['TX_AMOUNT'] / df['AccountMeanAmt'].replace(0, 1)
        return df

    def _add_categorical_cross_features(self, df):
        df['Channel_ProductCategory'] = df['ChannelId'].astype(str) + "_" + df['ProductCategory'].astype(str)
        df['ProductCategory_Account'] = df['ProductCategory'].astype(str) + "_" + df['AccountId'].astype(str)
        df['ProductCategory_Customer'] = df['ProductCategory'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df['Country_Currency'] = df['CountryCode'].astype(str) + "_" + df['CurrencyCode'].astype(str)
        df['Channel_PricingStrategy'] = df['ChannelId'].astype(str) + "_" + df['PricingStrategy'].astype(str)
        df['Provider_Product'] = df['ProviderId'].astype(str) + "_" + df['ProductId'].astype(str)
        return df

    def _add_temporal_identity_interactions(self, df):
        df['IsNight_Android'] = df['IsNight'].astype(str) + df['ChannelId'].astype(str)
        df['Weekend_Channel'] = df['IsWeekend'].astype(str) + df['ChannelId'].astype(str)
        df['Hour_Channel'] = df['Hour'].astype(str) + "_" + df['ChannelId'].astype(str)
        df['Hour_Account'] = df['Hour'].astype(str) + "_" + df['AccountId'].astype(str)
        df['Hour_Customer'] = df['Hour'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df['DayOfWeek_Account'] = df['DayOfWeek'].astype(str) + "_" + df['AccountId'].astype(str)
        df['DayOfWeek_Customer'] = df['DayOfWeek'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df['Country_Hour'] = df['CountryCode'].astype(str) + "_" + df['Hour'].astype(str)
        return df

    def _add_frequency_features(self, df):
        df['TxnDate'] = df['TX_DATETIME'].dt.date
        txn_freq = df.groupby(['AccountId', 'TxnDate'])['TRANSACTION_ID'].count().rename('DailyAccountTxnCount')
        df = df.merge(txn_freq, on=['AccountId', 'TxnDate'])
        return df

    def _add_fraud_rate_features(self, df):
        df = df.merge(self.product_fraud_rate, on='ProductId', how='left')
        df = df.merge(self.provider_fraud_rate, on='ProviderId', how='left')
        df = df.merge(self.channel_fraud_rate, on='ChannelId', how='left')
        return df
    
    def _compute_behavioral_drift(self, df):
        df.set_index('TX_DATETIME', inplace=True)
        
        for col in self.behavioral_drift_cols:
            val_7d = df.groupby(col)['TX_AMOUNT'].transform(lambda x: x.rolling('7d').mean())
            val_30d = df.groupby(col)['TX_AMOUNT'].transform(lambda x: x.rolling('30d').mean())
            df[col+'_RatioTo7dAvg'] = df['TX_AMOUNT'] / val_7d
            df[col+'_RatioTo30dAvg'] = df['TX_AMOUNT'] / val_30d
            df[col+'_ZScore_7d'] = (df['TX_AMOUNT'] - val_7d) / df.groupby(col)['TX_AMOUNT'].transform(lambda x: x.rolling('7d').std())
            df[col+'_ZScore_30d'] = (df['TX_AMOUNT'] - val_30d) / df.groupby(col)['TX_AMOUNT'].transform(lambda x: x.rolling('30d').std())            
            
        df.reset_index(inplace=True)
        return df
    
    def _compute_avg_txn_features(self, df):
        
        for col in self.behavioral_drift_cols:
            df[f'{col}_MovingAvg5'] = (
                df.groupby(col)['TX_AMOUNT']
                .rolling(window=5, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            long_term_avg = df.groupby(col)['TX_AMOUNT'].agg(['mean']).rename(columns={'mean': f'{col}_LongTermAvg'})
            df = df.merge(long_term_avg, on=col, how='left')
            df['PctChangeFromAvg'] = (df[f'{col}_MovingAvg5'] - df[f'{col}_LongTermAvg']) / df[f'{col}_LongTermAvg']
        return df
    
    def _compute_batch_gap_features(self, df):
        df = df.sort_values(by=['BatchId', 'TX_DATETIME'])
        batch_time = df.groupby('BatchId')['TX_DATETIME'].min().sort_values()
        batch_time_gap = batch_time.diff().dt.total_seconds().rename('TimeBetweenBatches')
        txn_per_batch = df.groupby('BatchId')['TransactionId'].count().rename('TxnPerBatch')
        df = df.merge(txn_per_batch, on='BatchId', how='left')
        df = df.merge(batch_time_gap, left_on='BatchId', right_index=True, how='left')
        return df
    
    def _compute_session_features(self, df):
        df = df.sort_values(by=['AccountId', 'TX_DATETIME'])
        df['TimeDiff'] = df.groupby('AccountId')['TX_DATETIME'].diff().dt.total_seconds().div(60)
        df['NewSession'] = (df['TimeDiff'] > self.session_gap_minutes).fillna(True)
        df['SessionId'] = df.groupby('AccountId')['NewSession'].cumsum()
        session_stats = df.groupby(['AccountId', 'SessionId']).agg(
            SessionTxnCount=('TRANSACTION_ID', 'count'),
            SessionValue=('TX_AMOUNT', 'sum'),
            SessionDuration=('TX_DATETIME', lambda x: (x.max() - x.min()).total_seconds() / 60),
            SessionChannel=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        ).reset_index()
        df = df.merge(session_stats, on=['AccountId', 'SessionId'], how='left')
        return df
    
    def _add_customer_clusters(self, df):
        df = df.merge(self.customer_cluster_labels, on='AccountId', how='left')
        df['ClusterChannelInteraction'] = df['CustomerCluster'].astype(str) + '_' + df['ChannelId'].astype(str)
        df['ClusterChannelInteraction'] = df['ClusterChannelInteraction'].astype('category').cat.codes
        return df
    
    def _cleanup(self, df):
        
        df.drop(columns=['AccountMeanAmt', 'AccountStdAmt','CustomerMeanAmt', 'CustomerStdAmt', 'TxnDate'], 
                inplace=True)
        
        return df



