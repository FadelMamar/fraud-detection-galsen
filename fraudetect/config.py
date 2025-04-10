from dataclasses import dataclass

COLUMNS_TO_DROP = ['CurrencyCode','CountryCode','SubscriptionId','BatchId','CUSTOMER_ID','AccountId','TRANSACTION_ID','TX_DATETIME','TX_TIME_DAYS']
COLUMNS_TO_ONE_HOT_ENCODE = ['PricingStrategy','ChannelId', 'ProductCategory', 'ProviderId', 'ProductId']
COLUMNS_TO_SCALE = ['TX_DURING_WEEKEND','TX_DURING_NIGHT','TX_AMOUNT','Value'] # or None to select all numeric columns

@dataclass
class Arguments:


    # data pre-processing
    

    # training parameters
    max_epochs:int
    learning_rate:float
    weightdecay:float

    # data augmentation



