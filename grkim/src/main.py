import pandas as pd

from data_preprocessing import Preprocessing


class MainModel:
    def __init__(self):
        self.processing = Preprocessing()

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv("../data/bank_customer_churn_prediction.csv")

        return df

    def main(self):
        df = self.load_data()

        self.processing.fit(df)
        df = self.processing.transform(df)

        

        

