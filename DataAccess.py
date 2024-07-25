import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataAccess:

    def __init__(self):
        self.csv_path = "data.csv"
        self.column_names = ['thumb_x', 'thumb_y', 'index_x', 'index_y', 'middle_x', 
                           'middle_y', 'ring_x', 'ring_y', 'little_x', 'little_y', 'is_biting']
        

    def write_to_csv(self, data):
        df = pd.DataFrame(data=[data], columns=self.column_names)

        mode,header,message = 'w', True, "data.csv created"  
        if self.check_if_csv_exist(self.csv_path):
            mode='a'
            header=False
            message = "data.csv updated"
        df.to_csv(self.csv_path, mode=mode, index=False, header=header)
        return message

    def read_from_csv(self):
        df = pd.read_csv(self.csv_path)
        df = pd.DataFrame(df)
        return df
    
    def read_with_parameters(self):
        df = self.read_from_csv()

        X = df.drop(['is_biting'], axis=1) 
        y = df.pop('is_biting')

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
        # train_dataset = df.sample(frac=0.8)
        # validation_dataset = df.drop(train_dataset.index)
        # test_dataset = validation_dataset.sample(frac=0.5)
        # validation_dataset = validation_dataset.drop(test_dataset.index)

        # train_features = train_dataset.copy()
        # validation_features = validation_dataset.copy()
        # test_features = test_dataset.copy()

        # train_labels = train_features.pop('is_biting')
        # validation_labels = validation_features.pop('is_biting')
        # test_labels = test_features.pop('is_biting')

        # return train_features, train_labels, validation_features, validation_labels, test_features, test_labels
    
    def delete(self, args):
        df = self.read_from_csv()
        df.drop(axis=0, inplace=True, index=args)
        # df.reset_index(drop=True)
        return df

    def check_if_csv_exist(self,path):
        if os.path.exists(path):
            return True
        else:
            return False
    
if __name__ == "__main__":
    d_a = DataAccess()
    print(d_a.read_with_parameters())
    # df = d_a.read_from_csv()
    # df.drop(index=[11,12,13], axis=0, inplace=True)
    # df.to_csv("data.csv", mode='w', index=)
    # values = input("which row you want to drop?")
    # rows_to_drop = [int(index.strip()) for index in values.split()]
    # df = d_a.delete(rows_to_drop)
    # df = df[~df.index.isin(rows_to_drop)]
    # result = d_a.write_to_csv(df)
    # print(result)
    # d_a.write_to_csv(df)
