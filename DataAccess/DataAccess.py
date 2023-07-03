import pandas as pd
import os

class Writer:
    def __init__(self, data):
        self.data = data
        self.csv_path = "data.csv"

    def write_to_csv(self, mode='a'):
        column_names = ["mouth_pos_x", "mouth_pos_y", "thumb_x", "thumb_y", "index_x", "index_y", "middle_x", 
                        "middle_y", "ring_x", "ring_y", "little_x", "little_y", "is_biting"]
        df = pd.DataFrame(data=self.data, columns=column_names)
        df.to_csv(self.csv_path, mode=mode, index=False, header=not Rules.check_if_csv_exist(self.csv_path))
        return df


class Reader:
    def __init__(self):
        self.csv_path = "data.csv"

    def read_from_csv(self):
        df = pd.read_csv(self.csv_path)
        df = pd.DataFrame(df)
        return df
    
    def read_with_parameters(self):
        df = pd.read_csv(self.csv_path)
        df = pd.DataFrame(df)

        train_dataset = df.sample(frac=0.8)
        validation_dataset = df.drop(train_dataset.index)
        test_dataset = validation_dataset.sample(frac=0.5)
        validation_dataset = validation_dataset.drop(test_dataset.index)

        return train_dataset, validation_dataset, test_dataset

    def drop(self):
        df = self.read_from_csv()
        df.drop(labels=51, axis=0)
        return df

class Rules:
    def check_if_csv_exist(path):
        if os.path.exists(path):
            return True
        else:
            return False
        
    
if __name__ == "__main__":
    reader = Reader()
    df = reader.drop()
    print(df)
