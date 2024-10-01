import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataAccess:
    def __init__(self, name=None):
        self.csv_path = name
        self.column_names = ['thumb_x', 'thumb_y', 'index_x', 'index_y', 'middle_x', 
                        'middle_y', 'ring_x', 'ring_y', 'little_x', 'little_y', 'is_biting']
        

    def write_to_csv(self, data):
        df = pd.DataFrame(data=[data], columns=self.column_names)

        mode,header,message = 'w', True, "csv file created"  
        if self.check_if_csv_exist(self.csv_path):
            mode='a'
            header=False
            message = "csv file updated"
        df.to_csv(self.csv_path, mode=mode, index=False, header=header)
        return message

    def combine(self, train_path="train.csv", test_path="test.csv", val_path="validation.csv", output_path="final_data.csv"):
        # Read all three datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        val_df = pd.read_csv(val_path)
        
        # Concatenate the DataFrames
        combined_df = pd.concat([train_df, test_df, val_df], ignore_index=True)
        
        # Shuffle the dataset (optional)
        # combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        # Write combined data to a new CSV
        combined_df.to_csv(output_path, mode='w', index=False)
        print(f"Combined CSV saved to {output_path}")
        
        return combined_df
    
    def read_csv(self, data_path="final_data.csv"):
        # Read the final combined CSV file
        if self.check_if_csv_exist(data_path):
            final_df = pd.read_csv(data_path)
            print(f"Final data loaded from {data_path}")
            return final_df
        else:
            print(f"{data_path} does not exist.")
            return None

    def split_final_data(self, final_data_path="final_data.csv", test_size=0.2):
        # Read the final combined CSV
        final_df = self.read_csv(final_data_path)
        
        if final_df is not None:
            # Separate the features (X) and labels (y)
            X = final_df.drop('is_biting', axis=1)  # Features: everything except 'is_biting'
            y = final_df['is_biting']  # Label: the 'is_biting' column
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
        else:
            print("Final data could not be read.")
            return None, None, None, None

    def check_if_csv_exist(self,path):
        if os.path.exists(path):
            return True
        else:
            return False

if __name__ == "__main__":
    d_a = DataAccess()
    a, b, c, d = d_a.split_final_data()
    print(a, b, c, d)
