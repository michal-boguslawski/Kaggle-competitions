import pathlib
import os
import pandas as pd

if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.parent
    train_file_path = os.path.join(path, "raw", "train.csv")
    test_file_path = os.path.join(path, "raw", "test.csv")
    
    y_var = "y"
    
    # load data
    train_df = pd.read_csv(train_file_path, index_col=0)
    test_df = pd.read_csv(test_file_path, index_col=0)
    
    # find mode value
    mode = train_df[y_var].mode()
    print(mode)
    
    # add prediction and write file
    test_df[y_var] = mode[0]
    test_df[[y_var, ]].to_csv(
        os.path.join(pathlib.Path(__file__).parent, "submission.csv"), 
        index=True
    )

