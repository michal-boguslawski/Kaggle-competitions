import pathlib
import os
import pandas as pd

path = pathlib.Path(__file__).parent.parent
file_path = os.path.join(path, "raw", "test.csv")
if __name__ == "__main__":
    test_df = pd.read_csv(file_path)
    test_df["Personality"] = "Extrovert"
    test_df[["id", "Personality"]].to_csv("submission.csv", index=False)
