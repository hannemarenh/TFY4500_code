from node.NodeAlgorithm import NodeAlgorithm
import pandas as pd

if __name__ == '__main__':
    title = r"test3.csv"
    file = r"C:\Users\Hanne Maren\Documents\nivo\project\calibrationData\\" + title

    # Load df made in current_df.py
    current_df = pd.read_pickle('current_df.pkl')

    leg = NodeAlgorithm(df=current_df)


