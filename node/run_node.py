from node.NodeAlgorithm import NodeAlgorithm
import pandas as pd

if __name__ == '__main__':
    title = r"pendel_30cm.csv"
    file = r"C:\Users\Hanne Maren\Documents\prosjektoppgave\data\control\\" + title

    # Load df made in current_df.py
    current_df = pd.read_pickle('current_df.pkl')

    # Run either with file or df
    leg = NodeAlgorithm(file=file)


