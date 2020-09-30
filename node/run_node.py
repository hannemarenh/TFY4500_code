from node.NodeAlgorithm import NodeAlgorithm
import pandas as pd

if __name__ == '__main__':
    # Horse motion:
    title = r"flair1_2808_proj.csv"
    file = r"C:\Users\Hanne Maren\Documents\prosjektoppgave\data\front_left\\" + title

    # Load df made in current_df.py
    current_df = pd.read_pickle('current_df.pkl')

    # Run either with file or df
    leg = NodeAlgorithm(file=file)




