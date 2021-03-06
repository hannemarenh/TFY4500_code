import pandas as pd

if __name__ == '__main__':
    # Load and prepare csv file to dataframe
    #Load csv file. Skip lines where measurements are missing
    title = r"flair1_2808_proj.csv"
    file = r"C:\\Users\\Hanne Maren\\Documents\\Prosjektoppgave\\Data\\front_left\\" + title
    df = pd.read_csv(file, error_bad_lines=False)

    # Remove "forskyvede" rows
    # Dont know why they occur, but they do... :( Probably something with the sensor
    # For sensorTile
    check = df.iloc[:, -1].notnull()
    for i in range(0, len(check)):
        if check[i]:
            df = df.drop(i)

    # Delete nan rows
    df = df.iloc[:, : -1].dropna(axis=0)
    # If df needs to be shortened (first 5 sec should be still!!
    #df = df[18000:]
    #df = df.reset_index()

    df.to_pickle('current_df.pkl')