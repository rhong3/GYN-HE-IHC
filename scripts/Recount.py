# Recount tiles and summarize to csv for certain slides
import pandas as pd
import os


if __name__ == '__main__':
    ref = pd.read_csv('../NYU/IHC4_sum.csv', header=0)
    for idx, row in ref.iterrows():
        for i in range(1, 4):
            ls = list(filter(lambda ff: "_{}.png".format(row['Slide_ID']) in ff,
                             os.listdir("../tiles/{}/level{}".format(row['Patient_ID'], i))))
            outdf = pd.DataFrame(columns=["Num", "X_pos", "Y_pos", "X", "Y", "Loc"])
            outdf['Loc'] = ["../tiles/{}/level{}/".format(row['Patient_ID'], i) + itm for itm in ls]
            outdf['Num'] = range(len(ls))
            outdf['X'] = [itmx.split("x-")[1].split("-y")[0] for itmx in ls]
            outdf['Y'] = [itmy.split("y-")[1].split("_{}.png".format(row['Slide_ID']))[0] for itmy in ls]
            outdf['X_pos'] = pd.to_numeric(outdf['X']) / (500 * (2 ** (i - 1)))
            outdf['Y_pos'] = pd.to_numeric(outdf['Y']) / (500 * (2 ** (i - 1)))
            outdf.to_csv("../tiles/{}/level{}/{}_dict.csv".format(row['Patient_ID'], i, row['Slide_ID']), index=False)

