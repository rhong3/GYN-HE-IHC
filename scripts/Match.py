# Match IHC-labeled tiles
import pandas as pd

if __name__ == '__main__':
    ref = pd.read_csv('../align/final_summary.csv', header=0)
    for idx, row in ref.iterrows():
        IHC = row['IHC_ID'].split('-')[-1]
        for level in range(1, 4):
            tile_dict = pd.read_csv('../tiles/{}/level{}/{}_dict.csv'.format(row['Patient_ID'], level, row['H&E_ID']),
                                    header=0)
            label_dict = pd.read_csv('../autolabel/{}/{}/{}/ratio_level{}.csv'.format(row['Patient_ID'], row['H&E_ID'],
                                                                                      row['IHC_ID'], level), header=0,
                                     usecols=['abs_x', 'abs_y', 'ratio', 'label'])
            out_dict = pd.merge(tile_dict, label_dict, how='left', left_on=['X', 'Y'], right_on=['abs_x', 'abs_y'])
            out_dict = out_dict[['Num',	'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'ratio', 'label']]
            out_dict.to_csv('../tiles/{}/level{}/{}_{}_label.csv'.format(row['Patient_ID'], level, row['H&E_ID'], IHC),
                            index=False)


