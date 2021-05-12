# Match IHC-labeled tiles
import pandas as pd

if __name__ == '__main__':
    ref = pd.read_csv('../align/final_summary_full.csv', header=0)
    for idx, row in ref.iterrows():
        for level in range(1, 4):
            try:
                PMS2 = pd.read_csv('../tiles/{}/level{}/{}_PMS2_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']), header=0)
                PMS2 = PMS2.drop(columns=['ratio'])
                MLH1 = pd.read_csv('../tiles/{}/level{}/{}_MLH1_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']),
                                   header=0, usecols=['Num', 'label'])
                out_dicta = PMS2.merge(MLH1, how='inner', on='Num', suffixes=['_PMS2', '_MLH1'])
                out_dicta['label'] = out_dicta['label_PMS2'] + out_dicta['label_MLH1']
                out_dicta['label'] = out_dicta['label'].clip(lower=0, upper=1)
                out_dicta = out_dicta[['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'label']]
                out_dicta.to_csv(
                    '../tiles/{}/level{}/{}_PMS2-MLH1_label.csv'.format(row['Patient_ID'], level, row['H&E_ID']),
                    index=False)
            except FileNotFoundError as e:
                print(e)

            try:
                MSH2 = pd.read_csv('../tiles/{}/level{}/{}_MSH2_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']), header=0)
                MSH2 = MSH2.drop(columns=['ratio'])
                MSH6 = pd.read_csv('../tiles/{}/level{}/{}_MSH6_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']),
                                   header=0, usecols=['Num', 'label'])
                out_dictb = MSH2.merge(MSH6, how='inner', on='Num', suffixes=['_MSH2', '_MSH6'])
                out_dictb['label'] = out_dictb['label_MSH2'] + out_dictb['label_MSH6']
                out_dictb['label'] = out_dictb['label'].clip(lower=0, upper=1)
                out_dictb = out_dictb[['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'label']]
                out_dictb.to_csv(
                    '../tiles/{}/level{}/{}_MSH2-MSH6_label.csv'.format(row['Patient_ID'], level, row['H&E_ID']),
                    index=False)
            except FileNotFoundError as e:
                print(e)

            try:
                PMS2 = pd.read_csv('../tiles/{}/level{}/{}_PMS2_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']), header=0)
                PMS2 = PMS2.drop(columns=['ratio'])
                MLH1 = pd.read_csv('../tiles/{}/level{}/{}_MLH1_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']),
                                   header=0, usecols=['Num', 'label'])
                MSH2 = pd.read_csv('../tiles/{}/level{}/{}_MSH2_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']), header=0)
                MSH2 = MSH2.drop(columns=['ratio'])
                MSH6 = pd.read_csv('../tiles/{}/level{}/{}_MSH6_label.csv'.format(row['Patient_ID'], level,
                                                                                row['H&E_ID']),
                                   header=0, usecols=['Num', 'label'])
                out_dicta = PMS2.merge(MLH1, how='inner', on='Num', suffixes=['_PMS2', '_MLH1'])
                out_dicta['label'] = out_dicta['label_PMS2'] + out_dicta['label_MLH1']
                out_dicta['label'] = out_dicta['label'].clip(lower=0, upper=1)
                out_dicta = out_dicta[['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'label']]

                out_dictb = MSH2.merge(MSH6, how='inner', on='Num', suffixes=['_MSH2', '_MSH6'])
                out_dictb['label'] = out_dictb['label_MSH2'] + out_dictb['label_MSH6']
                out_dictb['label'] = out_dictb['label'].clip(lower=0, upper=1)
                out_dictb = out_dictb[['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'label']]

                out_dict = out_dicta.merge(out_dictb, how='inner', on=['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc'],
                                           suffixes=['_a', '_b'])
                out_dict['label'] = out_dict['label_a'] + out_dict['label_b']
                out_dict['label'] = out_dict['label'].clip(lower=0, upper=1)
                out_dict = out_dict[['Num', 'X_pos', 'Y_pos', 'X', 'Y', 'Loc', 'label']]
                out_dict.to_csv(
                    '../tiles/{}/level{}/{}_PMS2-MLH1-MSH2-MSH6_label.csv'.format(row['Patient_ID'], level, row['H&E_ID']),
                    index=False)
            except FileNotFoundError as e:
                print(e)


