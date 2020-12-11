# Prepare IHC slides for cutting
import pandas as pd

case = pd.read_excel('../NYU/Cases ready for Runyu.xlsx', header=0, usecols=['NYU_name', 'Group', 'IHC', 'Diagnosis'])
case.columns = ['Patient_ID', 'subtype', 'IHC', 'diagnosis']

for bat in ['1', '2', '3']:
    batch = pd.read_csv(str('../NYU/Samples_Runyu_Hong_Batch'+bat+'.csv'), header=0)
    batch.columns = ['Slide_ID', 'stain', 'num', 'file']

    aa = batch.Slide_ID.str.rsplit('-', n=1, expand=True)
    aa.columns = ['Patient_ID', 'Slide_ID']
    batch['Patient_ID'] = aa['Patient_ID']
    batch['Slide_ID'] = aa['Slide_ID']
    batch = batch[batch['stain'] == 'IHC']
    batch = batch.drop(columns=['stain', 'num'])

    aaa = case['diagnosis'].str.split(',', expand=True)
    case['histology'] = aaa[0]
    case['FIGO'] = aaa[1]

    case['subtype'] = case['subtype'].replace({'MMR-D': 'MSI', 'CNH': 'CNV-H', 'CNL': 'CNV-L'})
    combined = batch.join(case.set_index('Patient_ID'), on='Patient_ID', how='left')
    combined = combined[['Patient_ID', 'Slide_ID', 'histology', 'subtype', 'FIGO', 'file']]
    combined = combined.dropna(subset=['histology', 'file'])
    combined['file'] = combined['file'].str[:-1]
    combined.to_csv(str('../NYU/IHC'+bat+'_sum.csv'), index=False)


