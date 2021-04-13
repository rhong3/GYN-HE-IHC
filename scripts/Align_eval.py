# Alignment results evaluation
import pandas as pd
import os
import numpy as np
from PIL import Image

if __name__ == '__main__':
    ref = pd.read_csv('../NYU/align_full.csv', header=0)
    aligneval = []
    for idx, row in ref.iterrows():
        PID = row['H&E_ID'].split('-')[0]
        HEID = row['H&E_ID'].split('-')[1]
        if os.path.exists("../align/{}/{}/{}".format(PID, HEID, row['IHC_ID'])):
            HEbinary = Image.open("../align/{}/{}/{}/he-b.jpg".format(PID, HEID, row['IHC_ID']))
            HEbinary = (np.array(HEbinary)[:, :, 0]/255).astype(np.uint8)
            HEsum = np.sum(HEbinary)
            IHCbinary = Image.open("../align/{}/{}/{}/ihc-b.jpg".format(PID, HEID, row['IHC_ID']))
            IHCbinary = (np.array(IHCbinary)[:, :, 0] / 255).astype(np.uint8)
            IHCsum = np.sum(IHCbinary)
            denominator = np.amin([HEsum, IHCsum])
            try:
                overlap = Image.open("../align/{}/{}/{}/overlap.jpg".format(PID, HEID, row['IHC_ID']))
                overlap = (np.array(overlap)[:, :, 0]/255).astype(np.uint8)
                overlapsum = np.sum(overlap)
                score = overlapsum/denominator
                aligneval.append([row['H&E_ID'], row['IHC_ID'], round(score, 5)])
                print(row['IHC_ID'], round(score, 5))
            except FileNotFoundError:
                pass
    alignpd = pd.DataFrame(aligneval, columns=['H&E_ID', 'IHC_ID', 'Align_score'])
    alignpd.to_csv('../align/align_eval_full.csv', header=True, index=False)

