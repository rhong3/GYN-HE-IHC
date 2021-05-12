"""
Prepare training and testing datasets as CSV dictionaries

Created on 11/26/2018

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re


# get all full paths of images
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv', 'all.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


# Get intersection of 2 lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def tile_ids_in(slide, level, root_dir, label):
    ids = []
    try:
        for id in os.listdir(root_dir):
            if '.png' in id.split("_")[-1] and len(id.split("_")[-1]) < 7:
                ids.append([slide, level, root_dir+'/'+id, label])

    except FileNotFoundError:
        print('Ignore:', root_dir)

    return ids


def ihc_tile_ids_in(ihcl, slide, level, root_dir, label):
    if os.path.isdir(root_dir):
        feature = ihcl[0]
        ih = ihcl[1]
        if list(filter(lambda file: '{}_label.csv'.format(ih) in file, os.listdir(root_dir))):
            print('{}/{} mapping exists.'.format(slide, ih))
            prpd = pd.DataFrame(columns=['slide', 'level', 'path', 'label'])
            for ff in list(filter(lambda file: '{}_label.csv'.format(ih) in file, os.listdir(root_dir))):
                la = pd.read_csv(root_dir + '/' + ff, header=0, usecols=['Loc', 'label'])
                if feature == "MSI":
                    la['label'] = np.abs(1 - la['label'])
                la['level'] = level
                la = la.rename(index=str, columns={"Loc": "path"})
                la['slide'] = slide
                la = la[la['label'] == label]
                la = la.dropna()
                prpd = prpd.append(la)
            prpd = sku.shuffle(prpd)
        else:
            print('{}/{} mapping does not exist.'.format(slide, ih))
            pr = tile_ids_in(slide, level, root_dir, label)
            prpd = pd.DataFrame(pr, columns=['slide', 'level', 'path', 'label'])
    else:
        prpd = pd.DataFrame(columns=['slide', 'level', 'path', 'label'])
    return prpd


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(slide, label, root_dir, age=None, BMI=None):
    dira = os.path.isdir(root_dir + 'level1')
    dirb = os.path.isdir(root_dir + 'level2')
    dirc = os.path.isdir(root_dir + 'level3')
    if dira and dirb and dirc:
        if "TCGA" in root_dir:
            fac = 2000
        else:
            fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id.split("_")[-1] and len(id.split("_")[-1]) < 7:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        dup = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                    except IndexError:
                        dup = np.nan
                    ids.append([slide, label, level, dirr + '/' + id, x, y, dup])
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['slide', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['slide', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
        idsa['age'] = age
        idsa['BMI'] = BMI
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])

    return idsa


# pair tiles of 10x, 5x, 2.5x of the same area for IHC-labeled tiles
def IHC_paired_tile_ids_in(ihcl, slide, label, root_dir, age=None, BMI=None):
    if os.path.isdir(root_dir + 'level1') and os.path.isdir(root_dir + 'level2') and os.path.isdir(root_dir + 'level3'):
        feature = ihcl[0]
        ih = ihcl[1]
        if list(filter(lambda file: '{}_label.csv'.format(ih) in file, os.listdir(root_dir + 'level1'))):
            print('{}/{} mapping exists.'.format(slide, ih))
            fac = 1000
            prpd = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
            for ff in list(filter(lambda file: '{}_label.csv'.format(ih) in file, os.listdir(root_dir + 'level1'))):
                try:
                    la = pd.read_csv(root_dir + 'level1/' + ff, header=0, usecols=['X', 'Y', 'Loc', 'label'])
                    if feature == "MSI":
                        la['label'] = np.abs(1-la['label'])
                    la['level'] = 1
                    la['X'] = la['X'] / fac
                    la['X'] = la['X'].round(0)
                    la['Y'] = la['Y'] / fac
                    la['Y'] = la['Y'].round(0)
                    la = la.rename(index=str, columns={"Loc": "L0path", "label": "L0label"})
                    lb = pd.read_csv(root_dir + 'level2/' + ff, header=0, usecols=['X', 'Y', 'Loc', 'label'])
                    if feature == "MSI":
                        lb['label'] = np.abs(1-lb['label'])
                    lb['level'] = 2
                    lb['X'] = lb['X'] / fac
                    lb['X'] = lb['X'].round(0)
                    lb['Y'] = lb['Y'] / fac
                    lb['Y'] = lb['Y'].round(0)
                    lb = lb.rename(index=str, columns={"Loc": "L1path", "label": "L1label"})
                    lc = pd.read_csv(root_dir + 'level3/' + ff, header=0, usecols=['X', 'Y', 'Loc', 'label'])
                    if feature == "MSI":
                        lc['label'] = np.abs(1-lc['label'])
                    lc['level'] = 3
                    lc['X'] = lc['X'] / fac
                    lc['X'] = lc['X'].round(0)
                    lc['Y'] = lc['Y'] / fac
                    lc['Y'] = lc['Y'].round(0)
                    lc = lc.rename(index=str, columns={"Loc": "L2path", "label": "L2label"})

                    ll = pd.merge(la, lb, on=['X', 'Y'], how='left', validate="many_to_many")
                    ll['X'] = ll['X'] - (ll['X'] % 2)
                    ll['Y'] = ll['Y'] - (ll['Y'] % 2)
                    ll = pd.merge(ll, lc, on=['X', 'Y'], how='left', validate="many_to_many")
                    if label == 0:
                        ll['label'] = ll[['L0label', 'L1label', 'L2label']].min(axis=1)
                        ll = ll[ll['label'] == 0]
                    else:
                        ll['label'] = ll[['L0label', 'L1label', 'L2label']].max(axis=1)
                        ll = ll[ll['label'] == 1]
                    ll = ll.drop(columns=['X', 'Y', 'L0label', 'L1label', 'L2label'])
                    ll['slide'] = slide
                    ll = ll.dropna()
                    ll['age'] = age
                    ll['BMI'] = BMI
                    prpd = prpd.append(ll)
                except Exception as e:
                    print(e)
                    pass
            prpd = sku.shuffle(prpd)
        else:
            print('{}/{} mapping does not exist.'.format(slide, ih))
            prpd = paired_tile_ids_in(slide, label, root_dir, age=age, BMI=BMI)
    else:
        prpd = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])

    return prpd

