#!/usr/bin/env python
"""Prepare model inputs from the unpartitioned H5AD data archive."""
import argparse,hashlib,json,shutil,random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from Prepare_data import build_program_routes


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def prepare(release_data,shared_resources,dataset,partition,cells,output_root):
    release_data=Path(release_data).resolve();shared_resources=Path(shared_resources).resolve()
    output_root=Path(output_root).resolve();dataset_dir=output_root/'datasets'/dataset
    if dataset_dir.exists() and any(dataset_dir.iterdir()):raise FileExistsError(dataset_dir)
    source=release_data/dataset/f'{dataset}_processed.h5ad'
    workbook=release_data/'Data1.xlsx'
    assignments=pd.read_excel(workbook,sheet_name=dataset,dtype={'donor_id':str})
    role_columns = {int(str(name).split('_')[-2]): name for name in assignments.columns
                    if str(name).endswith('_split') and str(name).split('_')[-2].isdigit()}
    column = role_columns[partition]
    splits=assignments[['donor_id','age_years',column]].rename(columns={column:'split'})
    for key in ['sex','disease_type']:
        if key in assignments:splits[key]=assignments[key]
    assert splits.donor_id.is_unique and set(splits.split)=={'train','val','test'}
    data=ad.read_h5ad(source,backed='r')
    expected=data.uns['benchmark_data']['vocabulary_sha256']
    assert digest(shared_resources/'human_gene_vocab.json')==expected,'Mismatched gene vocabulary'
    metadata=data.obs[['cell_index','donor_id','age_years','cell_type','cell_type_id']].copy()
    metadata.donor_id=metadata.donor_id.astype(str)
    metadata=metadata.reset_index(drop=True)
    assert np.array_equal(metadata.cell_index,np.arange(len(metadata)))
    assert set(metadata.donor_id)==set(splits.donor_id)
    np.testing.assert_array_equal(metadata.age_years.to_numpy(),metadata.donor_id.map(splits.set_index('donor_id').age_years).to_numpy())
    selected=data.obs[f'selected_{cells}'].to_numpy(bool)
    pool=metadata.loc[selected,['cell_index','donor_id']].copy();pool['cap']=cells
    assert pool.cell_index.is_unique and set(pool.donor_id)==set(splits.donor_id)
    dataset_dir.mkdir(parents=True);cache=dataset_dir/'cache';cache.mkdir()
    for filename in ['human_gene_vocab.json','GeneSets.json','encoder_checkpoint.pth']:
        target=output_root/filename
        if not target.exists():shutil.copyfile(shared_resources/filename,target)
        else:assert digest(target)==digest(shared_resources/filename),target
    splits.to_csv(dataset_dir/'donor_splits.csv',index=False)
    pool.to_parquet(dataset_dir/f'cells{cells}.parquet',index=False)
    metadata.to_parquet(cache/'cell_metadata.parquet',index=False)
    for key,name in [('gene_ids','gene_ids.int32.npy'),('expression_values','expression_values.float16.npy'),
                     ('pathway_activity','pathway_scores.float16.npy')]:
        np.save(cache/name,np.asarray(data.obsm[key]))
    names=np.asarray(data.uns['pathway_names']).astype(str).tolist()
    (cache/'pathway_names.json').write_text(json.dumps(names,indent=2)+'\n')


    train=set(splits.loc[splits.split.eq('train'),'donor_id'])
    fit=metadata.loc[selected & metadata.donor_id.isin(train)].copy()
    view=dataset_dir/'program_bank_training_view';(view/'cache').mkdir(parents=True)
    fit.to_parquet(view/'cache/cell_metadata.parquet',index=False)
    splits.to_csv(view/'donor_splits.csv',index=False)
    for filename in ['pathway_names.json','pathway_scores.float16.npy']:
        (view/'cache'/filename).symlink_to((cache/filename).resolve())

    attempts=[]
    for attempt in range(100):
        try:
            build_program_routes(view,dataset_dir/'program_routes.csv',64,8,30000,random.randrange(2**32))
            attempts.append({'attempt':attempt+1,'feasible':True});break
        except RuntimeError as error:
            if 'requested pathway overlap' not in str(error):raise
            attempts.append({'attempt':attempt+1,'feasible':False})
    else:raise RuntimeError('No feasible bank under the predefined construction settings')
    data.file.close()
    record={'dataset':dataset,'partition':partition,'cells_per_donor':cells,'cached_cells':len(metadata),
            'selected_cells':len(pool),'fit_training_cells':len(fit),'num_programs':64,
            'donor_counts':splits.split.value_counts().to_dict(),'all_model_input_arrays_preserved':True,
            'validation_and_test_donors_used_in_clustering':False,'construction_attempts':attempts,
            'Data1_sha256':digest(workbook),'H5AD_sha256':digest(source)}
    (dataset_dir/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-data',required=True,type=Path)
    parser.add_argument('--shared-resources',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--dataset',required=True,choices=['AIDA','Indonesia','OneK1K','GSE158055'])
    parser.add_argument('--partition',type=int,required=True,choices=range(1,6))
    parser.add_argument('--cells',type=int,default=500,choices=[500,750,1000])
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    prepare(args.release_data,args.shared_resources,args.dataset,args.partition,args.cells,args.output_root)


if __name__=='__main__':main()
