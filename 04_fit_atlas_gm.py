"""
"""

import click
from pathlib import Path
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import statsmodels.api as sm
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option(
    "--results_dir", "-r", required=True, type=Path,
    help="",
)
@click.option(
    "--seed_name", "-s", required=True,
    help="",
)
@click.option(
    "--glob_pattern", "-g", required=True,
    help="Example: 'mean-all4', or 'ses-REST1LR'",
)
@click.option(
    "--atlas", "-a", required=True,
    help="",
)
@click.option(
    "--data_type", "-d", default="fisherz",
    help="",
)
def main(results_dir, seed_name, glob_pattern, atlas, data_type):

    if atlas == "destrieux":
        atlas_data = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
        masker = NiftiLabelsMasker(atlas_data.maps)
        labels = list(atlas_data.labels.drop([0,42,117]).reset_index(drop=True).name)
        labels = [seed_name + "__" + _.replace(" ", "_") for _ in labels]
    else:
        print(f"{atlas} not implemented")
        return

    csv_dir = results_dir / "atlas_means" / f"{atlas}_{data_type}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    all_data_corrected = {}
    for image in tqdm(list(results_dir.glob(f"{seed_name}/*/*{glob_pattern}*{data_type}.nii.gz"))):
        participant_id = image.parts[-2].split('-')[1]
        gm_image = Path(f"/data/brambati/dataset/HCP/downloaded_data/{participant_id}/MNINonLinear/T1w_restore_brain.nii.gz")

        image_data = masker.fit_transform(image).flatten()
        gm_data = masker.fit_transform(gm_image).flatten()
        gm_data = gm_data - np.mean(gm_data)
        gm_data = preprocessing.scale(gm_data, with_mean=False)

        model = sm.GLM(image_data, sm.add_constant(gm_data))
        results = model.fit()
        image_data_corrected = image_data - gm_data * results.params[1]

        all_data[f"sub-{participant_id}"] = image_data
        all_data_corrected[f"sub-{participant_id}"] = image_data_corrected

    df = pd.DataFrame(all_data).T.sort_index()
    df.columns = labels
    df_corrected = pd.DataFrame(all_data_corrected).T.sort_index()
    df_corrected.columns = labels

    df.to_csv(csv_dir / f"seed-{seed_name}_atlas-{atlas}_{glob_pattern}__{data_type}.csv")
    df_corrected.to_csv(csv_dir / f"seed-{seed_name}_atlas-{atlas}_{glob_pattern}_gmcorrected__{data_type}.csv")



if __name__ == '__main__':
    main()

