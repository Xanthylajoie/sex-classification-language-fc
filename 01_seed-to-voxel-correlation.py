"""
"""

import click
import json

from pathlib import Path
import numpy as np
from nilearn.image import load_img, iter_img
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
import pandas as pd


def seed_to_voxel(image, seeds_coord, mask, confounds):
    image = load_img(image)

    # Seeds timeseries
    seed_masker = NiftiSpheresMasker(
        seeds_coord,
        radius=8,
        detrend=True,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        t_r=0.72,
        clean__butterworth__padtype="even",  # kwarg to modify Butterworth filter
    )
    seed_timeseries = seed_masker.fit_transform(image, confounds=confounds)

    # Brain timeseries
    brainmasker = NiftiMasker(
        mask_img=mask,
        smoothing_fwhm=6,
        detrend=True,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        t_r=0.72,
        clean__butterworth__padtype="even",  # kwarg to modify Butterworth filter
    )
    brain_timeseries = brainmasker.fit_transform(image, confounds=confounds)

    # Voxel correlations
    correlations = (
        np.dot(brain_timeseries.T, seed_timeseries) /
        seed_timeseries.shape[0]
    )
    correlations_img = brainmasker.inverse_transform(correlations.T)

    return seed_timeseries, correlations_img


@click.command()
@click.option(
    "--participant_id", "-p", required=True, type=Path,
    help="",
)
@click.option(
    "--seeds_json", "-s", required=True, type=Path,
    help="JSON file is a list of seed(s). A seed is ['name_of_the_seeds', [x_coord, y_coord, z_coord]]",
)
@click.option(
    "--mask", "-m", required=True, type=Path,
    help="",
)
@click.option(
    "--output_dir", "-o", required=True, type=Path,
    help="",
)
@click.option(
    "--confound_filename", "-c", default="Movement_Regressors_dt.txt",
    help="",
)
def main(participant_id, seeds_json, mask, confound_filename, output_dir):
    outputs = [*output_dir.glob(f"*/sub-{participant_id}/sub-{participant_id}_ses-*__voxelcorrelations.nii.gz")]

    with seeds_json.open("r") as seeds_file:
        seeds_data = json.load(seeds_file)

    seeds_coords = [_[1] for _ in seeds_data]

    if len(outputs) == 4 * len(seeds_data):
        print(f"sub-{participant_id} already computed")
        return

    else:
        print(f"sub-{participant_id} running")

    downloaded_data = Path("/data/brambati/dataset/HCP/downloaded_data") / participant_id

    for image in downloaded_data.glob("MNINonLinear/Results/*/*clean.nii.gz"):
        session = image.parts[-2].split("_")
        session = session[1] + session[2]
        confounds = pd.read_csv(image.parent / confound_filename, header=None, delim_whitespace=True).to_numpy()
        seed_timeseries, correlations_img = seed_to_voxel(image, seeds_coords, mask, confounds)

        # Savings timeseries
        seed_timeseries_file = (
            output_dir / "timeseries" / f"sub-{participant_id}" /
            f"sub-{participant_id}_ses-{session}__timeseries.npy"
        )
        seed_timeseries_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(seed_timeseries_file, seed_timeseries)

        # Savings seed to voxel images
        seeds_labels = [_[0] for _ in seeds_data]
        for label, volume in zip(seeds_labels, iter_img(correlations_img)):
            output = (
                output_dir / label / f"sub-{participant_id}" /
                f"sub-{participant_id}_ses-{session}_seed-{label}__voxelcorrelations.nii.gz"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            volume.to_filename(output)


if __name__ == '__main__':
    main()

