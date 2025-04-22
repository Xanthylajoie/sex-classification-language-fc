"""
"""

import click
import json
import sys
import itertools
from pathlib import Path
import numpy as np
from nilearn import image
from tqdm import tqdm


def compute(participant_dir):
    participant_id = participant_dir.parts[-1]
    seed_id = participant_dir.parts[-2]
    
    mean4_output = participant_dir / f"{participant_id}_seed-{seed_id}_mean-all4__voxelcorrelations.nii.gz"
    rest1_output = participant_dir / f"{participant_id}_seed-{seed_id}_mean-REST1__voxelcorrelations.nii.gz"
    rest2_output = participant_dir / f"{participant_id}_seed-{seed_id}_mean-REST2__voxelcorrelations.nii.gz"
    restlr_output = participant_dir / f"{participant_id}_seed-{seed_id}_mean-RESTLR__voxelcorrelations.nii.gz"
    restrl_output = participant_dir / f"{participant_id}_seed-{seed_id}_mean-RESTRL__voxelcorrelations.nii.gz"
    
    if all([_.exists() for _ in [mean4_output, rest1_output, rest2_output, restlr_output, restrl_output]]):
        #print(f"{participant_id}_seed-{seed_id} already computed")
        return
    else:
        print(f"{participant_id}_seed-{seed_id} running")
    
    niis = list(participant_dir.glob("*_ses-REST*_voxelcorrelations.nii.gz"))
    niis.sort()
    
    if len(niis) == 4:
        # mean4
        #print(niis)
        mean_img = image.mean_img(niis)
        mean_img.to_filename(mean4_output)

        # REST1
        #print(niis[:2])
        mean_img = image.mean_img(niis[:2])
        mean_img.to_filename(rest1_output)
    
        # REST2
        #print(niis[2:])
        mean_img = image.mean_img(niis[2:])
        mean_img.to_filename(rest2_output)
        
        # RESTLR
        #print([niis[0]] + [niis[2]])
        mean_img = image.mean_img([niis[0]] + [niis[2]])
        mean_img.to_filename(restlr_output)
    
        # RESTRL
        #print([niis[1]] + [niis[3]])
        mean_img = image.mean_img([niis[1]] + [niis[3]])
        mean_img.to_filename(restrl_output)


@click.command()
@click.option(
    "--results_dir", "-r", required=True, type=Path,
    help="",
)
@click.option(
    "--seeds_json", "-s", required=True, type=Path,
    help="JSON file is a list of seed(s). A seed is ['name_of_the_seeds', [x_coord, y_coord, z_coord]]",
)
def main(results_dir, seeds_json):
    with seeds_json.open("r") as seeds_file:
        seeds_data = json.load(seeds_file)
    seeds_labels = [_[0] for _ in seeds_data]
    
    for label in seeds_labels:
        #print(label)
        for participant_dir in tqdm((results_dir / label).iterdir()):
            if participant_dir.is_dir():
                compute(participant_dir)

if __name__ == '__main__':
    main()

