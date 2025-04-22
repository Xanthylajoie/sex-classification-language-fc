"""
"""

import click
import json
from pathlib import Path
import numpy as np
from nilearn import image
from tqdm import tqdm


def fisher_transformation(pearson_filename, fisherz_filename):
    pearson = image.get_data(pearson_filename)
    z_map = np.arctanh(pearson)
    z_img = image.new_img_like(pearson_filename, z_map)
    z_img.to_filename(fisherz_filename)


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
        for pearson_filename in tqdm((results_dir / label).glob("*/*__voxelcorrelations*")):
            fisherz_filename = Path(str(pearson_filename).replace("__voxelcorrelations", "__fisherz"))
            if fisherz_filename.exists():
                pass
            else:
                #print(fisherz_filename)
                fisher_transformation(pearson_filename, fisherz_filename)

if __name__ == '__main__':
    main()

