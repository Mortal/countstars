import argparse
import os

import bintable
import numpy as np

from single_from_qhull import single_linkage


parser = argparse.ArgumentParser()
parser.add_argument("--input-data", required=True)
parser.add_argument("--output-clusters", required=True)


def main() -> None:
    args = parser.parse_args()
    assert args.input_data.endswith("/bintable.json")
    assert args.output_clusters.endswith(".npy")

    df = bintable.read(
        os.path.dirname(args.input_data),
        only_columns=[
            "SOURCE_ID_GAIA",
            "scaled_En",
            "scaled_Lz",
            "scaled_Lperp",
        ],
    )
    Z = single_linkage(np.c_[df["scaled_En"], df["scaled_Lz"], df["scaled_Lperp"]])
    np.save(args.output_clusters, Z, allow_pickle=False)


if __name__ == "__main__":
    main()
