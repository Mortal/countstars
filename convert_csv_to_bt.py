import argparse

import bintable
import numpy as np
from astropy.table import Table


parser = argparse.ArgumentParser()
parser.add_argument("--input-csv", required=True)
parser.add_argument("--output", required=True)


def main() -> None:
    args = parser.parse_args()
    assert args.input_csv.endswith(".csv")
    assert args.output.endswith("/bintable.json")
    t = Table.read(args.input_csv)
    bintable.write(t, os.path.dirname(args.output))


if __name__ == "__main__":
    main()
