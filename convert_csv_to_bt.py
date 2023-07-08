import numpy as np
from astropy.table import Table
import bintable


def main() -> None:
    t = Table.read("df.csv")
    bintable.write(t, "df")
    t = Table.read("df_artificial.csv")
    bintable.write(t, "df_artificial")


if __name__ == "__main__":
    main()
