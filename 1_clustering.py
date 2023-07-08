import bintable
import numpy as np

from single_from_qhull import single_linkage


def main() -> None:
    df = bintable.read(
        "df",
        only_columns=[
            "SOURCE_ID_GAIA",
            "scaled_En",
            "scaled_Lz",
            "scaled_Lperp",
        ],
    )
    # df_artificial = bintable.read(
    #     "df_artificial",
    #     only_columns=[
    #         "SOURCE_ID_GAIA",
    #         "scaled_En",
    #         "scaled_Lz",
    #         "scaled_Lperp",
    #         "index",
    #     ],
    # )
    Z = single_linkage(np.c_[df["scaled_En"], df["scaled_Lz"], df["scaled_Lperp"]])
    np.save("Z.npy", Z, allow_pickle=False)


if __name__ == "__main__":
    main()
