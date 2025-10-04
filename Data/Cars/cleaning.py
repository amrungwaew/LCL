import numpy as np
import polars as pl

from pathlib import Path


def main():
    cars_df = pl.read_csv(
        "https://raw.github.com/arteagac/xlogit/master/examples/data/car100_long.csv"
    )
    cars_df2 = cars_df.with_columns(
        neg_price=-pl.col("price"), neg_operating_cost=-pl.col("opcost")
    ).drop("price", "opcost")

    cars_df2.write_parquet(Path(__file__).parent / "cars.parquet")
    cars_df2.glimpse()


if __name__ == "__main__":
    main()
