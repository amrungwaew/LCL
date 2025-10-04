import sys
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump

import polars as pl
from jax import config as jax_config

from _latent_utility import (
    alt_vars_fullnames,
    alt_vars_prefixes,
    dem_vars_fullnames,
    dem_vars_prefixes,
    random_vars,
)

sys.path.insert(0, str(Path(__file__).parents[2]))

from databuilder.specs import specify_latent_class_logit
from mixlogit.latent_class_logit import LatentClassLogit


def main():
    data_dir = Path(__file__).parents[2] / "Data" / "cars"
    cars_df = pl.read_parquet(data_dir / "cars.parquet")
    cars_df.glimpse()

    configure_polars_jax_and_stdout()
    num_classes = 10

    data_dir = Path(__file__).parent / "train-test-data"
    est_df = pl.read_parquet(data_dir / "estimation.parquet")

    # Load data
    empirical_specs_dict, est_df = specify_latent_class_logit(
        est_df,
        [*random_vars] + alt_vars_fullnames,
        alt_vars_prefixes,
        dem_vars_fullnames,
        dem_vars_prefixes,
        num_classes,
    )

    # Fit model
    model = LatentClassLogit()
    model.fit(**empirical_specs_dict)

    # Print and save estimated coefficients
    coeffs_txt = "\n".join(["\n=== COEFFICIENTS ===\n", str(model.summary())])
    print(coeffs_txt)

    # Store estimated model (in binary format)
    model_storage_dir = Path(__file__).parent / "fitted-models"
    if not model_storage_dir.exists():
        model_storage_dir.mkdir()
    with open(model_storage_dir / f"lclogit{num_classes}.pkl", "wb") as handle:
        dump(model, handle, protocol=HIGHEST_PROTOCOL)


def configure_polars_jax_and_stdout(double_precision=True):
    if double_precision:
        jax_config.update("jax_enable_x64", True)
    pl.Config.set_engine_affinity(engine="streaming")
    pl.Config(tbl_width_chars=10_000, fmt_str_lengths=10_000, tbl_cols=-1, tbl_rows=-1)
    sys.stdout = open(Path(__file__).parent / f"{Path(__file__).stem}.txt", "w")


if __name__ == "__main__":
    main()


# Bug shield
