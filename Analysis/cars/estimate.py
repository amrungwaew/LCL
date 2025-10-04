from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
import polars as pl
import sys
from jax import config as jax_config

sys.path.insert(0, str(Path(__file__).parents[1]))
from lcl.latent_class_conditional_logit import LatentClassConditionalLogit


def main():
    configure_polars_and_jax()  # Streaming for Polars and double precision for Jax

    # Load data
    data_dir = Path(__file__).parents[2] / "Data" / "cars"
    cars_df = pl.read_parquet(data_dir / "cars.parquet")
    cars_df.glimpse()

    # Specify model
    alt_specific_vars = [
        "neg_price",
        "neg_operating_cost",
        "hiperf",
        "medhiperf",
        "range",
        "ev",
        "hybrid",
    ]
    estimation_spec = {
        "X": cars_df.select(alt_specific_vars).cast(pl.Float64).to_jax(),
        "varnames": alt_specific_vars,
    }

    # model = LatentClassConditionalLogit()

    # model.fit(**estimation_spec)


def configure_polars_and_jax():
    jax_config.update("jax_enable_x64", True)
    pl.config.set_engine_affinity(engine="streaming")


if __name__ == "__main__":
    main()
