from __future__ import annotations

import argparse
import datetime as dt

from backend.ocean_datasets import DATASET_SPECS, OceanDatasetManager, daterange


def parse_args():
    parser = argparse.ArgumentParser(description="Download ocean NetCDF datasets by template")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--base-url", required=True, help="Base URL where files are hosted")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manager = OceanDatasetManager(base_urls={args.dataset: args.base_url})
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    for day in daterange(start, end):
        path = manager.download_dataset_for_day(args.dataset, day, overwrite=args.overwrite)
        print(f"downloaded: {path}")


if __name__ == "__main__":
    main()
