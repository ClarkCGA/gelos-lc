#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import boto3
from gelos.config import AWS_ACCESS_KEY, AWS_REGION, AWS_SECRET_KEY
import typer

app = typer.Typer()


@app.command()
def upload(
    processed_data_dir: Path = typer.Option(
        Path("/app/data/processed"), help="Processed data directory"
    ),
    data_version: str = typer.Option("v0.50.1", help="Data version string"),
    bucket: str = typer.Option("gelos-fm", help="Target S3 bucket name"),
):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    app_files_dir = processed_data_dir / data_version / "app_files"
    json_dir = app_files_dir / "json"
    pmtiles_dir = app_files_dir / "pmtiles"
    config_dir = app_files_dir / "config"

    def upload_dir(local_dir: Path, prefix: str):
        if not local_dir.exists():
            typer.echo(f"WARNING: {local_dir} does not exist, skipping {prefix}/ uploads")
            return
        for p in sorted(local_dir.glob("*")):
            if not p.is_file():
                continue
            key = f"{prefix}/{p.name}"
            s3_client.upload_file(str(p), bucket, key)
            typer.echo(f"Uploaded {p} -> s3://{bucket}/{key}")

    upload_dir(json_dir, "json")
    upload_dir(pmtiles_dir, "pmtiles")
    upload_dir(config_dir, "config")


if __name__ == "__main__":
    app()
