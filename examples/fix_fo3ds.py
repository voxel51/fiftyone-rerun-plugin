#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from google.cloud import storage


def fix_fo3d_files(bucket_name, prefix):
    """
    Fix fo3d files in GCS bucket by replacing local PCD paths with GCS paths.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Path prefix in the bucket (e.g., "fiftyone-rerun-nuscenes-test/data")
    """
    # Initialize GCS client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # List all fo3d files in the specified path
    blobs = list(bucket.list_blobs(prefix=f"{prefix}/"))
    fo3d_blobs = [blob for blob in blobs if blob.name.endswith(".fo3d")]

    print(f"Found {len(fo3d_blobs)} .fo3d files to process.")

    for blob in fo3d_blobs:
        # Download fo3d file to temporary location
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".fo3d") as temp_file:
            blob.download_to_filename(temp_file.name)

            # Read the file content
            with open(temp_file.name, "r") as f:
                fo3d_content = json.load(f)

            # Find and replace all pcdPath values
            modified = False
            for child in fo3d_content.get("children", []):
                if "pcdPath" in child and child["pcdPath"].startswith("/Users"):
                    # Extract filename from the path
                    pcd_filename = os.path.basename(child["pcdPath"])

                    # Replace with GCS path
                    new_path = f"gs://{bucket_name}/{prefix}/{pcd_filename}"
                    print(f"Replacing {child['pcdPath']} with {new_path}")
                    child["pcdPath"] = new_path
                    modified = True

            if modified:
                # Write the modified content back to the temporary file
                with open(temp_file.name, "w") as f:
                    json.dump(fo3d_content, f, indent=2)

                # Upload the modified file back to GCS
                blob.upload_from_filename(temp_file.name)
                print(f"Updated {blob.name}")
            else:
                print(f"No changes needed for {blob.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix fo3d files in GCS to use cloud PCD paths"
    )
    parser.add_argument("--bucket", default="voxel51-test", help="GCS bucket name")
    parser.add_argument(
        "--prefix",
        default="fiftyone-rerun-nuscenes-test/data",
        help="Path prefix in the bucket",
    )

    args = parser.parse_args()

    fix_fo3d_files(args.bucket, args.prefix)
    print("Done!")


if __name__ == "__main__":
    main()
