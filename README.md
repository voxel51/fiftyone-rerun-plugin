# FiftyOne Rerun Plugin

A plugin that enables users to visualize [Rerun](https://rerun.io/) data files
(`.rrd`) inside the [FiftyOne App](https://docs.voxel51.com/user_guide/app.html).

![rerun-fo-nuscenes-small](https://github.com/user-attachments/assets/d93e86bb-ae59-493b-9028-4b9c677afca9)

## Installation

```shell
fiftyone plugins download https://github.com/voxel51/fiftyone-rerun-plugin
```

## Release artifacts and Rerun version compatibility

Prebuilt plugin artifacts are published on the repository's
[GitHub Releases](https://github.com/voxel51/fiftyone-rerun-plugin/releases) page.
Each release includes a zip file named like:

`fiftyone-rerun-plugin-<version>.zip`

The zip contains:
- `dist/` (the built plugin bundle)
- `fiftyone.yaml` (plugin metadata with the matching plugin version)

Version coupling:
- Plugin release version `vX.Y.Z` is paired with `@rerun-io/web-viewer-react` `^X.Y.Z`
- Choose a plugin release that matches the Rerun web viewer version you want to use.

## Local versioned build

To build a versioned plugin artifact locally:

```shell
./scripts/build-release-local.sh 0.29.2
```

This produces:
- `dist_artifacts/fiftyone-rerun-plugin-0.29.2/`

To also generate a zip:

```shell
./scripts/build-release-local.sh --zip 0.29.2
```

This additionally produces:
- `dist/fiftyone-rerun-plugin-0.29.2.zip`

## Example usage with NuScenes dataset

We have a Python script in the [examples](examples/load-nuscenes.py) folder that:
1. Creates RRD files with a timeline containing lidar points of each scene in the NuScenes mini dataset
2. Creates a grouped FiftyOne dataset with all camera images, lidar, radar, as well as references to the RRD files from (1)

Before you run the script, make sure you have the NuScenes mini split dataset downloaded and extracted.
You can download it from [here](https://www.nuscenes.org/data/v1.0-mini.tgz).

After downloading, extract the dataset to a folder. You'll need the path of the extracted dataset to run the script.

```shell
# download the NuScenes mini dataset
wget https://www.nuscenes.org/data/v1.0-mini.tgz

# extract the dataset to a folder named nuscenes-mini
tar -xvf v1.0-mini.tgz
```

Run the script to prepare the dataset:

```shell
# show help
python examples/load-nuscenes.py -h

# create rrd files as well as create a fiftyone dataset with references to the RRD files
python examples/load-nuscenes.py --nuscenes-data-dir /path/to/nuscenes --rrd --fiftyone
```

Then launch the App:

```shell
# start rerun server
rerun --serve

# start fiftyone app
fiftyone app launch
```
