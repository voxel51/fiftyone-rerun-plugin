# FiftyOne Rerun Plugin

A plugin that enables users to visualize [Rerun](https://rerun.io/) data files (`.rrd`) inside the FiftyOne App.
![rerun-fo-nuscenes-small](https://github.com/user-attachments/assets/d93e86bb-ae59-493b-9028-4b9c677afca9)

## Installation
```
fiftyone plugins download \
    https://github.com/voxel51/fiftyone-rerun-plugin \
    --plugin-names @fiftyone/rerun-plugin
```

## Example usage with NuScenes dataset

We have a Python script in the [examples](examples/load-nuscenes.py) folder that:
1. Creates RRD files with a timeline containing lidar points of each scene in the NuScenes mini dataset.
2. Creates a grouped FiftyOne dataset with all camera images, lidar, radar, as well as references to the RRD files from (1)

```
# make sure fiftyone and rerun-sdk are installed first
pip install fiftyone rerun-sdk

# show help
python examples/load-nuscenes.py -h

# create rrd files as well as create a fiftyone dataset with references to the RRD files
python examples/load-nuscenes.py --rrd --fiftyone

# start rerun server
rerun --serve

# start fiftyone app
fiftyone app launch
```
