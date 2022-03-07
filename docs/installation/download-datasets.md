# Downloading datasets 

**Note:** These instructions assume you have
[installed the full library](../installation/installation-allenact.md#full-library) and, generally, [installed
specific plugin requirements](../installation/installation-allenact.md#plugins-extra-requirements).

The below provides instructions on how to download datasets necessary for defining the train, validation, and
test sets used within the ObjectNav/PointNav tasks in the `iTHOR` and `RoboTHOR` environments.

<!--
Note that these datasets **do not include** scene assets for the below datasets. For `iTHOR` and `RoboTHOR`
these scene assets will be downloaded automatically, for `habitat` please following the instructions
in [this tutorial](installation-framework.md).
-->

## Point Navigation (PointNav)

### RoboTHOR
To get the PointNav dataset for `RoboTHOR` run the following command:
```bash
bash datasets/download_navigation_datasets.sh robothor-pointnav
```
This will download the dataset into `datasets/robothor-pointnav`.

### iTHOR
To get the PointNav dataset for `iTHOR` run the following command:
```bash
bash datasets/download_navigation_datasets.sh ithor-pointnav
```
This will download the dataset into `datasets/ithor-pointnav`.

## Object Navigation (ObjectNav)

### RoboTHOR
To get the ObjectNav dataset for `RoboTHOR` run the following command:

```bash
bash datasets/download_navigation_datasets.sh robothor-objectnav
```
This will download the dataset into `datasets/robothor-objectnav`.

### iTHOR
To get the ObjectNav dataset for `iTHOR` run the following command:
```bash
bash datasets/download_navigation_datasets.sh ithor-objectnav
```
This will download the dataset into `datasets/ithor-objectnav`.
