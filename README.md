# drumVAE

## 1. Introduction
[MusicVAE](https://github.com/magenta/magenta/tree/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae) with Pytorch. MusicVAE generate more various types of music. This repo use only drum dataset.

**Example**

input

https://user-images.githubusercontent.com/52961246/175804062-280a33b0-0d1d-4aac-927d-407f78a7f3cf.mp4



output

https://user-images.githubusercontent.com/52961246/175804070-ff83e3ad-bc61-4051-89fa-56afb9fa6c55.mp4


## 2. Preparation
### 2.1 Prerequisites
python 3.8
Pytorch 1.9.0+cu111 

```bash
pip install -r requirements.txt
```

Install fluidsynth
```bash
sudo apt-get update -y
sudo apt-get install -y fluidsynth
```

### 2.2 Data
Download dataset from [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove#control-changes)
