# drumVAE

## 1. Introduction
[MusicVAE](https://github.com/magenta/magenta/tree/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae) with Pytorch. MusicVAE generate more various types of music. This repo use only drum dataset.

**Example**

input

<audio controls>
<source src='./result/input.mp4'>
</audio>

output

<audio controls>
<source src='./result/output.mp4'>
</audio>

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