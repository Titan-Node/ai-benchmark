# Ai-Benchmark

Benchmark script for running different ai models from the [Livepeer AI Runner repo](https://github.com/livepeer/ai-worker/tree/main)

The script runs 22 Benchmarks over 6 different models

- sd-turbo
- sdxl-turbo
- stable-diffusion-v1-5
- openjourney-v4
- stable-diffusion-xl-base-1.0
- stable-video-diffusion-img2vid-xt

(10 text to image, 10 image to image, 2 image to video)

## Supported Systems

- Windows
- Linux
- Nvidia GPUs only

## Dependancies

1. Please make sure you have [Docker](https://www.docker.com/) installed and running before starting the Benchmark
2. You will need Nvidia Driver 535 or greater
3. For Linux - You may need the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) with the experimental packages enabled to detect the GPUs properly.

> ℹ️ **Note:** This will download approx 36GB disk space for the models and 10GB for the docker image.

## Download from releases and run

[Download here](https://github.com/Titan-Node/ai-benchmark/releases)

## Results

Feel free to add your results to the [Spreadsheet](https://docs.google.com/spreadsheets/d/1G3oH3fR3L9rc6qMFmQ8aOosyeELkJKJN0Aw1Bl_Xsi4/edit#gid=0).

Or send a copy of your `results.txt` to my Discord and I can upload them.

## Linux Install Commands

`sudo apt update`

`sudo apt upgrade`

`sudo apt install docker.io`

`sudo apt install nvidia-driver-535`

```
sudo curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

`sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list`

`sudo apt update`

`sudo apt install -y nvidia-container-toolkit`

`sudo nvidia-ctk runtime configure --runtime=docker`

`sudo systemctl restart docker`

## Run the Benchmark

1. Install the python dependencies `pip install -r requirements.txt`.
2. Run the benchmark `python benchmark.py`.
3. The results will be saved to a `results.txt` and `results.csv` file.
