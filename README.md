# Ai-Benchmark

Benchmark script for running different AI models from the [Livepeer AI Runner repo](https://github.com/livepeer/ai-worker/tree/main).

The script runs 22 Benchmarks over 6 different models:

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

## System Dependencies

1. Please make sure you have [Docker](https://www.docker.com/) installed and running before starting the Benchmark
2. You will need Nvidia Driver 535 or greater.
3. For Linux - You may need the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) with the experimental packages enabled to detect the GPUs properly.

> ℹ️ **Note:** This will download approx 36GB disk space for the models and 10GB for the docker image.

### Linux Install Commands

On linux, you can install the required system dependencies with the following commands:

```bash
sudo apt update
sudo apt upgrade
sudo apt install docker.io
sudo apt install nvidia-driver-535
sudo curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### How To Use

#### Download Binary and Run

You can download the latest binary release [here](https://github.com/Titan-Node/ai-benchmark/releases) and run the benchmark.

#### Run Python Script Directly

You can also run the benchmark directly from the Python script:

1. Install the Python dependencies `pip install -r requirements.txt`.
2. Run the benchmark `python benchmark.py`.

#### Results

The benchmark will output several files:

- `system_info.txt` - Information about your system.
- `results.txt` - The benchmark results in a human-readable format.
- `results.csv` - The benchmark results in a CSV format.

Additionally, at the end of the benchmark, the user is given the option to write the results to tje [LivePeer AI benchmarking community spreadsheet](https://docs.google.com/spreadsheets/d/1G3oH3fR3L9rc6qMFmQ8aOosyeELkJKJN0Aw1Bl_Xsi4/edit#gid=0). Sharing your results is optional. Still, it helps the community to understand the performance of different hardware and configurations 🚀. If anything goes wrong during this step, feel free to add them directly to the [Spreadsheet](https://docs.google.com/spreadsheets/d/1G3oH3fR3L9rc6qMFmQ8aOosyeELkJKJN0Aw1Bl_Xsi4/edit#gid=0) or send a copy of your `results.csv` to my Discord, and I will upload them.

### Development Troubleshooting

#### Setup Google Sheets Integration

The benchmarking tool provides the functionality to record the results directly into the [LivePeer AI benchmarking community spreadsheet](https://docs.google.com/spreadsheets/d/1G3oH3fR3L9rc6qMFmQ8aOosyeELkJKJN0Aw1Bl_Xsi4/edit#gid=0). This feature is enabled when using the binaries. Still, it requires some setup when running the script directly. To enable this feature, follow these steps given in the [Google Sheets API Overview documentation](https://developers.google.com/sheets/api/guides/concepts):

1. **Create a Google Cloud Project**: Start by creating a new project on Google Cloud.
2. **Enable Google Sheets API**: Navigate to the 'Library' in your Google Cloud project, search for 'Google Sheets API' and enable it for your project.
3. **Create a Service Account**: In the 'IAM & Admin' section, [create a new service account](https://support.google.com/a/answer/7378726?hl=en). During creation, grant this account 'Editor' access to your project.
4. **Download Credentials**: You can create a key once the service account is created. Select 'JSON' as the key type and download the generated `credentials.json` file.
5. **Place the Credentials File**: Move the downloaded `credentials.json` file to the root directory of this project. The benchmarking tool will look for this file to authenticate with the Google Sheets API. If the file is not found, the tool will skip writing to the Google workbook.
6. **Share the Workbook**: Open the Google Sheets workbook you want to write to and share it with the email address associated with your created service account. This email can be found in the `credentials.json` file.
7. **Set the SPREADSHEET_ID**: Open the `benchmark.py` file and set the `SPREADSHEET_ID` variable to the ID of your Google Sheets workbook. The ID is the string between 'd/' and '/edit' in your workbook URL.

After completing these steps, the benchmarking tool can write the results to the specified Google Sheets workbook.
