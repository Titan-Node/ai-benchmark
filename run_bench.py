from huggingface_hub import snapshot_download
from subprocess import run
import re
import os
import csv
import GPUtil
import cpuinfo
import psutil
import platform
import distro
import csv

access_token = "ACCESS_TOKEN_HERE"

# Get the absolute path of the current directory
current_directory = os.getcwd()

listOfFolders = ["models--stabilityai--sd-turbo",
                 "stabilityai/sdxl-turbo",
                 "runwayml/stable-diffusion-v1-5",
                 "stabilityai/stable-diffusion-xl-base-1.0",
                 "prompthero/openjourney-v4",
                 "stabilityai/stable-video-diffusion-img2vid-xt",
                 "stabilityai/stable-video-diffusion-img2vid-xt-1-1"]

listOfDownloadCommands = ['repo_id="stabilityai/sd-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models"',
                  'repo_id="stabilityai/sdxl-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models"',
                  'repo_id="runwayml/stable-diffusion-v1-5", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models"',
                    'repo_id="stabilityai/stable-diffusion-xl-base-1.0", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models"',
                    'repo_id="prompthero/openjourney-v4", allow_patterns=["*.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models"',
                    'repo_id="stabilityai/stable-video-diffusion-img2vid-xt", allow_patterns=["*.fp16.safetensors", "*.json"], cache_dir="models"',
                    'repo_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1", allow_patterns=["*.fp16.safetensors", "*.json"], token="$HF_TOKEN", cache_dir="models"']

listOfBenchmarks = ['"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt", "--runs", "3"',
                    '"docker", "run", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "' + current_directory + '/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"']


def getGPUCard():
    print("Welcome to the AI Runner Benchmarking Tool")
    print("Please enter which GPU slot you will be using (0, 1, 2, etc.)")
    print("To show the GPU slots, type 'show'")
    card = input("Enter Here (default 0): ")

    while card == "show":
        nvidia_smi = run(["nvidia-smi"], capture_output=True, text=True)
        print(nvidia_smi.stdout)
        print("Please enter which GPU slot you will be using (0, 1, 2, etc.)")
        print("To show the GPU slots, type 'show'")
        card = input("Enter Here (default 0): ")


    if card == "":
        card = "0"
    print("Using GPU slot " + card)
    return card


def getGPUModel(card: int) -> str:
    """Return the GPU model information.

    Args:
        card: The GPU card number.

    Returns:
        The GPU model information.
    """
    gpus = GPUtil.getGPUs()
    gpu = gpus[int(card)]

    return f"{gpu.name} / {gpu.memoryTotal/1e3}GB"


def getHardwareSpecs() -> dict:
    """Returns the hardware specifications of the system.

    Returns:
        str: The hardware specifications of the system.
    """
    # Get the CPU type.
    cpu_info = cpuinfo.get_cpu_info()
    cpu_model = cpu_info["brand_raw"]

    # Get RAM size.
    ram_info = psutil.virtual_memory()
    total_ram = ram_info.total / (1024.0**3)  # Convert bytes to GB

    # Get OS information.
    os_info = f"{platform.system()} ({distro.id()} {distro.version()})"

    return {"cpu_model": cpu_model, "total_ram": total_ram, "os_info": os_info}


def getModelName(benchmark_command: str) -> str:
    """Returns the model name from the docker run command.

    Args:
        benchmark_command: The docker run command.

    Returns:
        str: The model name.
    """
    # Get model id from the command.
    args = [item.strip().strip('"') for item in benchmark_command.split(",")]
    model_id_index = args.index("--model_id")
    model_name = args[model_id_index + 1]

    # Get pipeline from the command.
    pipeline_index = args.index("--pipeline")
    pipeline = args[pipeline_index + 1]

    # Check if SFAST is enabled and build the model name.
    sfast = " SFAST - " if "--SFAST" in args else ""
    model_name = f"{model_name} ({sfast}{pipeline})"

    return model_name


def downloadAllModels():
    print("Downloading model... models--stabilityai--sd-turbo - This may take a few minutes.")
    try:
        snapshot_download(repo_id="stabilityai/sd-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... models--stabilityai--sdxl-turbo - This may take a few minutes.")
    try:
        snapshot_download(repo_id="stabilityai/sdxl-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... models--runwayml--stable-diffusion-v1-5 - This may take a few minutes.")
    try:
        snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... models--stabilityai--stable-diffusion-xl-base-1.0 - This may take a few minutes.")
    try:
        snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... models--prompthero--openjourney-v4 - This may take a few minutes.")
    try:
        snapshot_download(repo_id="prompthero/openjourney-v4", allow_patterns=["*.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... models--stabilityai--stable-video-diffusion-img2vid-xt - This may take a few minutes.")
    try:
        snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", allow_patterns=["*.fp16.safetensors", "*.json"], cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("Downloading model... smodels--stabilityai--stable-video-diffusion-img2vid-xt-1-1 - This may take a few minutes.")
    try:
        snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1", allow_patterns=["*.fp16.safetensors", "*.json"], token=access_token, cache_dir="models")
    except Exception as e:
        print(e)
        print("An error occurred while downloading the model. Attempting to continue...")
    print("Download complete!")

    print("All models downloaded in the models folder.")



def runBenchmark(command, card, pause):
    # Run the benchmark
    command_list = command.strip('"').split('", "')
    command_list.insert(2, "--gpus")
    command_list.insert(3, card)
    print("Running Command: ", command_list)
    print("This may take a few minutes...")
    benchmark = run(command_list, capture_output=True, text=True)
    # Get Regular expression of "avg inference time:" from stout
    tempInferenceTime = re.findall(r'avg inference time: (.+?)s\n', benchmark.stdout)
    tempMaxMemory = re.findall(r'avg inference max GPU memory allocated: (.+?)GiB\n', benchmark.stdout)
    tempMaxReserveMemory = re.findall(r'avg inference max GPU memory reserved: (.+?)GiB\n', benchmark.stdout)
    if tempInferenceTime != []:
        print(benchmark.stdout)
        try:
            avgInferenceTime = tempInferenceTime[0]
            maxGPUMemoryAllocated = tempMaxMemory[0]
            maxGPUMemoryReserved = tempMaxReserveMemory[0]
        except:
            avgInferenceTime = '0'
            maxGPUMemoryAllocated = '0'
            maxGPUMemoryReserved = '0'

    else:
        print(benchmark.stdout)
        print(benchmark.stderr)
        tempMemoryError = re.findall(r'out of memory', benchmark.stdout)
        if tempMemoryError != []:
            avgInferenceTime = 'out of memory'
            maxGPUMemoryAllocated = 'out of memory'
            maxGPUMemoryReserved = 'out of memory'
        else:
            avgInferenceTime = '0'
            maxGPUMemoryAllocated = '0'
            maxGPUMemoryReserved = '0'


    # Open a file in write mode and write the output
    with open('results.txt', 'a') as file:
        file.write("===================================================================================================== \n")
        file.write(str(command_list) + "\n")
        file.write("===================================================================================================== \n")
        file.write(benchmark.stdout)
        if tempInferenceTime == []:
            file.write(benchmark.stderr)
    return avgInferenceTime, maxGPUMemoryAllocated, maxGPUMemoryReserved


def pullLatestDockerImage():
    print("Pulling latest docker image...")
    pull = run(["docker", "pull", "livepeer/ai-runner:latest"])
    print("Docker image pulled successfully")


class BenchMarkResults:
    """Class to store the benchmark results and write them to a CSV file."""
    def __init__(self, gpu_info: str, system_info: dict):
        self.results = []

        # Store GPU and system information in csv.
        with open("system_info.csv", mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["GPU Information", gpu_info])
            for key, value in system_info.items():
                key_str = key.replace("_", " ").title()
                writer.writerow([key_str, value])

    def add_result(
        self,
        model_name: str,
        avg_inference_time: float,
        max_gpu_memory_allocated: float,
        max_gpu_memory_reserved: float,
    ):
        """Add a result to the benchmark results.
        
        Args:
            model_name: The name of the model.
            avg_inference_time: The average inference time.
            max_gpu_memory_allocated: The maximum GPU memory allocated.
        """
        self.results.append(
            {
                "model_name": model_name,
                "avg_inference_time": avg_inference_time,
                "max_gpu_memory_allocated": max_gpu_memory_allocated,
                "max_gpu_memory_reserved": max_gpu_memory_reserved,
            }
        )

    def write_to_csv(self, file_name: str):
        """Write the benchmark results to a CSV file.

        Args:
            file_name: The name of the CSV file.
        """
        # Prepare the header row
        header = ["Metric"]
        for result in self.results:
            header.append(result["model_name"])

        # Prepare the rows for each metric
        avg_inference_time_row = ["avg_inference_time"]
        max_gpu_memory_allocated_row = ["max_gpu_memory_allocated"]
        max_gpu_memory_reserverd_row = ["max_gpu_memory_reserved"]
        for result in self.results:
            avg_inference_time_row.append(result["avg_inference_time"])
            max_gpu_memory_allocated_row.append(result["max_gpu_memory_allocated"])
            max_gpu_memory_reserverd_row.append(result["max_gpu_memory_reserved"])

        # Write to CSV
        with open(file_name, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(avg_inference_time_row)
            writer.writerow(max_gpu_memory_allocated_row)


if __name__ == "__main__":
    avg_inference_time = []
    max_GPU_memory_allocated = []
    max_GPU_memory_reserved = []
    card = getGPUCard()
    print("Getting GPU Information...")
    GPUModel = getGPUModel(card)
    print(f"GPU Information: {GPUModel}")
    print("Getting System Information...")
    hardwareSpecs = getHardwareSpecs()
    print(f"System Information: {hardwareSpecs}")
    print("Benchmarks are heavy and will pause between each benchmark to ensure no errors occur. Only skip if you are sure.")
    pause = input("Skip pausing between benchmarks? (y/n): ")
    print("Using current directory:", current_directory)
    downloadAllModels()
    pullLatestDockerImage()
    with open('results.txt', 'w') as file:
        file.write("===================================================================================================== \n")
        file.write("GPU Slot: " + card + "\n")
        file.write("===================================================================================================== \n")
    benchmark_results = BenchMarkResults(GPUModel, hardwareSpecs)
    for command in listOfBenchmarks:
        inferenceTime, GPUMemory, GPUReserve = runBenchmark(command, card, pause)
        avg_inference_time.append(inferenceTime)
        max_GPU_memory_allocated.append(GPUMemory)
        max_GPU_memory_reserved.append(GPUReserve)
        benchmark_results.add_result(
            getModelName(command), avg_inference_time[-1], max_GPU_memory_allocated[-1], max_GPU_memory_reserved[-1]
        )
        with open('results.txt', 'a') as file:
            file.write("===================================================================================================== \n")
            file.write("Average Inference Time: " + str(avg_inference_time) + "\n")
            file.write("Max GPU Memory Allocated: " + str(max_GPU_memory_allocated) + "\n")
            file.write("Max GPU Memory Reserved: " + str(max_GPU_memory_reserved) + "\n")
            file.write("===================================================================================================== \n")
        print("=====================================================================================================")
        print("Average Inference Time: ", avg_inference_time)
        print("Max GPU Memory Allocated: ", max_GPU_memory_allocated)
        print("Max GPU Memory Reserved: ", max_GPU_memory_reserved)
        print("=====================================================================================================")
        print("Benchmark complete. Results saved to results.txt - Moving on to the next benchmark.")
        if pause != "y":
            input("Press Enter to continue")

    benchmark_results.write_to_csv("results.csv")
    print("Running Benchmark complete!")
    input("Press Enter to exit")
