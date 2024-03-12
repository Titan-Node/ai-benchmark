from huggingface_hub import snapshot_download
from subprocess import run
import re
import os
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

if __name__ == "__main__":
    avg_inference_time = []
    max_GPU_memory_allocated = []
    max_GPU_memory_reserved = []
    card = getGPUCard()
    print("Benchmarks are heavy and will pause between each benchmark to ensure no errors occur. Only skip if you are sure.")
    pause = input("Skip pausing between benchmarks? (y/n): ")
    print("Using current directory:", current_directory)
    downloadAllModels()
    pullLatestDockerImage()
    with open('results.txt', 'w') as file:
        file.write("===================================================================================================== \n")
        file.write("GPU Slot: " + card + "\n")
        file.write("===================================================================================================== \n")
    for command in listOfBenchmarks:
        inferenceTime, GPUMemory, GPUReserve = runBenchmark(command, card, pause)
        avg_inference_time.append(inferenceTime)
        max_GPU_memory_allocated.append(GPUMemory)
        max_GPU_memory_reserved.append(GPUReserve)
        with open('results.txt', 'a') as file:
            file.write("===================================================================================================== \n")
            file.write("Average Inference Time: " + str(avg_inference_time) + "\n")
            file.write("Max GPU Memory Allocated: " + str(max_GPU_memory_allocated) + "\n")
            file.write("Max GPU Memory Reserved: " + str(max_GPU_memory_reserved) + "\n")
            file.write("===================================================================================================== \n")
        csvData = [["Row 1 - Average Inference Time", "Row 2 - Max GPU Memory Allocated", "Row 3 - Max GPU Memory Reserved"],
                   avg_inference_time,
                   max_GPU_memory_allocated,
                   max_GPU_memory_reserved]
        with open('results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write each row to the file
            for row in csvData:
                writer.writerow(row)
        print("=====================================================================================================")
        print("Average Inference Time: ", avg_inference_time)
        print("Max GPU Memory Allocated: ", max_GPU_memory_allocated)
        print("Max GPU Memory Reserved: ", max_GPU_memory_reserved)
        print("=====================================================================================================")
        print("Benchmark complete. Results saved to results.txt - Moving on to the next benchmark.")
        if pause != "y":
            input("Press Enter to continue")

    print("Running Benchmark complete!")
    input("Press Enter to exit")
