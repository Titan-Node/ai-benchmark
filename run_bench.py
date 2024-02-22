from huggingface_hub import snapshot_download
from subprocess import run
import os

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

listOfBenchmarks = ['"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sd-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/sdxl-turbo", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "runwayml/stable-diffusion-v1-5", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "stabilityai/stable-diffusion-xl-base-1.0", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "text-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-image", "--model_id", "prompthero/openjourney-v4", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt", "--runs", "3"']
                    #'"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"',
                    #'"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"']


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
    if not os.path.exists("models/models--stabilityai--sd-turbo"):
        print("Downloading model... models--stabilityai--sd-turbo - This may take a few minutes.")
        try:
            snapshot_download(repo_id="stabilityai/sd-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    if not os.path.exists("models/models--stabilityai--sdxl-turbo"):
        print("Downloading model... models--stabilityai--sdxl-turbo - This may take a few minutes.")
        try:
            snapshot_download(repo_id="stabilityai/sdxl-turbo", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    if not os.path.exists("models/models--runwayml--stable-diffusion-v1-5"):
        print("Downloading model... models--runwayml--stable-diffusion-v1-5 - This may take a few minutes.")
        try:
            snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    if not os.path.exists("models/models--stabilityai--stable-diffusion-xl-base-1.0"):
        print("Downloading model... models--stabilityai--stable-diffusion-xl-base-1.0 - This may take a few minutes.")
        try:
            snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    if not os.path.exists("models/models--prompthero--openjourney-v4"):
        print("Downloading model... models--prompthero--openjourney-v4 - This may take a few minutes.")
        try:
            snapshot_download(repo_id="prompthero/openjourney-v4", allow_patterns=["*.safetensors", "*.json", "*.txt"], ignore_patterns=[".onnx", ".onnx_data"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    if not os.path.exists("models/models--stabilityai--stable-video-diffusion-img2vid-xt"):
        print("Downloading model... models--stabilityai--stable-video-diffusion-img2vid-xt - This may take a few minutes.")
        try:
            snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", allow_patterns=["*.fp16.safetensors", "*.json"], cache_dir="models")
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")

    #if not os.path.exists("models/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1"):
    #    print("Downloading model... smodels--stabilityai--stable-video-diffusion-img2vid-xt - This may take a few minutes.")
    #    try:
    #        snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1", allow_patterns=["*.fp16.safetensors", "*.json"], token="$HF_TOKEN", cache_dir="models")
    #    except Exception as e:
    #        print(e)
   #         print("An error occurred while downloading the model. Attempting to continue...")
    #    print("Download complete!")

    print("All models downloaded in the models folder.")



def runBenchmark(command, card):
    # Run the benchmark
    command_list = command.strip('"').split('", "')
    command_list.insert(2, "--gpus")
    command_list.insert(3, card)
    print("Running Command: ", command_list)
    benchmark = run(command_list, capture_output=True, text=True)
    print(benchmark.stdout)
    print(benchmark.stderr)
    output = benchmark.stdout
    # Open a file in write mode and write the output
    with open('results.txt', 'a') as file:
        file.write("===================================================================================================== \n")
        file.write(str(command_list) + "\n")
        file.write("===================================================================================================== \n")
        file.write(output)
        print("Benchmark complete. Results saved to results.txt - Moving on to the next benchmark.")
    input("Press Enter to continue")



if __name__ == "__main__":
    card = getGPUCard()
    downloadAllModels()
    for command in listOfBenchmarks:
        runBenchmark(command, card)
    print("Running Benchmark complete!")
    input("Press Enter to exit")