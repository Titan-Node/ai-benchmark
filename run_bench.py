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
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt", "--runs", "3"',
                    '"docker", "run", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"',
                    '"docker", "run", "-e", "SFAST=true", "-v", "/models:/models", "livepeer/ai-runner:latest", "python", "bench.py", "--pipeline", "image-to-video", "--model_id", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "--runs", "3"']


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





def downloadModel(folder, command):
    # If folder exists, skip download
    if not os.path.exists("models/" + folder):
        print("Downloading model... " + folder + " - This may take a few minutes.")
        try:
            snapshot_download(command)
        except Exception as e:
            print(e)
            print("An error occurred while downloading the model. Attempting to continue...")
        print("Download complete!")
    print("Model " + folder + " already exists in the models folder.")
    input("Press Enter to continue")


def runBenchmark(command, card):
    # Run the benchmark
    print("Running Command: " + command)
    benchmark = run([command + " --gpus, " + card], capture_output=True, text=True)
    print(benchmark.stdout)
    output = benchmark.stdout
    # Open a file in write mode and write the output
    with open('results.txt', 'a') as file:
        file.write("=====================================================================================================")
        file.write(command)
        file.write("=====================================================================================================")
        file.write(output)
        print("Benchmark complete. Results saved to results.txt - Moving on to the next benchmark.")



if __name__ == "__main__":
    card = getGPUCard()
    for folder, command in zip(listOfFolders, listOfDownloadCommands):
        downloadModel(folder, command)
    for command in listOfBenchmarks:
        runBenchmark(command, card)
    print("Running Benchmark complete!")
    input("Press Enter to exit")