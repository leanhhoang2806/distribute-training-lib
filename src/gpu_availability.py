import pynvml

def count_gpus():
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return num_gpus

if __name__ == "__main__":
    try:
        num_gpus = count_gpus()
        if num_gpus > 0:
            print(f"Number of available GPUs: {num_gpus}")
        else:
            print("No GPUs found on your computer.")
    except pynvml.NVMLError as err:
        print(f"Error: {err}")
