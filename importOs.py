import os
import time
import torch
import torch.nn as nn
import psutil
import subprocess
import re
from torch.utils.data import DataLoader, TensorDataset

# --- Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load test data
X_test = torch.load(os.path.join(script_dir, "X_test.pt"))
y_test = torch.load(os.path.join(script_dir, "y_test.pt"))

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Constructor ---
def create_vegnet(input_size=2151, hidden_size=128):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )

# --- Temperature Reader (Jetson Nano) ---
def read_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.readline()) / 1000.0
    except:
        return -1.0

# --- Power Usage Reader (Jetson Nano) ---
def read_power():
    try:
        output = subprocess.check_output(['tegrastats'], timeout=1).decode("utf-8")
        match = re.search(r"POM_5V_IN\s+(\d+)mW", output)
        if match:
            return int(match.group(1))
        return -1
    except Exception:
        return -1

# --- GPU Usage Reader (Jetson Nano) ---
def read_gpu_usage():
    try:
        output = subprocess.check_output(['tegrastats'], timeout=1).decode("utf-8")
        match = re.search(r"GR3D_FREQ\s+(\d+)%", output)
        if match:
            return int(match.group(1))
        return -1
    except Exception:
        return -1

# --- Evaluation ---
def evaluate_model_with_timing(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    process = psutil.Process(os.getpid())

    start_time = time.time()
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    power_before = read_power()
    gpu_before = read_gpu_usage()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    total_time = time.time() - start_time
    cpu_after = process.cpu_percent(interval=0.1)
    mem_after = process.memory_info().rss / (1024 * 1024)
    power_after = read_power()
    gpu_after = read_gpu_usage()

    accuracy = 100 * correct / total
    avg_cpu = (cpu_before + cpu_after) / 2
    avg_mem = (mem_before + mem_after) / 2
    avg_power = (power_before + power_after) / 2 if power_before != -1 and power_after != -1 else -1
    avg_gpu = (gpu_before + gpu_after) / 2 if gpu_before != -1 and gpu_after != -1 else -1
    temperature = read_temperature()

    return accuracy, total_time, avg_cpu, avg_mem, temperature, avg_power, avg_gpu

# --- Run Evaluation ---
results = []

model_files = sorted([
    f for f in os.listdir(script_dir)
    if f.startswith("vegnet_pruned_") and f.endswith(".pt")
])
COOLDOWN_SECONDS = 10
EVAL_REPEATS = 3

for model_file in model_files:
    level = model_file.split("_")[-1].replace(".pt", "")

    raw_state = torch.load(os.path.join(script_dir, model_file), map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in raw_state.items()}
    hidden_size = state_dict["0.weight"].shape[0]
    model = create_vegnet(hidden_size=hidden_size).to(device)
    model.load_state_dict(state_dict)

    accs, times, cpus, mems, temps, powers, gpus = [], [], [], [], [], [], []

    for _ in range(EVAL_REPEATS):
        acc, inf_time, cpu, mem, temp, power, gpu = evaluate_model_with_timing(model, test_loader)
        accs.append(acc)
        times.append(inf_time)
        cpus.append(cpu)
        mems.append(mem)
        temps.append(temp)
        powers.append(power)
        gpus.append(gpu)

    avg_acc = sum(accs) / EVAL_REPEATS
    avg_time = sum(times) / EVAL_REPEATS
    avg_cpu = sum(cpus) / EVAL_REPEATS
    avg_mem = sum(mems) / EVAL_REPEATS
    avg_temp = sum(temps) / EVAL_REPEATS
    avg_power = sum(powers) / EVAL_REPEATS
    avg_gpu = sum(gpus) / EVAL_REPEATS

    num_nodes = model[0].weight.shape[0]

    results.append((level, avg_acc, avg_time, num_nodes, avg_cpu, avg_mem, avg_temp, avg_power, avg_gpu))
    print(f"Model {model_file} | Acc: {avg_acc:.2f}% | Time: {avg_time:.2f}s | "
          f"Nodes: {num_nodes} | CPU: {avg_cpu:.2f}% | RAM: {avg_mem:.2f}MB | Temp: {avg_temp:.1f}°C | "
          f"Power: {avg_power}mW | GPU: {avg_gpu}%")

    # Clean up model + cooldown
    del model
    torch.cuda.empty_cache()
    time.sleep(COOLDOWN_SECONDS)


# --- Save Results ---
output_path = os.path.join(script_dir, "results.txt")
with open(output_path, "w") as f:
    for level, acc, inf_time, nodes, cpu, mem, temp, power, gpu in results:
        f.write(f"Pruning: {level}% | Accuracy: {acc:.2f}% | Time: {inf_time:.4f}s | "
                f"Nodes: {nodes} | CPU: {cpu:.2f}% | RAM: {mem:.2f}MB | Temp: {temp:.1f}°C | "
                f"Power: {power}mW | GPU: {gpu}%\n")

print(f"\n✅ Done. Results saved to {output_path}")

