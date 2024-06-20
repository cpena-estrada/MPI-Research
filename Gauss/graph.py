import matplotlib.pyplot as plt

def read_timing_results(file_path):
    processors = []
    times = []
    with open(file_path, 'r') as f:
        for line in f:
            proc, time = line.strip().split(',')
            processors.append(int(proc))
            times.append(float(time))
    return processors, times

def plot_timing_results(processors, times):
    plt.figure(figsize=(10, 6))
    plt.plot(processors, times, marker='o')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs. Number of Processes')
    plt.grid(True)
    plt.savefig('guass_plot200x200.png')

if __name__ == "__main__":
    processors, times = read_timing_results('timing_results200x200.txt')
    plot_timing_results(processors, times)
