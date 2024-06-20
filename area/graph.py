import matplotlib.pyplot as plt

# Read the timing data from the log file
processors = []
times = []
try:
    with open("timing_log.txt", "r") as log_file:
        for line in log_file:
            proc, time = line.strip().split(",")
            processors.append(int(proc))
            times.append(float(time))
except FileNotFoundError:
    print("The file timing_log.txt was not found.")
    exit()

# Check if the data is being read correctly
print("Processors:", processors)
print("Times:", times)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(processors, times, marker='o')
plt.title("Elapsed Time vs. Number of Processors")
plt.xlabel("Number of Processors")
plt.ylabel("Elapsed Time (seconds)")
plt.grid(True)
plt.savefig('plot.png')
  # Ensure that the plot is displayed
