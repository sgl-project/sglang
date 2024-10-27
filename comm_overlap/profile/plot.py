import matplotlib.pyplot as plt
import numpy as np
import re

# Raw data
ag_data = """SOL time for GEMM(M=1024,N=49152,K=12288,TP=2):
torch #0: total 8.157 ms, gemm 5.198 ms, comm 2.960 ms
torch #1: total 8.152 ms, gemm 5.208 ms, comm 2.944 ms
te #0: total 11.678 ms, gemm 8.790 ms, comm 2.888 ms
te #1: total 11.674 ms, gemm 8.692 ms, comm 2.982 ms

SOL time for GEMM(M=2048,N=49152,K=12288,TP=2):
torch #0: total 15.988 ms, gemm 10.125 ms, comm 5.863 ms
torch #1: total 15.982 ms, gemm 9.981 ms, comm 6.001 ms
te #0: total 20.387 ms, gemm 14.551 ms, comm 5.836 ms
te #1: total 20.382 ms, gemm 14.472 ms, comm 5.910 ms

SOL time for GEMM(M=4096,N=49152,K=12288,TP=2):
torch #0: total 36.080 ms, gemm 23.305 ms, comm 12.775 ms
torch #1: total 37.126 ms, gemm 22.099 ms, comm 15.027 ms
te #0: total 27.741 ms, gemm 15.855 ms, comm 11.886 ms
te #1: total 27.757 ms, gemm 15.986 ms, comm 11.771 ms

SOL time for GEMM(M=8192,N=49152,K=12288,TP=2):
torch #0: total 83.937 ms, gemm 44.253 ms, comm 23.683 ms
torch #1: total 82.058 ms, gemm 43.760 ms, comm 24.297 ms
te #0: total 49.692 ms, gemm 25.764 ms, comm 23.928 ms
te #1: total 49.656 ms, gemm 26.092 ms, comm 23.563 ms"""

rs_data = """SOL time for GEMM(M=1024,N=12288,K=49152,TP=2):
torch #0: gemm 4.822 ms, comm 2.998 ms, total 7.820 ms
torch #1: gemm 4.844 ms, comm 2.977 ms, total 7.821 ms
te  #0: gemm 4.674 ms, comm 2.985 ms, total 7.659 ms
te  #1: gemm 4.686 ms, comm 2.968 ms, total 7.655 ms

SOL time for GEMM(M=2048,N=12288,K=49152,TP=2):
torch #0: gemm 11.239 ms, comm 6.169 ms, total 17.409 ms
torch #1: gemm 10.004 ms, comm 7.405 ms, total 17.409 ms
te  #0: gemm 9.592 ms, comm 5.992 ms, total 15.584 ms
te  #1: gemm 9.650 ms, comm 5.942 ms, total 15.592 ms

SOL time for GEMM(M=4096,N=12288,K=49152,TP=2):
torch #0: gemm 27.630 ms, comm 11.844 ms, total 49.474 ms
torch #1: gemm 27.836 ms, comm 21.637 ms, total 49.473 ms
te  #0: gemm 19.587 ms, comm 12.249 ms, total 31.836 ms
te  #1: gemm 19.444 ms, comm 12.392 ms, total 31.835 ms

SOL time for GEMM(M=8192,N=12288,K=49152,TP=2):
torch #0: gemm 82.039 ms, comm 24.045 ms, total 106.084 ms
torch #1: gemm 82.227 ms, comm 23.860 ms, total 106.087 ms
te  #0: gemm 39.027 ms, comm 24.092 ms, total 63.119 ms
te  #1: gemm 38.509 ms, comm 24.615 ms, total 63.125 ms"""

# Function to parse the raw data
def parse_data(data):
    results = []
    for match in re.finditer(r'SOL time for GEMM\(M=(\d+),N=\d+,K=\d+,TP=2\):\n((?:.*\n?)+?)(?=(SOL time for GEMM|$))', data):
        M = int(match.group(1))
        entries = match.group(2).strip().split('\n')
        torch_times = {'gemm': [], 'comm': []}
        te_times = {'gemm': [], 'comm': []}
        for entry in entries:
            method = 'torch' if 'torch' in entry else 'te'
            times = re.findall(r'gemm ([\d.]+) ms, comm ([\d.]+) ms', entry)
            if times:
                gemm_time, comm_time = map(float, times[0])
                if method == 'torch':
                    torch_times['gemm'].append(gemm_time)
                    torch_times['comm'].append(comm_time)
                else:
                    te_times['gemm'].append(gemm_time)
                    te_times['comm'].append(comm_time)
        results.append({
            'M': M,
            'torch_gemm': np.mean(torch_times['gemm']),
            'torch_comm': np.mean(torch_times['comm']),
            'te_gemm': np.mean(te_times['gemm']),
            'te_comm': np.mean(te_times['comm'])
        })
    return results

ag_results = parse_data(ag_data)
rs_results = parse_data(rs_data)

combined_results = rs_results + ag_results

x_labels = [f'{entry["M"]}\nRS' for entry in rs_results] + [f'{entry["M"]}\nAG' for entry in ag_results]
x = np.arange(len(x_labels))
bar_width = 0.2

fig, ax = plt.subplots(figsize=(8, 8))

for idx, entry in enumerate(combined_results):
    # pytorch
    ax.bar(x[idx] - bar_width / 2, entry['torch_gemm'], width=bar_width, color='blue', label='torch gemm' if idx == 0 else "")
    ax.bar(x[idx] - bar_width / 2, entry['torch_comm'], width=bar_width, bottom=entry['torch_gemm'], color='orange', label='torch comm' if idx == 0 else "")
    
    # te
    ax.bar(x[idx] + bar_width / 2, entry['te_gemm'], width=bar_width, color='green', hatch='xx', label='te gemm' if idx == 0 else "")
    ax.bar(x[idx] + bar_width / 2, entry['te_comm'], width=bar_width, bottom=entry['te_gemm'], color='red', hatch='//', label='te comm' if idx == 0 else "")
    
    # flux

ax.set_xlabel('M size, communication method')
ax.set_ylabel('Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('benchmark.png')
plt.show()
