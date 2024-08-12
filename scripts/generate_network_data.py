import os
import random
import pandas as pd

# Directory for storing network data
network_data_dir = 'data/network/'

if not os.path.exists(network_data_dir):
    os.makedirs(network_data_dir)

# Sample network traffic data generation
def generate_network_data(num_entries=1000):
    timestamps = pd.date_range('2024-01-01', periods=num_entries, freq='min')
    src_ips = ['192.168.1.1', '10.0.0.1', '172.16.0.1', '192.168.1.2', '10.0.0.2']
    dst_ips = ['8.8.8.8', '8.8.4.4', '192.168.1.3', '10.0.0.3', '172.16.0.2']
    protocols = ['TCP', 'UDP', 'ICMP']
    actions = ['ALLOW', 'BLOCK']
    values = [random.uniform(-1, 1) for _ in range(num_entries)]
    packet_size = [random.uniform(100, 1500) for _ in range(num_entries)]
    num_retries = [random.randint(0, 10) for _ in range(num_entries)]

    # Label based on action
    labels = [1 if action == 'BLOCK' else 0 for action in random.choices(actions, k=num_entries)]

    data = {
        'timestamp': random.choices(timestamps, k=num_entries),
        'src_ip': random.choices(src_ips, k=num_entries),
        'dst_ip': random.choices(dst_ips, k=num_entries),
        'protocol': random.choices(protocols, k=num_entries),
        'action': random.choices(actions, k=num_entries),
        'value': values,
        'packet_size': packet_size,
        'num_retries': num_retries,
        'label': labels
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(network_data_dir, 'synthetic_network.csv'), index=False)
    print(f'Synthetic network data generated at {os.path.join(network_data_dir, "synthetic_network.csv")}')

if __name__ == "__main__":
    generate_network_data()

