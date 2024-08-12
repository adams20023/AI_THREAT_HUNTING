import os
import random
import pandas as pd

# Directory for storing log files
log_data_dir = 'data/logs/'

if not os.path.exists(log_data_dir):
    os.makedirs(log_data_dir)

# Sample log data generation
def generate_log_data(num_entries=1000):
    timestamps = pd.date_range('2024-01-01', periods=num_entries, freq='T')
    log_levels = ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
    messages = [
        'User login successful',
        'User login failed',
        'File accessed',
        'Unauthorized access attempt',
        'Service started',
        'Service stopped',
        'Configuration change detected'
    ]

    data = {
        'timestamp': random.choices(timestamps, k=num_entries),
        'log_level': random.choices(log_levels, k=num_entries),
        'message': random.choices(messages, k=num_entries),
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(log_data_dir, 'synthetic_logs.csv'), index=False)
    print(f'Synthetic log data generated at {os.path.join(log_data_dir, "synthetic_logs.csv")}')

if __name__ == "__main__":
    generate_log_data()

