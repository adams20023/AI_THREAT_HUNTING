import os
import random
import pandas as pd

# Directory for storing log files
log_data_dir = 'data/logs/'

if not os.path.exists(log_data_dir):
    os.makedirs(log_data_dir)

# Sample log data generation
def generate_log_data(num_entries=1000):
    timestamps = pd.date_range('2024-01-01', periods=num_entries, freq='min')
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
    values = [random.uniform(-1, 1) for _ in range(num_entries)]
    session_lengths = [random.uniform(1, 100) for _ in range(num_entries)]
    num_failed_attempts = [random.randint(0, 5) for _ in range(num_entries)]

    # Label based on log level
    labels = [1 if log_level in ['CRITICAL', 'ERROR', 'WARNING'] else 0 for log_level in random.choices(log_levels, k=num_entries)]

    data = {
        'timestamp': timestamps,
        'log_level': random.choices(log_levels, k=num_entries),
        'message': random.choices(messages, k=num_entries),
        'value': values,
        'session_length': session_lengths,
        'num_failed_attempts': num_failed_attempts,
        'label': labels
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(log_data_dir, 'synthetic_logs.csv'), index=False)
    print(f'Synthetic log data generated at {os.path.join(log_data_dir, "synthetic_logs.csv")}')

if __name__ == "__main__":
    generate_log_data()

