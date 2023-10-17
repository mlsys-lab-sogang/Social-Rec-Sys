import os
import re
import pandas as pd

# Paths and Seeds
log_directory = './logs'
seeds = [62, 365, 462, 583, 731, 765, 921, 1234, 12355]
user_lengths = [20, 30]
item_lengths = [300, 350, 400, 450, 500]

# Create a DataFrame that mirrors the structure of the spreadsheet
df = pd.DataFrame(columns=pd.MultiIndex.from_product([user_lengths, item_lengths, ['RMSE', 'MAE']], names=['user', 'item', 'metric']))

def parse_results(file_name):
    with open(file_name, 'r') as file:
        content = file.read()

        # Extract RMSE and MAE
        eval_results = re.search(r'\[Evaluation Results\].*RMSE: ([\d.]+).*MAE: ([\d.]+)', content, re.DOTALL)

        if eval_results:
            rmse = eval_results.group(1)
            mae = eval_results.group(2)
            return rmse, mae
        else:
            print(f"No match found in {file_name}")
            return None, None

# Iterate over seeds and lengths to populate the DataFrame
for seed in seeds:
    for user_length in user_lengths:
        for item_length in item_lengths:
            file_path = f'logs/log_seed_{seed}/epinions/train.model_compare_seed_{seed}_original_user_{user_length}_item_{item_length}.log'
            
            # Check if the file exists before attempting to parse it
            if os.path.exists(file_path):
                rmse, mae = parse_results(file_path)
                df.loc[seed, (user_length, item_length, 'RMSE')] = rmse
                df.loc[seed, (user_length, item_length, 'MAE')] = mae
            else:
                print(f"File {file_path} not found. Skipping...")

# Transpose the DataFrame
df = df.T

# Save the populated DataFrame to CSV
df.to_csv('results.csv')
