import pandas as pd
import matplotlib.pyplot as plt

df_local = pd.read_csv('analysis/compression_results_local.csv')
df = pd.read_csv('analysis/compression_results.csv')
# First, let's create combined 'M,N' labels for use as the x-axis in our plots.
df_local['MN_Combined'] = df_local.apply(lambda row: f'{row.M},{row.N}', axis=1)
df['MN_Combined'] = df.apply(lambda row: f'{row.M},{row.N}', axis=1)

# Now, let's calculate the mean compression time and compression ratio for each 'M,N' combination.
mean_local = df_local.groupby('MN_Combined').mean().reset_index()
mean_server = df.groupby('MN_Combined').mean().reset_index()

# Create plots to compare the performance between local and server for each 'M,N' combination
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), sharex=True)

# Plot for Compression Time

axes[0].plot(mean_local['MN_Combined'], mean_local['CompressionTime'], marker='o', linestyle='-', color='red', label='Non-parallelized')
axes[0].plot(mean_server['MN_Combined'], mean_server['CompressionTime'], marker='x', linestyle='--', color='blue', label='Parallelized')
axes[0].set_xlabel('M,N Combinations')
axes[0].set_ylabel('Average Compression Time (s)')
axes[0].set_title('Average Compression Time (Non-parallelized vs Parallelized)')
axes[0].legend()
axes[0].grid(True)

# Plot for Compression Ratio
# To illustrate that the average compression ratio remains relatively consistent between local and server environments
# for the same 'M,N' combinations, let's create a bar chart comparing the compression ratios.

# First, let's merge the local and server datasets based on 'M,N' combinations
merged_mn_ratios = pd.merge(mean_local[['MN_Combined', 'CompressionRatio']], 
                            mean_server[['MN_Combined', 'CompressionRatio']], 
                            on='MN_Combined', suffixes=('_Local', '_Server'))

# Plotting the compression ratios for local and server
bar_width = 0.35
index = range(len(merged_mn_ratios))

axes[1].bar(index, merged_mn_ratios['CompressionRatio_Local'], bar_width, label='Local', color='skyblue')
axes[1].bar([i + bar_width for i in index], merged_mn_ratios['CompressionRatio_Server'], bar_width, label='Server', color='lightgreen')

axes[1].set_xlabel('M,N Combinations')
axes[1].set_ylabel('Average Compression Ratio')
axes[1].set_title('Average Compression Ratio Comparison: (Non-parallelized vs Parallelized)')
axes[1].set_xticks([i + bar_width / 2 for i in index])
axes[1].set_xticklabels(merged_mn_ratios['MN_Combined'], rotation=45)
axes[1].legend()
axes[1].grid(axis='y')

# Adding a line to show the average compression ratio across all configurations
avg_ratio_local = merged_mn_ratios['CompressionRatio_Local'].mean()
avg_ratio_server = merged_mn_ratios['CompressionRatio_Server'].mean()
axes[1].axhline(y=avg_ratio_local, color='blue', linestyle='--', label=f'Non-parallelized Avg Ratio: {avg_ratio_local:.2f}')
axes[1].axhline(y=avg_ratio_server, color='green', linestyle='--', label=f'Parallelized Avg Ratio: {avg_ratio_server:.2f}')
axes[1].legend()


plt.tight_layout()
plt.savefig('Extras/comparison.png')
plt.show()




