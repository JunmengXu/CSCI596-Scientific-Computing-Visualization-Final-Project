import pandas as pd
import matplotlib.pyplot as plt

df_local1 = pd.read_csv('analysis/compression_results_local.csv')
df_server = pd.read_csv('analysis/compression_results.csv')
# To analyze the compression performance under the same 'M,N' combination with different 'Ntasks' and 'CpusPerTask' as x-axis,
# let's choose a specific 'M,N' combination for this analysis. For example, let's use '4,16'.
df_server['MN_Combined'] = df_server.apply(lambda row: f'{row.M},{row.N}', axis=1)

selected_mn = '4,16'
df_selected = df_server[df_server['MN_Combined'] == selected_mn]

# Create a combined label for 'Ntasks' and 'CpusPerTask'
df_selected['TaskCpuComb'] = df_selected['Ntasks'].astype(str) + ',' + df_selected['CpusPerTask'].astype(str)

# Group by this new label and calculate the mean
grouped_selected = df_selected.groupby('TaskCpuComb').mean().reset_index()

# Now let's plot the compression time and compression ratio against this new combined label
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Plot for Compression Time
axes[0].plot(grouped_selected['TaskCpuComb'], grouped_selected['CompressionTime'], marker='o', linestyle='-')
axes[0].set_title('Average Compression Time for Ntasks,CpusPerTask (M,N=4,16)')
axes[0].set_xlabel('Ntasks,CpusPerTask')
axes[0].set_ylabel('Average Compression Time (s)')
axes[0].grid(True)

# To analyze the compression ratio in a way that highlights its relative stability across different 'Ntasks' and 'CpusPerTask' combinations,
# let's plot the compression ratios as a bar chart to emphasize the minimal variation.



# Plotting the compression ratios
axes[1].bar(grouped_selected['TaskCpuComb'], grouped_selected['CompressionRatio'], color='skyblue')

axes[1].set_xlabel('Ntasks,CpusPerTask')
axes[1].set_ylabel('Average Compression Ratio')
axes[1].set_title('Average Compression Ratio for Different Ntasks,CpusPerTask (M,N=4,16)')
axes[1].grid(axis='y')

# Adding a horizontal line to show the average compression ratio across all configurations
avg_ratio = grouped_selected['CompressionRatio'].mean()
axes[1].axhline(y=avg_ratio, color='red', linestyle='--', label=f'Average Ratio: {avg_ratio:.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('Extras/comparison1.png')
plt.show()



