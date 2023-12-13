import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# First, let's load the two new datasets provided for local and server compression results.
df_local1 = pd.read_csv('analysis/compression_results_local.csv')
df_server = pd.read_csv('analysis/compression_results.csv')

# Now, we need to compare the average compression times and ratios for each image before and after optimization.
# For that, let's first process the dataframes to extract the necessary information.

# Extract the image names without file extension and compression parameters
df_local1['Image'] = df_local1['InputFilename'].apply(lambda x: x.split('/')[-1].split('.')[0])
df_server['Image'] = df_server['InputFilename'].apply(lambda x: x.split('/')[-1].split('.')[0])

# Calculate the average compression time and ratio for each image in both local and server datasets
local1_avg = df_local1.groupby('Image').agg({'CompressionTime': 'mean', 'CompressionRatio': 'mean'}).reset_index()
server_avg = df_server.groupby('Image').agg({'CompressionTime': 'mean', 'CompressionRatio': 'mean'}).reset_index()

# Merge the average results to compare them
comparison_df = pd.merge(local1_avg, server_avg, on='Image', suffixes=('_Local', '_Server'))

# Calculate the reduction in compression time and the change in compression ratio
comparison_df['TimeReduction'] = comparison_df['CompressionTime_Local'] - comparison_df['CompressionTime_Server']
comparison_df['RatioChange'] = comparison_df['CompressionRatio_Server'] - comparison_df['CompressionRatio_Local']

# Filter out the three images we are interested in
comparison_df = comparison_df[comparison_df['Image'].isin(['beach', 'capybara', 'Lenna'])]

# Display the comparison dataframe
comparison_df[['Image', 'TimeReduction', 'RatioChange']]


# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a bar plot for Time Reduction and Ratio Change for each image
fig, ax1 = plt.subplots(figsize=(14, 6))

# Bar plot for Time Reduction
sns.barplot(x='Image', y='TimeReduction', data=comparison_df, color='lightblue', label='Time Reduction', ax=ax1)
ax1.set_ylabel('Time Reduction (seconds)')

# Create another y-axis to plot Ratio Change using the same x-axis
ax2 = ax1.twinx()
sns.lineplot(x='Image', y='RatioChange', data=comparison_df, marker='o', color='orange', label='Ratio Change', ax=ax2)
ax2.set_ylabel('Ratio Change')

# Set plot title and legend
fig.suptitle('Compression Performance Comparison: Time Reduction and Ratio Change')
fig.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

plt.savefig('Extras/comparison2.png')
plt.show()


# Prepare data for the table
table_data = comparison_df[['Image', 'TimeReduction', 'RatioChange']]

# Create a table plot
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # Hide the axis

# Create a table
table = ax.table(
    cellText=table_data.values, 
    colLabels=table_data.columns, 
    loc='center',
    cellLoc='center'
)

# Set font size
table.set_fontsize(14)
table.scale(1.2, 1.5)

# Display the table
plt.savefig('Extras/comparison2_2.png')
plt.show()