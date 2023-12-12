import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('/Users/gaoyiyuan/呆毛地盘/CSCI596/final_project/CSCI596-Scientific-Computing-Visualization-Final-Project/image_compression/compression_results.csv')

# Plot Compression Time
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(data['M'].astype(str) + ',' + data['N'].astype(str), data['CompressionTime'], marker='o')
plt.title('Compression Time')
plt.xlabel('Parameter (M, N)')
plt.ylabel('Time (seconds)')

# Plot Compression Ratio
plt.subplot(3, 1, 2)
plt.plot(data['M'].astype(str) + ',' + data['N'].astype(str), data['CompressionRatio'], marker='o', color='green')
plt.title('Compression Ratio')
plt.xlabel('Parameter (M, N)')
plt.ylabel('Ratio')

# Plot PSNR
plt.subplot(3, 1, 3)
plt.plot(data['M'].astype(str) + ',' + data['N'].astype(str), data['PSNR'], marker='o', color='red')
plt.title('PSNR')
plt.xlabel('Parameter (M, N)')
plt.ylabel('PSNR (dB)')

plt.tight_layout()
plt.show()
