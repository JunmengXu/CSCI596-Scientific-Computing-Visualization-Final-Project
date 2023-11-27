# CSCI596-Scientific-Computing-Visualization-Final-Project
This repository is for the Final Project of USC CSCI596 Scientific Computing &amp; Visualization

## Parallelize Image Compression with Vector Quantization

### Step 1: Understand Vector Quantization (VQ)
Vector Quantization is a technique used in image compression where blocks of pixels are replaced by a representative vector (codebook entry). The codebook is constructed by clustering similar vectors from the image.

Pros:
- Blindingly fast decompression (often faster than simply copying the uncompressed data, orders of magnitude faster than decompressing PNGs or JPEGs)
- Good quality at excellent compression ratios
- A flexible choice of the trade-off between compression ratio and fidelity

Cons:
- **Very slow compression: compressing any practical amount of art assets is definitely an overnight batch job.**
- Nonstandard, not widely supported in hardware.

![VQ-compressed "Lena," codebook size 2,048, compression time 10 minutes, 2.9 bits/pixel](https://eu-images.contentstack.com/v3/assets/blt95b381df7c12c15d/bltb9e5ff1c13355208/615541336e537906d1ff9d1b/fig9.png?width=828&quality=80&format=webply&disable=upscale "VQ-compressed Lena codebook size 2,048, compression time 10 minutes, 2.9 bits/pixel")


### Step 2: Goal &amp; Framework
This image compression method involves Codebook design so it is time-consuming, mostly in finding the nearest code vectors and calculating vector distances. In pursuit of the project's overarching objective, our focus is on enhancing the temporal efficiency of this method through the strategic application of parallelization techniques.
Decide whether we will use MPI, OpenMP, or a combination of both for parallelization.

### Step 3: Implementation

#### Step 3.1: Divide the Image
Divide the input image into smaller blocks or segments. Each block will be processed independently, enabling parallelization. The size of the blocks can be determined based on the available parallelization framework and the characteristics of the image.

![Lena in grayscale](https://content.iospress.com/media/ica/2017/24-3/ica-24-3-ica546/ica-24-ica546-g001.jpg "Lena in grayscale")

The diagonal line along which the density of the input vectors is concentrated is the x = y line. The areas on the diagram which would represent abrupt intensity changes from one pixel to the next are sparsely populated.

![Distribution of pairs of adjacent pixels from grayscale Lena](https://eu-images.contentstack.com/v3/assets/blt95b381df7c12c15d/bltc537b696f606f878/611e40f810f00930b842a689/fig2.png?width=828&quality=80&format=webply&disable=upscale "Distribution of pairs of adjacent pixels from grayscale Lena")


#### Step 3.2: Initialize Codebook
Initialize the codebook with representative vectors. This can be done by randomly selecting vectors from the input image or using a more sophisticated initialization method.

![Initialize Codebook](https://eu-images.contentstack.com/v3/assets/blt95b381df7c12c15d/blt474b9c163ee40553/611e40fabe258b650745de79/fig3.png?width=828&quality=80&format=webply&disable=upscale "Initialize Codebook")

#### Step 3.3: Cluster Vectors
Implement the clustering algorithm to group similar vectors together. Common algorithms include k-means clustering or hierarchical clustering. Ensure that the clustering algorithm is parallelized to handle different blocks concurrently.

![Cluster Vectors](https://eu-images.contentstack.com/v3/assets/blt95b381df7c12c15d/blt5bff97f755a7d09e/611e40fca6b36d3e6e0d9250/fig4.png?width=828&quality=80&format=webply&disable=upscale "Cluster Vectors")

#### Step 3.4: Update Codebook
After clustering, update the codebook by replacing each block with the representative vector of its cluster. This step may involve finding the centroid or median vector of each cluster.

#### Step 3.5: Quantize Image
Apply the updated codebook to quantize the entire image. Replace each block with its corresponding codebook entry.

![Quantize Image](https://github.com/JunmengXu/CSCI596-Scientific-Computing-Visualization-Final-Project/blob/main/Extras/encode%20image.png "Quantize Image")

#### Step 3.6: Encode the Image
Encode the quantized image, considering the indices of the codebook entries. The goal is to represent the image using a smaller number of bits.

#### Step 3.7: Parallelize Compression Steps
Identify which steps of the compression process can be parallelized. For example, the clustering and codebook update steps can be parallelized across multiple processors or threads. Ensure that communication and synchronization are handled correctly in a distributed environment.

![Compression Flow](https://github.com/JunmengXu/CSCI596-Scientific-Computing-Visualization-Final-Project/blob/main/Extras/compression%20flow.png "Compression Flow")


### Step 4: Optimize and Test
Optimize our parallelized code for performance. Experiment with different block sizes, cluster counts, and compression parameters to find the optimal configuration. Test the parallelized image compression on various images to ensure its effectiveness.

### Step 5: Evaluate Performance
Measure the performance of our parallelized image compression algorithm. Compare the execution time and compression ratio with a sequential implementation. Consider factors such as load balancing and scalability.

### Step 6: Document and Report
Document our parallelization strategy, including the algorithms used, data structures, and parallelization frameworks. Prepare a report summarizing our findings, including performance comparisons and any challenges faced during the parallelization process.

### Step 7: Present and Discuss
If required, present our parallelized image compression project to Professor and teaching staff. Discuss the innovation and efficiency gains achieved through parallelization. Based on feedback and our own evaluation, iterate on our parallelized image compression implementation to address any issues or enhance its performance further.

### Step 8: Summary
Conclude the project by summarizing its key findings and outcomes. Additionally, provide a comprehensive summary that encapsulates the project's achievements and areas for future improvement and development.
