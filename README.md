# CSCI596-Scientific-Computing-Visualization-Final-Project
This repository is for the Final Project of USC CSCI596 Scientific Computing &amp; Visualization

## Parallelize Image Compression with Vector Quantization

### Step 1: Understand Vector Quantization (VQ)
Vector Quantization is a technique used in image compression where blocks of pixels are replaced by a representative vector (codebook entry). The codebook is constructed by clustering similar vectors from the image.

### Step 2: Choose a Parallelization Framework
Decide whether we will use MPI, OpenMP, or a combination of both for parallelization.

### Step 3: Implementation

#### Step 3.1: Divide the Image
Divide the input image into smaller blocks or segments. Each block will be processed independently, enabling parallelization. The size of the blocks can be determined based on the available parallelization framework and the characteristics of the image.

#### Step 3.2: Initialize Codebook
Initialize the codebook with representative vectors. This can be done by randomly selecting vectors from the input image or using a more sophisticated initialization method.

#### Step 3.3: Cluster Vectors
Implement the clustering algorithm to group similar vectors together. Common algorithms include k-means clustering or hierarchical clustering. Ensure that the clustering algorithm is parallelized to handle different blocks concurrently.

#### Step 3.4: Update Codebook
After clustering, update the codebook by replacing each block with the representative vector of its cluster. This step may involve finding the centroid or median vector of each cluster.

#### Step 3.5: Quantize Image
Apply the updated codebook to quantize the entire image. Replace each block with its corresponding codebook entry.

#### Step 3.6: Encode the Image
Encode the quantized image, considering the indices of the codebook entries. The goal is to represent the image using a smaller number of bits.

#### Step 3.7: Parallelize Compression Steps
Identify which steps of the compression process can be parallelized. For example, the clustering and codebook update steps can be parallelized across multiple processors or threads. Ensure that communication and synchronization are handled correctly in a distributed environment.

### Step 4: Optimize and Test
Optimize our parallelized code for performance. Experiment with different block sizes, cluster counts, and compression parameters to find the optimal configuration. Test the parallelized image compression on various images to ensure its effectiveness.

### Step 5: Evaluate Performance
Measure the performance of our parallelized image compression algorithm. Compare the execution time and compression ratio with a sequential implementation. Consider factors such as load balancing and scalability.

### Step 6: Document and Report
Document our parallelization strategy, including the algorithms used, data structures, and parallelization frameworks. Prepare a report summarizing our findings, including performance comparisons and any challenges faced during the parallelization process.

### Step 7: Present and Discuss
If required, present our parallelized image compression project to Professor and teaching staff. Discuss the innovation and efficiency gains achieved through parallelization.

### Step 8: Iterate and Improve
Based on feedback and our own evaluation, iterate on our parallelized image compression implementation to address any issues or enhance its performance further.
