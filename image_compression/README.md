To run the code

1. Compile

- Compile it in the local machine
`g++ -o image image.cpp`

- Compile it in CARC
`g++ -o image_omp image_omp.cpp -fopenmp -lm -std=c++11`

2. Run the following command

- In local machine
`./image input-filename output-filename M N`
Example:
`./image "./input.jpg" "./compressed.jpg" 36 16`

- In CARC
`sbatch pro.sl`
We can also configure the program in the last line
`./image_omp input-filename output-filename M N`