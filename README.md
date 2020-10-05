# Coursework 2 Concurrent and Parallel Systems - Napier University 2018/2019

## Aim
Implement a JPEG compression algorithm by applying parallelisation techniques where possible, evaluating their effectiveness


## Implementation 
The project utilizes an initial JPEG compression program available on GitHub (Kornel) which is based on the code by Rich Geldreich (Geldreich), with some extra enhancements. The initial program provides multiple steps in which parallelization has not been
attempted: read image from memory, decompress, check the compression success.
It has been carried out a Profiler to analyse possible areas of high CPU usage where it could have been attempted parallelisation.

![profiler](https://github.com/Willyees/cw2-CPS/blob/assets/assets/inti_high_main_f.png)

## Methodology
Times have been taken by utilizing the clean algorithm on different images with specific sizes.
Initially OpenMP was attempted to ascertain if a speed up would be possibly achieved. 
Later, other techniques as futures and OpenCl (with variants) were utilized. The different techniques’ code can be viewed in the other branches of the GitHub project.
A comparison has been carried out of times taken to compress different types of images (number of colours in the image) and different image sizes

![times](https://github.com/Willyees/cw2-CPS/blob/assets/assets/times.png)

Additionally, they have been compared by their calculated speed up and quality percentage compression applied.

A more in depth discussion can be found in the [report](../report_cw2.pdf).

## Conclusions
In general JPEG compression can achieve improvement utilizing parallelization, so if the main
focus is in performing faster, it is a viable option. In case the cost of utilizing parallelization is
considered, it might not be worth because the maximum efficiency obtained was 27%. This is because it was noticed that the algorithm has a few interdependencies which create sequential parts of code that cannot be parallelized.

![speedup](https://github.com/Willyees/cw2-CPS/blob/assets/assets/speedup.png)

### Research JPEG encoder

Codename "Nether Poppleton"

More information about the JPEG algorithm code used can be found at kornelski's jpeg-compressor [GitHub](https://github.com/kornelski/jpeg-compressor) project page.
