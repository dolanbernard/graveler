# Graveler

This program simulates dice rolls in a scenario presented by Pikasprey on YouTube. The scenario takes place in Pokémon FireRed or LeafGreen. In the proposed scenario, a Graveler must be fully paralyzed for 177 turns in a row in order to win a battle to advance the game. The RNG method used in this exercise is not the same as that used in the actual game, so this program is not really a simulation. This is just a thought experiment put forward by Austin (@ShoddyCast) on YouTube. The only purpose it serves is to illustrate how unlikely it would be to be able to get out of the original situation.

## Results

Using my RTX 2080TI, execution times were around 2.9 seconds for 1 billion attempts. The max number of 1s rolled is usually between 75 and 80. You can probably get better times than me if you have a newer GPU. See section [Building and Running](#building-and-running) for info on how to configure the kernel parameters for your GPU and running the program.

## The Code

For this challenge, I decided to use Cuda as the problem lends itself to mass-parallelization. Time is measured by counting CPU cycles while the main kernel is running. Timing is stopped after the kernel finishes. This means that the time it takes to copy the results back from the GPU isn't measured. Only the actual computation time is reported.

I didn't spend any time optimizing this yet. I may or may not put some more time into it because there are a few things that can be improved.

## Building and Running

Building and running this program requires Cuda and, by extension, an Nvidia GPU. You may want/need to change the kernel configuration by changing the values at the top of [graveler.cu](src/graveler.cu).

```c
#define THREADS_PER_BLOCK (512)
#define BLOCKS_PER_GRID (131072)
```

This configuration worked well on my obsolete RTX 2080TI, but since you probably have a more modern GPU, you can probably squeeze out more performance by playing with these values.

If you want to change the number of attempts, update this line:

```c
#define N (1000000000)
```

If you try very large numbers like 1000000000000, you'll run into memory issues. If you want to test computation for numbers this large, you can call the kernel in a loop. If you want to preserve and process results of each batch, move the data copying and processing loop into the kernel loop as well. Keep track of the max number in between batches and see if it beats the prior batch.

### Building

```bash
make all
```

### Running

```bash
make test
```
