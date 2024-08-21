# Graveler

This program simulates dice rolls in a scenario presented by Pikasprey on YouTube. The scenario takes place in Pokémon FireRed or LeafGreen. In the proposed scenario, a Graveler must be fully paralyzed for 177 turns in a row in order to win a battle to advance the game. The RNG method used in this exercise is not the same as that used in the actual game, so this program is not really a simulation. This is just a thought experiment put forward by Austin (@ShoddyCast) on YouTube. The only purpose it serves is to illustrate how unlikely it would be to be able to get out of the original situation.

## Results

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

### Building

```bash
make all
```

### Running

```bash
make test
```
