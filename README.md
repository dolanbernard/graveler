# Graveler

[Results](#results)

This program simulates dice rolls in a scenario presented by Pikasprey on YouTube. The scenario takes place in Pokémon FireRed or LeafGreen. In the proposed scenario, a Graveler must be fully paralyzed for 177 turns in a row in order to win a battle to advance the game. The RNG method used in this exercise is not the same as that used in the actual game, so this program is not really a simulation. This is just a thought experiment put forward by Austin (@ShoddyCast) on YouTube. The only purpose it serves is to illustrate how unlikely it would be to be able to get out of the original situation.

## The Code

For this challenge, I decided to use Cuda as the problem lends itself to mass-parallelization. Time is measured by counting CPU time while the main kernel is running. Timing is stopped after the kernel finishes. This means that the time it takes to copy the results back from the GPU isn't measured. Only the actual computation time is reported.

### Differences in Computation

The original Python program [here](https://github.com/arhourigan/graveler) will continue rolling until either 231 rolls have been made or until 177 rolls have resulted in ones. In the actual game scenario, however, 177 consecutive ones must be rolled in order to advance. Since, as previously mentioned, this dice rolling algorithm does not actually represent the RNG method used by the game, it is unclear if this is an oversight or not. Since the probability calculations that follow in the video consider consecutive events, the default assumption I've made is that the intention is to consider consecutive events.

The current configuration of this program ends each attempt as soon as a non-one roll is encountered. To always roll 231 times each attempt, change the Cuda kernel call to `graveler_total` instead of `graveler_streaks` in [graveler.cu](src/graveler.cu). Results for both configurations are discussed in [Results](#results).

## Results

Using my RTX 2080 Ti, execution times were around 0.71 seconds for 1 billion attempts. The max number of ones rolled is usually between 10 and 13. You can probably get better times than me if you have a newer GPU. See section [Building and Running](#building-and-running) for info on how to configure the kernel parameters for your GPU and running the program.

The following results come from my attempts using an RTX 2080 Ti. You can probably get better times than me if you have a newer GPU. See section [Building and Running](#building-and-running) for info on how to configure the kernel parameters for your GPU and running the program.

I've separated the results by run configuration (detailed [here](#differences-in-computation)).

| Configuration | Typical Execution Time (s) | Typical Max Counts |
|---------------|----------------------------|--------------------|
| Streaks       | 0.71                       | 10-13              |
| Total         | 2.96                       | 75-80              |

## Building and Running

Building and running this program requires Cuda and, by extension, an Nvidia GPU. You may want/need to change the kernel configuration by changing the values at the top of [graveler.cu](src/graveler.cu).

```c
#define THREADS_PER_BLOCK (512)
#define BLOCKS_PER_GRID (131072)
```

This configuration worked well on my obsolete RTX 2080 Ti, but since you probably have a more modern GPU, you can probably squeeze out more performance by playing with these values.

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
