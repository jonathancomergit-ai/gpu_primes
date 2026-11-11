# gpu_primes

Ever wondered how many prime numbers you can chew through if you just throw a whole
graphics card at the problem? That's this!

I actually started this in Java. A basic `isPrime` that checks divisors up to the square
root, then multithreading with `ExecutorService`, then a Sieve of Eratosthenes so the small
divisors were precomputed instead of recalculated a million times. With 24 threads it hit
**7,678,896 checks per second**, and honestly I was pretty pleased with myself.

Then I remembered the RTX 5090 sitting in the same machine doing absolutely nothing, with
21,760 cores on it. So here we are in CUDA!

## The tricks in here

- **A 30-wheel.** Instead of testing every number, or even just the odd ones, it only tests
  numbers that aren't divisible by 2, 3, or 5. That's 8 valid numbers out of every 30, so
  26.7% of them instead of 50% for odds-only. Roughly HALF the work before you've even started.
- **Precomputed small primes**, so trial division can start at 7 and skip 2, 3 and 5 completely.
- **Big batches** (24 million numbers at a time, kept divisible by 8 so the wheel lines up)
  because the GPU is happiest when you hand it an enormous pile of work at once.

## Building it

This is the bit that got me (ask why I know this):

```
nvcc primes.cu -o primes.exe
```

You have to run that from the **x64 Native Tools Command Prompt for VS 2022**, NOT a normal
terminal and NOT the VS Code one. Otherwise `nvcc` can't find `cl.exe` and gives you a fatal
error that tells you nothing useful.

Then just run `primes.exe` and watch it go.

## Notes

`primes.txt` (the output) and the compiled `.exe` aren't in here on purpose - the output file
got to over a gigabyte and GitHub was NOT going to be happy about that.

## Licence

MIT, see [LICENSE](LICENSE).
