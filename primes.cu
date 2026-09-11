#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <vector>

// 30-wheel: only test numbers coprime to 2, 3, and 5
// 8 valid residues per 30 integers = 26.7% density vs 50% odd-only (~2x fewer candidates)
// Combined with precomputed prime divisors, trial division starts at 7 (skips 2,3,5 entirely)
#define BATCH_SIZE  24000000   // divisible by 8 for 30-wheel alignment
#define THREADS     256
#define SIEVE_LIMIT 1000000    // supports numbers up to 10^12

__constant__ int WHEEL_OFFSETS[8] = {1, 7, 11, 13, 17, 19, 23, 29};

__global__ void checkWheel30(long long wheelBase, int batchSize,
                              const int* smallPrimes, int numPrimes,
                              unsigned long long* d_count, unsigned long long* d_largest) {
    __shared__ unsigned long long s_count;
    __shared__ unsigned long long s_largest;
    if (threadIdx.x == 0) { s_count = 0; s_largest = 0; }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        long long n = wheelBase + (long long)(idx / 8) * 30 + WHEEL_OFFSETS[idx % 8];

        // wheel guarantees n is coprime to 2,3,5 — start divisor search at 7 (smallPrimes[3])
        bool isPrime = (n > 1);
        if (isPrime) {
            for (int j = 3; j < numPrimes; j++) {
                long long p = smallPrimes[j];
                if (p * p > n) break;
                if (n % p == 0) { isPrime = false; break; }
            }
        }

        if (isPrime) {
            atomicAdd(&s_count, 1ULL);
            atomicMax(&s_largest, (unsigned long long)n);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (s_count > 0) atomicAdd(d_count, s_count);
        if (s_largest > 0) atomicMax(d_largest, s_largest);
    }
}

std::vector<int> buildSieve(int limit) {
    std::vector<bool> composite(limit + 1, false);
    std::vector<int> primes;
    for (int i = 2; i <= limit; i++) {
        if (!composite[i]) {
            primes.push_back(i);
            for (long long j = (long long)i * i; j <= limit; j += i)
                composite[j] = true;
        }
    }
    return primes;
}

std::string commas(unsigned long long n) {
    std::string s = std::to_string(n);
    int pos = (int)s.length() - 3;
    while (pos > 0) { s.insert(pos, ","); pos -= 3; }
    return s;
}

std::string sci(unsigned long long n) {
    char buf[32];
    sprintf(buf, "%.3e", (double)n);
    return buf;
}

std::string formatRuntime(double seconds) {
    int h = (int)(seconds / 3600);
    int m = (int)(seconds / 60) % 60;
    int s = (int)(seconds) % 60;
    char buf[16];
    sprintf(buf, "%02d:%02d:%02d", h, m, s);
    return buf;
}

int main() {
    printf("Building sieve...\n");
    auto primes = buildSieve(SIEVE_LIMIT);
    int numPrimes = (int)primes.size();
    printf("Done: %d primes up to %d (handles numbers up to ~10^12)\n\n", numPrimes, SIEVE_LIMIT);

    int*  d_smallPrimes;
    unsigned long long* d_count;
    unsigned long long* d_largest;
    unsigned long long  h_count, h_largest;

    cudaMalloc(&d_smallPrimes, numPrimes * sizeof(int));
    cudaMemcpy(d_smallPrimes, primes.data(), numPrimes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_count,   sizeof(unsigned long long));
    cudaMalloc(&d_largest, sizeof(unsigned long long));

    unsigned long long primeCount  = 3;   // pre-count 2, 3, 5
    unsigned long long largestPrime = 5;
    long long wheelBase = 0;              // first candidate: 0 + 1 = 1 (handled by n > 1 check)

    const long long COVERED_PER_BATCH = (long long)(BATCH_SIZE / 8) * 30;  // 90M per batch

    printf("5090 go BERRRRRRRR!\n\n");
    printf("More compute pls\n\n");

    auto programStart = std::chrono::steady_clock::now();

    while (true) {
        int blocks = (BATCH_SIZE + THREADS - 1) / THREADS;

        cudaMemset(d_count,   0, sizeof(unsigned long long));
        cudaMemset(d_largest, 0, sizeof(unsigned long long));

        auto t0 = std::chrono::steady_clock::now();

        checkWheel30<<<blocks, THREADS>>>(wheelBase, BATCH_SIZE,
                                          d_smallPrimes, numPrimes,
                                          d_count, d_largest);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_count,   d_count,   sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_largest, d_largest, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        auto t1 = std::chrono::steady_clock::now();
        double batchElapsed = std::chrono::duration<double>(t1 - t0).count();
        double totalElapsed = std::chrono::duration<double>(t1 - programStart).count();
        double speed = (double)COVERED_PER_BATCH / batchElapsed;

        primeCount += h_count;
        if (h_largest > largestPrime) largestPrime = h_largest;

        long long checkedUpTo = wheelBase + COVERED_PER_BATCH - 1;

        printf("[%s] Counted: %s | Primes: %s | Largest: %s (%s) | Speed: %s/sec          \r",
            formatRuntime(totalElapsed).c_str(),
            commas((unsigned long long)checkedUpTo).c_str(),
            commas(primeCount).c_str(),
            commas(largestPrime).c_str(),
            sci(largestPrime).c_str(),
            commas((unsigned long long)speed).c_str());
        fflush(stdout);

        wheelBase += COVERED_PER_BATCH;
    }

    cudaFree(d_smallPrimes);
    cudaFree(d_count);
    cudaFree(d_largest);
    return 0;
}
