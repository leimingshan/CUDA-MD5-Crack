#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <driver_types.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#ifdef _WIN32
#include <Winsock2.h>
#pragma comment(lib,"ws2_32.lib")
typedef unsigned __int64 uint64_t;
#else
#include <sys/time.h>
#include <arpa/inet.h>
typedef unsigned long long uint64_t;
#endif

#ifndef UINT4
typedef unsigned long int UINT4;
#define uint UINT4
#endif

typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#define WORD_LENGTH 4

struct device_stats {
	unsigned char word[64];			// found word passed from GPU
	int hash_found;			// boolean if word is found
};

struct cuda_device {
	int device_id;
	struct cudaDeviceProp prop;

	int max_threads;
	int max_blocks;
	int shared_memory;

	int device_global_memory_len;

	void *host_memory;

	void *device_stats_memory;
	struct device_stats stats;

	unsigned int target_hash[4];
	uint64_t base_num;
	int word_length;

	// to be used for debugging
	void *device_debug_memory;
};

extern __shared__ unsigned int words[];			// shared memory where hash will be stored
__constant__ unsigned int target_hash[4];		// constant hash we will be searching for
__constant__ unsigned char char_set[63] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/*
__device__ unsigned int *format_shared_memory(unsigned int thread_id, unsigned int *memory) 
{
	unsigned int *shared_memory;
	unsigned int *global_memory;
	int i;

	// we need to get a pointer to our shared memory portion

	shared_memory = &words[threadIdx.x * 16];
	global_memory = &memory[thread_id * 16];

	for(i = 0; i < 16; i++) {
		shared_memory[i] = global_memory[i];
	}

	return shared_memory;
}
*/

/* F, G and H are basic MD5 functions: selection, majority, parity */

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
{(a) += F ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) \
{(a) += G ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

#define HH(a, b, c, d, x, s, ac) \
{(a) += H ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

#define II(a, b, c, d, x, s, ac) \
{(a) += I ((b), (c), (d)) + (x) + (UINT4)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

__device__ void md5(uint *in, uint *hash) 
{
	uint a, b, c, d;

	const uint a0 = 0x67452301;
	const uint b0 = 0xEFCDAB89;
	const uint c0 = 0x98BADCFE;
	const uint d0 = 0x10325476;

	a = a0;
	b = b0;
	c = c0;
	d = d0;

	/* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
	FF ( a, b, c, d, in[ 0], S11, 3614090360); /* 1 */
	FF ( d, a, b, c, in[ 1], S12, 3905402710); /* 2 */
	FF ( c, d, a, b, in[ 2], S13,  606105819); /* 3 */
	FF ( b, c, d, a, in[ 3], S14, 3250441966); /* 4 */
	FF ( a, b, c, d, in[ 4], S11, 4118548399); /* 5 */
	FF ( d, a, b, c, in[ 5], S12, 1200080426); /* 6 */
	FF ( c, d, a, b, in[ 6], S13, 2821735955); /* 7 */
	FF ( b, c, d, a, in[ 7], S14, 4249261313); /* 8 */
	FF ( a, b, c, d, in[ 8], S11, 1770035416); /* 9 */
	FF ( d, a, b, c, in[ 9], S12, 2336552879); /* 10 */
	FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */
	FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */
	FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */
	FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */
	FF ( c, d, a, b, in[14], S13, 2792965006); /* 15 */
	FF ( b, c, d, a, in[15], S14, 1236535329); /* 16 */

	/* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
	GG ( a, b, c, d, in[ 1], S21, 4129170786); /* 17 */
	GG ( d, a, b, c, in[ 6], S22, 3225465664); /* 18 */
	GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */
	GG ( b, c, d, a, in[ 0], S24, 3921069994); /* 20 */
	GG ( a, b, c, d, in[ 5], S21, 3593408605); /* 21 */
	GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */
	GG ( c, d, a, b, in[15], S23, 3634488961); /* 23 */
	GG ( b, c, d, a, in[ 4], S24, 3889429448); /* 24 */
	GG ( a, b, c, d, in[ 9], S21,  568446438); /* 25 */
	GG ( d, a, b, c, in[14], S22, 3275163606); /* 26 */
	GG ( c, d, a, b, in[ 3], S23, 4107603335); /* 27 */
	GG ( b, c, d, a, in[ 8], S24, 1163531501); /* 28 */
	GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */
	GG ( d, a, b, c, in[ 2], S22, 4243563512); /* 30 */
	GG ( c, d, a, b, in[ 7], S23, 1735328473); /* 31 */
	GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */

	/* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
	HH ( a, b, c, d, in[ 5], S31, 4294588738); /* 33 */
	HH ( d, a, b, c, in[ 8], S32, 2272392833); /* 34 */
	HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */
	HH ( b, c, d, a, in[14], S34, 4259657740); /* 36 */
	HH ( a, b, c, d, in[ 1], S31, 2763975236); /* 37 */
	HH ( d, a, b, c, in[ 4], S32, 1272893353); /* 38 */
	HH ( c, d, a, b, in[ 7], S33, 4139469664); /* 39 */
	HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */
	HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */
	HH ( d, a, b, c, in[ 0], S32, 3936430074); /* 42 */
	HH ( c, d, a, b, in[ 3], S33, 3572445317); /* 43 */
	HH ( b, c, d, a, in[ 6], S34,   76029189); /* 44 */
	HH ( a, b, c, d, in[ 9], S31, 3654602809); /* 45 */
	HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */
	HH ( c, d, a, b, in[15], S33,  530742520); /* 47 */
	HH ( b, c, d, a, in[ 2], S34, 3299628645); /* 48 */

	/* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
	II ( a, b, c, d, in[ 0], S41, 4096336452); /* 49 */
	II ( d, a, b, c, in[ 7], S42, 1126891415); /* 50 */
	II ( c, d, a, b, in[14], S43, 2878612391); /* 51 */
	II ( b, c, d, a, in[ 5], S44, 4237533241); /* 52 */
	II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */
	II ( d, a, b, c, in[ 3], S42, 2399980690); /* 54 */
	II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */
	II ( b, c, d, a, in[ 1], S44, 2240044497); /* 56 */
	II ( a, b, c, d, in[ 8], S41, 1873313359); /* 57 */
	II ( d, a, b, c, in[15], S42, 4264355552); /* 58 */
	II ( c, d, a, b, in[ 6], S43, 2734768916); /* 59 */
	II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */
	II ( a, b, c, d, in[ 4], S41, 4149444226); /* 61 */
	II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */
	II ( c, d, a, b, in[ 2], S43,  718787259); /* 63 */
	II ( b, c, d, a, in[ 9], S44, 3951481745); /* 64 */

	a += a0;
	b += b0;
	c += c0;
	d += d0;

	hash[0] = a;
	hash[1] = b;
	hash[2] = c;
	hash[3] = d;

	return;
}

__global__ void md5_cuda_calculate(struct device_stats *stats, unsigned int *debug_memory, uint64_t base, int word_length)
{
	unsigned int id;
	uint hash[4];
	int x;
	uint64_t value;
	unsigned char word[WORD_LENGTH];
	unsigned int orig_input_length;

	int i, j;

	unsigned char md5_padded[64];
	uint md5_padded_int[16];

	id = (blockIdx.x * blockDim.x) + threadIdx.x;		// get our thread unique ID in this run

	value = base + id;

	// calculate the word according to value
	memset(word, char_set[0], WORD_LENGTH);
	if (value < 62) {
		word[word_length - 1] = char_set[value];
		return;
	} else {
		uint64_t result = value;
		int i = word_length - 1;

		while (i >= 0 && result > 0)
		{
			uint64_t val = result % 62;
			word[i] = char_set[val];
			result /= 62;
			i--;
		}
	}

	// md5 padding
	orig_input_length = word_length * 8;
	memset(md5_padded, 0, 64);

	for(x = 0; x < word_length && x < 56; x++) {
		md5_padded[x] = word[x];
	}

	md5_padded[x] = 0x80;

	// add the original length, ignore the high 4 bytes (little Endian)
	memcpy(md5_padded + 56, (void *)&orig_input_length, 4);
	
	for (i = 0, j = 0; j < 64; i++, j += 4)
		md5_padded_int[i] = ((uint)md5_padded[j]) | (((uint)md5_padded[j+1]) << 8) |
		(((uint)md5_padded[j+2]) << 16) | (((uint)md5_padded[j+3]) << 24);


	md5(md5_padded_int, hash);	// actually calculate the MD5 hash

#ifdef DEBUG
	// passes the computed hashes into debug memory
	for (x = 0; x < 4; x++) 
		debug_memory[(id * 4) + x] = hash[x];
#endif

	if (hash[0] == target_hash[0] && hash[1] == target_hash[1] 
		&& hash[2] == target_hash[2] && hash[3] == target_hash[3]) {
		// !! WE HAVE A MATCH !!
		stats->hash_found = 1;
		// copy the matched word to stats memory
		for(x = 0; x < word_length; x++) 
			stats->word[x] = word[x];
		stats->word[x] = '\0';
	}
}

static void md5_calculate(struct cuda_device *device) 
{
	cudaEvent_t start, stop;
	float time;

	// put our target hash into the GPU constant memory as this will not change 
	// (and we can't spare shared memory for speed)
	if (cudaMemcpyToSymbol(target_hash, device->target_hash, 16, 0, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Error initializing constants\n");
		return;
	}

#ifdef GPU_BENCHMARK
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	cudaThreadSynchronize();
#endif

	md5_cuda_calculate <<< device->max_blocks, device->max_threads, device->shared_memory >>> ((struct device_stats *)device->device_stats_memory, (unsigned int *)device->device_debug_memory, device->base_num, device->word_length);

#ifdef GPU_BENCHMARK
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("CUDA kernel took %fms to calculate %d x %d (%d) hashes\n", time, device->max_blocks, device->max_threads, device->max_blocks * device->max_threads);
	// print GPU stats here
#endif

}

int get_cuda_device(struct cuda_device *device) 
{
	int device_count;

	if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
		// cuda not supported
		return -1;
	}

	while(device_count >= 0) {
		if (cudaGetDeviceProperties(&device->prop, device_count) == cudaSuccess) {
			// we have found our device
			device->device_id = device_count;
			return device_count;
		}

		device_count--;
	}

	return -1;
}

#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256

int calculate_cuda_params(struct cuda_device *device) 
{
	int max_threads;
	int max_blocks;
	int shared_memory;

	max_threads = device->prop.maxThreadsPerBlock;
	shared_memory = device->prop.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;
	
	// calculate the most threads that we can support optimally
	//while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) { max_threads--; } 

	// now we spread our threads across blocks 
	max_blocks = 54;		// we need to calculate & adjust this !

	device->max_threads = max_threads;		// most threads we support
	device->max_blocks = max_blocks;		// most blocks we support
	device->shared_memory = shared_memory;		// shared memory required

	// now we need to have (device.max_threads * device.max_blocks) number of words in memory for the graphics card
	device->device_global_memory_len = (device->max_threads * device->max_blocks) * 16;

	return 1;
}

/*********************************************************************
 *  TAKEN FROM: http://www.codeproject.com/KB/string/hexstrtoint.aspx
 *
 *  has been slightly modified
 *
 *  Many Thanks Anders Molin
 *********************************************************************/

struct CHexMap {
	char chr;
	int value;
};

#define true 1
#define false 0

#define HexMapL 22

int _httoi(const char *value) 
{
	struct CHexMap HexMap[HexMapL] = {
		{'0', 0}, {'1', 1},
		{'2', 2}, {'3', 3},
		{'4', 4}, {'5', 5},
		{'6', 6}, {'7', 7},
		{'8', 8}, {'9', 9},
		{'A', 10}, {'B', 11},
		{'C', 12}, {'D', 13},
		{'E', 14}, {'F', 15},
		{'a', 10}, {'b', 11},
		{'c', 12}, {'d', 13},
		{'e', 14}, {'f', 15},
	};
	int i;

	char *mstr = strdup(value);
	char *s = mstr;
	int result = 0;
	int found = false;
	int firsttime = true;

	if (*s == '0' && *(s + 1) == 'X') {
		s += 2;
	}

	while (*s != '\0') {
		for (i = 0; i < HexMapL; i++) {

			if (*s == HexMap[i].chr) {

				if (!firsttime) {
					result <<= 4;
				}
				
				result |= HexMap[i].value;
				found = true;
				break;
			}
		}

		if (!found) {
			break;
		}

		s++;
		firsttime = false;
	}

  free(mstr);
  return result;
}

/*************************************************************************/


/* Encodes input (UINT4) into output (unsigned char). Assumes len is
  a multiple of 4.
 */
static void Encode(unsigned char *output, unsigned int *input, unsigned int len)
{
  unsigned int i, j;

  for (i = 0, j = 0; j < len; i++, j += 4) {
    output[j] = (unsigned char)(input[i] & 0xff);
    output[j+1] = (unsigned char)((input[i] >> 8) & 0xff);
    output[j+2] = (unsigned char)((input[i] >> 16) & 0xff);
    output[j+3] = (unsigned char)((input[i] >> 24) & 0xff);
  }
}

static void print_info(void)
{
	printf("CUDA-MD5-Crack programmed by LMS-BUPT\n\n");
	return;
}

#define ARG_MD5 1
#define ARG_COUNT 2

int main(int argc, char **argv) 
{
	int x, i;
	struct cuda_device device;
	char input_hash[4][9];

	int min_length = 1;
	int max_length = WORD_LENGTH;
	int word_length;

	print_info();

	if (argc != ARG_COUNT) {
		printf("Usage: %s MD5_HASH\n", argv[0]);
		return -1;
	}

	// select our CUDA device
	if (get_cuda_device(&device) == -1) {
		printf("No Cuda Device Installed\n");
		return -1;
	}

	// we now need to calculate the optimal amount of threads to use for this card
	calculate_cuda_params(&device);

	// now we input our target hash
	if (strlen(argv[ARG_MD5]) != 32) {
		printf("Not a valid MD5 Hash (should be 32 bytes and only Hex Chars\n");
		return -1;
	}
	
	// we split the input hash into 4 blocks
	memset(input_hash, 0, sizeof(input_hash));

	for(x = 0; x < 4; x++) {
		strncpy(input_hash[x], argv[ARG_MD5] + (x * 8), 8);		
		device.target_hash[x] = htonl(_httoi(input_hash[x]));
	}

	// allocate the 'stats' that will indicate if we are successful in cracking
	if (cudaMalloc(&device.device_stats_memory, sizeof(struct device_stats)) != cudaSuccess) {
		printf("Error allocating memory on device (stats memory)\n");
		return -1;
	}

	// allocate debug memory if required
	if (cudaMalloc(&device.device_debug_memory, device.device_global_memory_len) != cudaSuccess) {
		printf("Error allocating memory on device (debug memory)\n");
		return -1;
	}

	// make sure the stats are clear on the device
	if (cudaMemset(device.device_stats_memory, 0, sizeof(struct device_stats)) != cudaSuccess) {
		printf("Error Clearing Stats on device\n");
		return -1;
	}
	
	// this is our host memory that we will copy to the graphics card
	if ((device.host_memory = malloc(device.device_global_memory_len)) == NULL) {
		printf("Error allocating memory on host\n");
		return -1;
	}

	#ifdef BENCHMARK
		// these will be used to benchmark
		int counter = 0;
		struct timeval start, end;
		// start timer
		gettimeofday(&start, NULL);
	#endif

	for (word_length = min_length; word_length <= max_length; word_length++) {
		long max_num = (long)pow(62, word_length);
		int max_thread_num = device.max_blocks * device.max_threads;
		int batch_num = ceil(max_num / max_thread_num);

		unsigned int *m;
		int j;

	#ifdef BENCHMARK
		counter+=max_num;;		// increment counter for this word
	#endif

		for (j = 0; j < batch_num; j++) {
			device.base_num = j * max_thread_num;
			device.word_length = word_length;

			md5_calculate(&device);		// launch the kernel of the CUDA device
			
			if (cudaMemcpy(&device.stats, device.device_stats_memory, sizeof(struct device_stats), cudaMemcpyDeviceToHost) != cudaSuccess) {
				printf("Error Copying STATS from the GPU\n");
				return -1;
			}


			#ifdef DEBUG
			// For debug, we will receive the hashes for verification
				memset(device.host_memory, 0, device.device_global_memory_len);
				if (cudaMemcpy(device.host_memory, device.device_debug_memory, device.device_global_memory_len, cudaMemcpyDeviceToHost) != cudaSuccess) {
					printf("Error Copying words from GPU\n");
					return -1;
				}

				cudaThreadSynchronize();

				// prints out the debug hash'es
				printf("MD5 registers:\n\n");
				m = (unsigned int *)device.host_memory;
				for(x = 0; x < (device.max_blocks * device.max_threads); x++) {
					unsigned char output[32];
					printf("word-value: [%lld] --\n", x + device.base_num);
					
					Encode(output, m + x * 4, 32);
					printf("%d\n", m[x*4]);
					printf("md5: [%s]", output);
					printf("-------------------\n\n");
				}
			#endif

			cudaThreadSynchronize();

			if (device.stats.hash_found == 1) {
				printf("WORD FOUND: [%s]\n", (char *)device.stats.word);
				break;
			}
		}
	}

	if (device.stats.hash_found != 1) {
		printf("No word could be found for the provided MD5 hash\n");
	}

	#ifdef BENCHMARK 
		gettimeofday(&end, NULL);
		long long time = (end.tv_sec * (unsigned int)1e6 + end.tv_usec) - (start.tv_sec * (unsigned int)1e6 + start.tv_usec);
		printf("Time taken to check %d hashes: %f seconds\n", counter, (float)((float)time / 1000.0) / 1000.0);
		printf("Words per second: %lld\n", counter / (time / 1000) * 1000);
	#endif

	cudaDeviceReset();
	return 0;
}

