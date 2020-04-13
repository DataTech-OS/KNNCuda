#include <Windows.h>
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

using namespace std;

void KNearestNeighborsCPU(float3 *dataArray, int *result, int cnt);
void KNearestNeighborsCPUTreeSearch(float3 *dataArray, int *result, int cnt);
__global__ void KNearestNeighborsGPU(float3 *dataArray, int *result, int cnt);
__global__ void KNearestNeighborsGPUTreeSearch(float3 *dataArray, int *result, int cnt);

int cnt = 10000;

int main(int argc, char **argv)
{
	clock_t init, end;

	// generate the data
	srand(time(NULL));

	int timt = 0;
	float3 *dataArray = new float3[cnt];
	int *result = new int[cnt];

	for (int i = 0; i < cnt; i++)
	{
		dataArray[i].x = (rand() / 10000) - 5000;
		dataArray[i].y = (rand() / 10000) - 5000;
		dataArray[i].z = (rand() / 10000) - 5000;
	}
	// first check the speed of the algorithm takes on the cpu
	for (int i = 0; i < 10; i++)
	{
		init = clock();
		KNearestNeighborsCPU(dataArray, result, cnt);
		end = clock();
		timt += end - init;

		cout << "Iteration number " << i << " took " << end - init << " milliseconds" << endl;
	}
	cout << "[+] The non optimized version of the algorithm on the CPU takes " << timt / 10 << " milliseconds" << endl;
	timt = 0;

	for (int i = 0; i < 10; i++)
		cout << i << " - " << result[i] << endl;

	memset(result, 0, cnt);
	goto n;

	// second check the speed of a k-d tree search optimization of the algorithm on the cpu
	for (int i = 0; i < 10; i++)
	{
		init = clock();
		KNearestNeighborsCPUTreeSearch(dataArray, result, cnt);
		end = clock();
		timt += end - init;
	}
	cout << "[+] The optimized version of the algorithm on the CPU takes " << timt / 10 << " milliseconds" << endl;
	timt = 0;

n:
	// allocate and copy memory to the gpu
	float3 *deviceData;
	int *deviceResult;

	if (cudaMalloc(&deviceData, sizeof(float3)*cnt) != cudaSuccess)
	{
		cout << "[+] Unable to allocate GPU memory" << endl;
		return -1;
	}
	if (cudaMalloc(&deviceResult, sizeof(int)*cnt) != cudaSuccess)
	{
		cout << "[+] Unable to allocate GPU memory" << endl;
		return -2;
	}
	if (cudaMemcpy(deviceData, dataArray, sizeof(float3)*cnt, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "[+] Error in moving memory to the GPU" << endl;
		return -3;
	}

	// third check the simple implementation speed on the gpu
	for (int i = 0; i < 10; i++)
	{
		init = clock();
		KNearestNeighborsGPU<<< (cnt / 128) + 1, 128 >>>(deviceData, deviceResult, cnt);
		cudaMemcpy(result, deviceResult, sizeof(int)*cnt, cudaMemcpyDeviceToHost);
		end = clock();
		timt += end - init;

		cout << "Iteration number " << i << " took " << end - init << " milliseconds" << endl;
	}
	cout << "[+] The non optimized version of the algorithm on the GPU takes " << timt / 10 << " milliseconds" << endl;
	timt = 0;

	for (int i = 0; i < 10; i++)
		cout << i << " - " << result[i] << endl;
	cin.get();

	return 0;

	// fourth check the time the k-d version takes to run on the gpu
	for (int i = 0; i < 50; i++)
	{
		init = clock();
		//KNearestNeighborsGPUTreeSearch(dataArray, result, cnt);
		end = clock();
		timt += end - init;
	}
	cout << "[+] The optimized version of the algorithm on the GPU takes " << timt / 50 << " milliseconds" << endl;

	return 0;
}

// non optimized cpu algorithm
void KNearestNeighborsCPU(float3 *dataArray, int *result, int cnt)
{
	for (int i = 0; i < cnt; i++)
	{
		float minimumDist = 3.4028234664e38f, distance = 0;
		for (int j = 0; j < cnt; j++)
		{
			if (i != j)
			{
				distance = (dataArray[i].x - dataArray[j].x) * (dataArray[i].x - dataArray[j].x);
				distance += (dataArray[i].y - dataArray[j].y) * (dataArray[i].y - dataArray[j].y);
				distance += (dataArray[i].z - dataArray[j].z) * (dataArray[i].z - dataArray[j].z);

				if (distance < minimumDist)
				{
					minimumDist = distance;
					result[i] = j;
				}
			}
		}
	}
}

// optimized cpu algorithm
void KNearestNeighborsCPUTreeSearch(float3 *dataArray, int *result, int cnt)
{

}

// non optimized gpu algorithm
__global__ void KNearestNeighborsGPU(float3 *dataArray, int *result, int cnt)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < cnt)
	{
		float3 point = dataArray[id];
		float minimumDist = 3.4028234664e38f, distance = 0;

		for (int j = 0; j < cnt; j++)
		{
			if (id != j)
			{
				float3 current = dataArray[j];
				distance = (point.x - current.x) * (point.x - current.x);
				distance += (point.y - current.y) * (point.y - current.y);
				distance += (point.z - current.z) * (point.z - current.z);
				
				if (distance < minimumDist)
				{
					minimumDist = distance;
					result[id] = j;
				}
			}
		}
	}
}

// optimized gpu algorithm
__global__ void KNearestNeighborsGPUTreeSearch(float3 *dataArray, int *result, int cnt)
{

}