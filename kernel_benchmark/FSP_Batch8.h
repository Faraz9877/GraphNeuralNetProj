#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M2 32
#define VTYPE int4
#define SPARSE_VEC 8

// This file is added to explore the implementation for Fine-grained Sparsity on CUDA Cores
int A2_inspection_matrix_batch8(
    half* A2, int M, int K, int K_cut, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    assert(M % M2 == 0);

    int *offsets = new int[M / M2];

    int padding = SPARSE_VEC * K_cut;

    for (int m = 0; m < M / M2; m++) {
        int max = 0;
        for (int m2 = 0; m2 < M2; m2++) {
            int counter = 0;
            for (int k = 0; k < K; k++) {
                if (__half2float(A2[(m * M2 + m2) * K + k]) != 0.0f) {
                counter++;
                }
            }
            if (counter > max) {
                max = counter;
            }
        }

        if (max % (padding) != 0)
            max = max + ((padding) - max % (padding));
        offsets[m] = max;
    }

    int size = 0;
    for (int m = 0; m < M / M2; m++)
    {
        size += offsets[m] * M2;
    }

    half *d_A2_vals = new half[size * sizeof(half)];
    ushort *matrix_indeces = new ushort[size * sizeof(ushort)];

    int index = 0;

    for (int m = 0; m < M / M2; m++) {

        int k_size = offsets[m];

        half * buffer_values = new half[M2 * k_size];
        ushort * buffer_indeces = new ushort[M2 * k_size];

        for (int m2 = 0; m2 < M2; m2++) {
            int counter = 0;
            for (int k = 0; k < K; k++) {
                half value_half = A2[(m * M2 + m2) * K + k];
                float value_float = __half2float(A2[(m * M2 + m2) * K + k]);
                if (value_float != 0.0f) {
                    buffer_values[m2 * k_size + counter] = value_half;
                    buffer_indeces[m2 * k_size + counter] = k;
                    counter++;
                }
            }

            if (counter < k_size) {
                for (int i = counter; i < k_size; i++) {
                    buffer_values[m2 * k_size + i] = 0;
                    buffer_indeces[m2 * k_size + i] = 0;
                }
            }

        }

        for (int k = 0; k < k_size / SPARSE_VEC; k++) {
            for (int m2 = 0; m2 < M2; m2++) {
                for (int k2 = 0; k2 < SPARSE_VEC; k2++) {
                d_A2_vals[index] = buffer_values[m2 * k_size + k * SPARSE_VEC + k2];
                matrix_indeces[index] = buffer_indeces[m2 * k_size + k * SPARSE_VEC + k2];
                index++;
                }
            }
        }

        delete[] buffer_values;
        delete[] buffer_indeces;

    }

    int *matrix_offsets = new int[M / M2 + 1];

    matrix_offsets[0] = 0;
    for (int m = 0; m < M / M2; m++)
    {
        matrix_offsets[m + 1] = matrix_offsets[m] + offsets[m];
    }

    *h_matrix_vals = d_A2_vals;
    *h_matrix_cols = matrix_indeces;
    *h_matrix_offset = matrix_offsets;

    int total = 0;
    for (int m = 0; m < M / M2; m++)
    {
        total += offsets[m];
    }

    std::cout << "average nnzs: " << total / (M / M2) << std::endl;

    delete [] offsets;

    assert (index == size);

    int zeros = M * K - size;

    std::cout << "sparsity: " << (float) (zeros * 100.0f / (M * K)) << std::endl;

    return size;
}


__global__ void A2_spmm_batch8(
                        const int M,
                        const half* __restrict__ d_A2_vals,
                        const unsigned short* __restrict__ d_A2_Idx,
                        const int* __restrict__ d_row_ptr,
                        const half* __restrict__ B,
                        half* Reduction_Workspace,
                        int Split_K)
{
    constexpr int N = 8;

    const int warpID = threadIdx.x / M2;
    const int laneID = threadIdx.x % M2;
    const int BatchID = blockIdx.z;

    // get the amount of nnzs before the row this warp is responsible for
    const int offset1 = d_row_ptr[blockIdx.x];
    const int offset2 = d_row_ptr[blockIdx.x + 1];
    const int nnzs = offset2 - offset1;
    const int nnzs_per_warp = nnzs / (SPARSE_VEC * Split_K); // amount of non-zeros to work on per warp

    // move global pointers of A
    d_A2_vals += M2 * offset1;
    d_A2_Idx += M2 * offset1;

    // 128 bits pointers for vectorized load A
    const int4 *d_A2_vals_int4 = reinterpret_cast<const int4 *>(d_A2_vals);
    const int4 *d_A2_Idx_int4 = reinterpret_cast<const int4 *>(d_A2_Idx);

    // pointer for vectorized load B
    const VTYPE *B_vec = reinterpret_cast<const VTYPE *>(B);

    // accumulator
    half sums[N] = {0};
    VTYPE *sums_vec = reinterpret_cast<VTYPE *>(sums);

    // buffer for vectorized loading A
    int4 a8, d8;
    half *a8_ptr = reinterpret_cast<half *>(&a8);
    ushort *d8_ptr = reinterpret_cast<ushort *>(&d8);

    // buffer for vectorized loading B
    VTYPE bs;
    half *bs_ptr = reinterpret_cast<half *>(&bs);

    // registers
    half a1;
    ushort d1;

    // iterate along K
    for (int k = 0; k < nnzs_per_warp; k++)
    {

        // Load 8 values and 8 k indeces of the values from A each time
        //int index = (nnzs_per_warp * warpID + k) * M2 + laneID;
        uint index = (BatchID + k * Split_K) * M2 + laneID;
        a8 = d_A2_vals_int4[index];
        d8 = d_A2_Idx_int4[index];

        // process the 8 values from A
        for (int i = 0; i < SPARSE_VEC; i++)
        {
            a1 = a8_ptr[i];
            d1 = d8_ptr[i];

            // vectorized load values from B
            bs = B_vec[d1];

            // process the values from B
            for (int j = 0; j < N; j++)
            {
                sums[j] += a1 * bs_ptr[j];
            }
        }
    }

    reinterpret_cast<VTYPE*>(Reduction_Workspace + BatchID * M * N + (blockIdx.x * M2 + laneID) * N)[0] = sums_vec[0];
}


__global__ void Reduction(half *C, half *reduction_space, int M, int N, int K, int K_cut)
{
  int laneID = threadIdx.x % 32;

  float4 sums = {};
  half* sums_half = reinterpret_cast<half *>(&sums);

  for (int k = 0; k < K_cut; k++)
  {
    float4 buffer = reinterpret_cast<float4 *>(reduction_space + k * M * N + (blockIdx.x * 32 + laneID) * N + blockIdx.y * 8)[0];
    half* buffer_half = reinterpret_cast<half *>(&buffer);

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
      sums_half[i] += buffer_half[i];
    }
  }

  reinterpret_cast<float4 *>(C + (blockIdx.x * 32 + laneID) * N + blockIdx.y * 8)[0] = sums;
}


cudaError_t
FSP_Batch8(cudaStream_t stream,
            const int M,
            const int N,
            const int K,
            const int K_cut,
            const half* __restrict__ d_A2_vals,
            const unsigned short* __restrict__ d_A2_Idx,
            const int* __restrict__ d_row_ptr,
            const half* __restrict__ d_B,
            half* d_C,
            half* Reduction_Workspace)
{
    dim3 gridDim(M / M2, 1, K_cut);
    dim3 blockDim(M2, 1, 1);

    A2_spmm_batch8<<<gridDim, blockDim>>>(M, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, Reduction_Workspace, K_cut);

    // dim3 grid2(M / 32, N / 8, 1);
    // dim3 block2(32, 1, 1);

    // Reduction<<<grid2, block2>>>(d_C, Reduction_Workspace, M, N, K, K_cut);

    return cudaGetLastError();
}
