/***************************************************************************
 * Copyright 2023 The HeteroSparse Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
// C = A*B
// C: col major
// A: row major
// B: col major

#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"

int cuda_CheckError()
{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
}

// New features: Copy size is X * 64, X can be any multiple to 8
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    // Permutation needed for avoiding bank conflicts using ldmatrix
    int lane_id = threadIdx.x % 32;
    int col     = lane_id % 8;
    int row1    = lane_id / 8;
    int row2    = lane_id / 8 + 4;
    //    int store_column1 = col ^ row1;
    //    int store_column2 = col ^ row2;

    // debug from eddy: disable permutation
    int store_column1 = col;
    int store_column2 = col;

    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
    }
}

// CopyTileFromGlobalToShared_64_X<TilingConfig::TILE_N2, TilingConfig>(
//             smem_write_PTR + TILE_M * (TILE_K/2), BTileGlobalPTR, N_Global, GlobalCopy);

template<int NumOfColsToCopy, typename TilingConfig>  // NumOfColsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_64_X(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    constexpr int LOAD_PER_ITER = 1 + (NumOfColsToCopy - 1) / 32; // 16: 1, 32: 1, 64: 2 AND 8: 1

    // Permutation needed for avoiding bank conflicts using ldmatrix
    int lane_id = threadIdx.x % WARP_SIZE;
    int col     = lane_id % (NumOfColsToCopy / HALF_PER_128B); // 8: 0
    int row1    = lane_id / (NumOfColsToCopy / HALF_PER_128B); // 8: lane_id

    int scol;
    int srow;

    if (TilingConfig::N8) {
        scol = (lane_id % (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2)) * 2;
        srow = lane_id / (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2);
    }
    else {
        scol = lane_id % (HALF_PER_SHMEM_BANK / HALF_PER_128B); // 0~7
        srow = lane_id / (HALF_PER_SHMEM_BANK / HALF_PER_128B); // 0~3
    }

    constexpr int COPY_UNIT_ROWS = (LOAD_PER_ITER * WARP_SIZE * HALF_PER_128B / NumOfColsToCopy); // 8: 32

    constexpr int XOR_RANGE = (HALF_PER_128B / (HALF_PER_SHMEM_BANK / NumOfColsToCopy)); // 8: 1

    //
    int       warp_id            = threadIdx.x / WARP_SIZE;
    int       TotalNumOfCopyUnit = TILE_K / COPY_UNIT_ROWS; // Total num of rows / num of rows per copy unit for a warp (8: 2)
    const int MaxIteration = 
        (TotalNumOfCopyUnit - 1) / (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1; // 8: 1
    //
    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 8: warp_id (0, 1)
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_ROWS * TilingConfig::TILE_N; // Here for batch 8, we need 16
        #pragma unroll
        for (int j = 0; j < LOAD_PER_ITER; j++) {
            int store_column = scol ^ ((srow + j * WARP_SIZE / (HALF_PER_SHMEM_BANK / HALF_PER_128B)) % XOR_RANGE);
            cp_async<16>(SharedPTR_Unit + store_column * HALF_PER_128B + (srow + j * WARP_SIZE / (HALF_PER_SHMEM_BANK / HALF_PER_128B)) * HALF_PER_SHMEM_BANK,
            // cp_async<16>(SharedPTR_Unit + col * HALF_PER_128B + (row1 + j * WARP_SIZE / (NumOfColsToCopy / HALF_PER_128B)) * NumOfColsToCopy,
                        GlobalPTR_Unit + col * HALF_PER_128B + (row1 + j * WARP_SIZE / (NumOfColsToCopy / HALF_PER_128B)) * GlobalStride,
                        AsyncCopyPredictor);
        }
    }
}

template<int NumOfColsToCopy, typename TilingConfig>  // NumOfColsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyCTileFromSharedToGlobal(const half* SharedPTR,
                                                                half* __restrict__ GlobalPTR,
                                                                const int   GlobalStride)
{
    constexpr int STORE_PER_ITER = 1 + (NumOfColsToCopy - 1) / 32; // 16: 1, 32: 1, 64: 2 AND 8: 1

    int lane_id = threadIdx.x % WARP_SIZE;
    int col     = lane_id % (NumOfColsToCopy / HALF_PER_128B);
    int row1    = lane_id / (NumOfColsToCopy / HALF_PER_128B);

    int scol;
    int srow;

    if (TilingConfig::N8) {
        scol = (lane_id % (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2)) * 2;
        srow = lane_id / (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2);
    }
    else {
        scol = lane_id % (HALF_PER_SHMEM_BANK / HALF_PER_128B);
        srow = lane_id / (HALF_PER_SHMEM_BANK / HALF_PER_128B);
    }


    constexpr int COPY_UNIT_ROWS = (STORE_PER_ITER * WARP_SIZE * HALF_PER_128B / NumOfColsToCopy); // 8: 32

    constexpr int XOR_RANGE = (HALF_PER_128B / (HALF_PER_SHMEM_BANK / NumOfColsToCopy)); // 8: 1
    //
    int       warp_id            = threadIdx.x / WARP_SIZE;
    int       TotalNumOfCopyUnit = TILE_M / COPY_UNIT_ROWS; // Total num of rows / num of rows per copy unit for a warp (8: 8)
    const int MaxIteration = 
        (TotalNumOfCopyUnit - 1) / (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1; // 8: 2
    //
    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) {

        int  COPY_UNIT_I        = (i * (BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id); // 8: 0 ~ (warp_id * 2 - 1)
        half* __restrict__ GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_ROWS * GlobalStride;
        const half* SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_ROWS * TilingConfig::TILE_N; // Here for batch 8, we need 16
        #pragma unroll
        for (int j = 0; j < STORE_PER_ITER; j++) {
            int load_column = scol ^ ((srow + j * WARP_SIZE / (HALF_PER_SHMEM_BANK / HALF_PER_128B)) % XOR_RANGE);

            // for (int k = 0; k < HALF_PER_128B; k++) {
            //     GlobalPTR_Unit[(row1 + j * WARP_SIZE / (NumOfColsToCopy / HALF_PER_128B)) * GlobalStride +
            //                    col * HALF_PER_128B + k] +=
            //             SharedPTR_Unit[
            //                     (srow + j * WARP_SIZE / (HALF_PER_SHMEM_BANK / HALF_PER_128B)) * HALF_PER_SHMEM_BANK +
            //                     load_column * HALF_PER_128B + k];
            // }

            reinterpret_cast<int4*>(&GlobalPTR_Unit[(row1 + j * WARP_SIZE / (NumOfColsToCopy / HALF_PER_128B)) * GlobalStride + col * HALF_PER_128B])[0] =
               reinterpret_cast<const int4*>(&SharedPTR_Unit[(srow + j * WARP_SIZE / (HALF_PER_SHMEM_BANK / HALF_PER_128B)) * HALF_PER_SHMEM_BANK + load_column * HALF_PER_128B])[0];
        }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR + TILE_M * TILE_K, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, (k + 1) * MMA_K);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR + TILE_M * TILE_K, warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++)
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
        //// only used for pipeline analysis
        // #pragma unroll
        //  for (int i = 0; i < WARP_ROW_TENSORS; i++)
        //{
        //   int j=0;
        //   MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
        // }
        // #pragma unroll
        //  for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++)
        //{
        //   int i=0;
        //   if(!TilingConfig::N8)
        //     MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS]+4 , a_read[i], b_read[j]+2 );    // c+4; b+2
        // }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedSPCoreComputations(float c[][REG_PER_C_TENSOR_16_16],
                                                            uint32_t __restrict__ a[][2],
                                                            uint32_t __restrict__ b[][4],
                                                            int32_t __restrict__ meta_data[][BLOCK_K_TENSORS],
                                                            half* __restrict__ SharedMemoryPTR,
                                                            int warp_start_row,
                                                            int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    FragSPLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0);
    // __syncthreads();
    // B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
    //     b, SharedMemoryPTR + TILE_M * (TILE_K / 2), warp_start_col, 0);
    B_RowMajFragLoadFromSharedToRegisters<TilingConfig>(
        b, SharedMemoryPTR + TILE_M * (TILE_K / 2), warp_start_col, 0);
    // __syncthreads();
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[2]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[2] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            FragSPLoadFromSharedToRegisters<WARP_ROW_TENSORS>(
                a_write, SharedMemoryPTR, warp_start_row, (k + 1) * (MMA_K / 2));
            // B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
            //     b_write, SharedMemoryPTR + TILE_M * (TILE_K / 2), warp_start_col, (k + 1) * MMA_K);
            B_RowMajFragLoadFromSharedToRegisters<TilingConfig>(
                b_write, SharedMemoryPTR + TILE_M * (TILE_K / 2), warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++) {
            int* meta = &(meta_data[i][k]);
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );

                MMA_SP_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j], meta);
                //                if(threadIdx.x==0)
                //                    printf("%d val=%f\n", i, __half2float(*(half*)(&a_read[i][0])));
                if (!TilingConfig::N8)
                    MMA_SP_FP16_M16N8K16(
                        c_uint32_t[i + j * WARP_ROW_TENSORS] + 2, a_read[i], b_read[j] + 2, meta);  // c+4; b+2
            }
        }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void SegmentScan_CUDACoreComputations(const int * __restrict__ row_ptr,
                                                                const half* __restrict__ A2_val,
                                                                const uint16_t* __restrict__ A2_idx,
                                                                half* __restrict__ B_smem_data,
                                                                half* __restrict__ last_partial_sum,
                                                                uint16_t* __restrict__ thread_start_flag,
                                                                half *C_smem_ptr)
{
    const int nnz = row_ptr[1] - row_ptr[0];

    const int VECTOR_SIZE = FINE_VECTOR_SIZE;
    //here is one assumption: each thread processed 8 elements,
    // TODO: need to tweak this to make it more flexible for different sparsity ratio
    const int num_waves = (nnz - 1) / (TilingConfig::BLOCK_THREADS * VECTOR_SIZE) + 1;

//    num_waves = 1;
    const int num_cols = TilingConfig::TILE_N2;

    #pragma unroll
    for(int nw=0; nw < num_waves-1; nw++)
    {
        int nnz_idx = row_ptr[0] + nw * TilingConfig::BLOCK_THREADS * VECTOR_SIZE + threadIdx.x * VECTOR_SIZE;

//        const int4 vals = reinterpret_cast<const int4*>(&A2_val[nnz_idx])[0];
//        const int4 idxs = reinterpret_cast<const int4*>(&A2_idx[nnz_idx])[0];

        const TypeUtils<FINE_VECTOR_SIZE>::vector_type vals = reinterpret_cast<const TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&A2_val[nnz_idx])[0];
        const TypeUtils<FINE_VECTOR_SIZE>::vector_type idxs = reinterpret_cast<const TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&A2_idx[nnz_idx])[0];

        const half *half_val_ptr = reinterpret_cast<const half*>(&vals);
        const uint16_t *half_idx_ptr = reinterpret_cast<const uint16_t*>(&idxs);

        half c2_accum[TilingConfig::TILE_N2] = {0};

        bool last_element_flag = ((half_idx_ptr[FINE_VECTOR_SIZE - 1] >> 14) & 1);

        bool contain_start = false;
        bool contain_end = false;

        bool first_element_not_start = (((half_idx_ptr[0] >> 15) & 1) != 1);

        int first_row_idx = (half_idx_ptr[0] & 0x3fff) / TILE_K;

        #pragma unroll
        for(int i = 0; i < VECTOR_SIZE; i++)
        {
            half val = half_val_ptr[i];
            uint16_t idx = half_idx_ptr[i];
            bool start_flag = (idx >> 15) & 1;
            bool end_flag = (idx >> 14) & 1;

            contain_start = contain_start || start_flag;
            contain_end = contain_end || end_flag;

            idx = idx & 0x3fff;
            int col = idx % TILE_K;
            int row = idx / TILE_K;

            #pragma unroll
            for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
            {
//                int4 B_vals = reinterpret_cast<int4*>(&B_smem_data[j * 8 + col * TilingConfig::TILE_N2])[0];
                const TypeUtils<FINE_VECTOR_SIZE>::vector_type B_vals = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&B_smem_data[
                        j * VECTOR_SIZE + col * TilingConfig::TILE_N2])[0];
                const half* B_val_ptr = reinterpret_cast<const half*>(&B_vals);
                #pragma unroll
                for (int q = 0; q < VECTOR_SIZE; q++) {
                    c2_accum[j * VECTOR_SIZE + q] = __hadd(c2_accum[j * VECTOR_SIZE + q], __hmul(val, B_val_ptr[q]));
                }
            }

            if(end_flag)
            {
                #pragma unroll
                for(int j=0; j < num_cols / VECTOR_SIZE; j++)
                {
//                    int4 CFrag_vals = reinterpret_cast<int4*>(&C_smem_ptr[row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                    TypeUtils<FINE_VECTOR_SIZE>::vector_type CFrag_vals = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                            row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                    half* CFrag_val_ptr = reinterpret_cast<half*>(&CFrag_vals);

                    #pragma unroll
                    for (int q = 0; q < VECTOR_SIZE; q++) {
                        CFrag_val_ptr[q] += c2_accum[j * VECTOR_SIZE + q];
                    }

//                    reinterpret_cast<int4*>(&C_smem_ptr[row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                    reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                            row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                    #pragma unroll
                    for (int q = 0; q < VECTOR_SIZE; q++) {
                        c2_accum[j * VECTOR_SIZE + q] = 0;
                    }
                }
            }
        }

        #pragma unroll
        for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
        {
//            int4 c2_vec_val;
            TypeUtils<FINE_VECTOR_SIZE>::vector_type c2_vec_val;
            half* c2_vec_val_ptr = reinterpret_cast<half*>(&c2_vec_val);

            #pragma unroll
            for (int q = 0; q < VECTOR_SIZE; q++) {
                c2_vec_val_ptr[q] = c2_accum[j * VECTOR_SIZE + q];
            }

            reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&last_partial_sum[j * VECTOR_SIZE +
                                                                                           threadIdx.x *
                                                                                           TilingConfig::TILE_N2])[0] = c2_vec_val;
//            reinterpret_cast<int4*>(&last_partial_sum[j * VECTOR_SIZE + threadIdx.x * TilingConfig::TILE_N2])[0] = c2_vec_val;
        }

        thread_start_flag[threadIdx.x] = contain_start ? 1 : 0;

        __syncthreads();

        // has the end of one row && the first element is continue ones from previous thread
        // first element should not be first element of the whole matrix
        if(contain_end && first_element_not_start)
        {
            //propagation from previous thread
            for(int prev_t = 1; prev_t < 6; prev_t++)
            {
//                assert(threadIdx.x - prev_t >= 0);

                #pragma unroll
                for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
                {
//                    int4 CFrag_vals = reinterpret_cast<int4*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                    TypeUtils<FINE_VECTOR_SIZE>::vector_type CFrag_vals = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                            first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                    half* CFrag_val_ptr = reinterpret_cast<half*>(&CFrag_vals);

//                    int4 last_partial_vec = reinterpret_cast<int4*>(&last_partial_sum[j * VECTOR_SIZE + (threadIdx.x - prev_t) * TilingConfig::TILE_N2])[0];
                    TypeUtils<FINE_VECTOR_SIZE>::vector_type last_partial_vec = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&last_partial_sum[
                            j * VECTOR_SIZE + (threadIdx.x - prev_t) * TilingConfig::TILE_N2])[0];
                    half* last_partial_vec_ptr = reinterpret_cast<half*>(&last_partial_vec);

                    #pragma unroll
                    for (int q = 0; q < VECTOR_SIZE; q++) {
                        CFrag_val_ptr[q] += last_partial_vec_ptr[q];
                    }

//                    reinterpret_cast<int4*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                    reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                            first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                }

                if(thread_start_flag[threadIdx.x - prev_t] == 1)
                {
                    break;
                }
            }

        }
        __syncthreads();
    }

    // last wave
    int nnz_idx = row_ptr[0] + (num_waves-1) * TilingConfig::BLOCK_THREADS * VECTOR_SIZE + threadIdx.x * VECTOR_SIZE;
    bool contain_start = false;
    bool contain_end = false;
    bool first_element_not_start = false;
    int first_row_idx = -1;

//    __syncthreads();

    // loading A_val and A_idx into smem for the last wave
    bool global_copy = (nnz_idx < row_ptr[1]);

//    if(threadIdx.x == 0) {
//        printf("nnz_idx=%d %d %d\n", nnz_idx, row_ptr[1], VECTOR_SIZE);
//    }
    if(nnz_idx < row_ptr[1]) {
//        printf("nnz_idx=%d %d %d\n", nnz_idx, row_ptr[1], threadIdx.x);
//        const int *vals_ptr = reinterpret_cast<const int*>(&A2_val[nnz_idx]);
//        const int *idxs_ptr = reinterpret_cast<const int*>(&A2_idx[nnz_idx]);
        const TypeUtils<FINE_VECTOR_SIZE>::vector_type vals = reinterpret_cast<const TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&A2_val[nnz_idx])[0];
        const TypeUtils<FINE_VECTOR_SIZE>::vector_type idxs = reinterpret_cast<const TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&A2_idx[nnz_idx])[0];

//        const half *half_val_ptr = reinterpret_cast<const half*>(vals_ptr);
//        const uint16_t *half_idx_ptr = reinterpret_cast<const uint16_t*>(idxs_ptr);
        const half *half_val_ptr = reinterpret_cast<const half*>(&vals);
        const uint16_t *half_idx_ptr = reinterpret_cast<const uint16_t*>(&idxs);

//        uint16_t debug_idxs[VECTOR_SIZE];
//
//        debug_idxs[0] = half_idx_ptr[0];
//        debug_idxs[1] = half_idx_ptr[1];

        half c2_accum[TilingConfig::TILE_N2] = {0};

        bool last_element_flag = ((half_idx_ptr[VECTOR_SIZE-1] >> 14) & 1);

//        printf("last_element_flag=%d %d\n", half_idx_ptr[VECTOR_SIZE-1], threadIdx.x);

        first_element_not_start = (((half_idx_ptr[0] >> 15) & 1) != 1);

        first_row_idx = (half_idx_ptr[0] & 0x3fff) / TILE_K;
//        printf("first_row_idx=%d\n", first_row_idx);
//        #pragma unroll
        for(int i=0; i<VECTOR_SIZE; i++)
        {
            half val = half_val_ptr[i];
            uint16_t idx = half_idx_ptr[i];
            bool start_flag = (idx >> 15) & 1;
            bool end_flag = (idx >> 14) & 1;

            contain_start = contain_start || start_flag;
            contain_end = contain_end || end_flag;

            idx = idx & 0x3fff;
            int col = idx % TILE_K;
            int row = idx / TILE_K;

            #pragma unroll
            for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
            {
                const TypeUtils<FINE_VECTOR_SIZE>::vector_type B_vals = reinterpret_cast<const TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&B_smem_data[j * VECTOR_SIZE + col * TilingConfig::TILE_N2])[0];
//                int4 B_vals = reinterpret_cast<int4*>(&B_smem_data[j * VECTOR_SIZE + col * TilingConfig::TILE_N2])[0];
                const half* B_val_ptr = reinterpret_cast<const half*>(&B_vals);
                #pragma unroll
                for (int q = 0; q < VECTOR_SIZE; q++) {
                    c2_accum[j * VECTOR_SIZE + q] = __hadd(c2_accum[j * VECTOR_SIZE + q], __hmul(val, B_val_ptr[q]));
                }
            }

            if(end_flag)
            {
                #pragma unroll
                for(int j=0; j < num_cols / VECTOR_SIZE; j++)
                {
                    TypeUtils<FINE_VECTOR_SIZE>::vector_type CFrag_vals = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
//                    int4 CFrag_vals = reinterpret_cast<int4*>(&C_smem_ptr[row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                    half* CFrag_val_ptr = reinterpret_cast<half*>(&CFrag_vals);

                    #pragma unroll
                    for (int q = 0; q < VECTOR_SIZE; q++) {
                        CFrag_val_ptr[q] += c2_accum[j * VECTOR_SIZE + q];
                    }

//                    reinterpret_cast<int4*>(&C_smem_ptr[row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                    reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                            row * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) +
                            j * VECTOR_SIZE])[0] = CFrag_vals;
#pragma unroll
                    for (int q = 0; q < VECTOR_SIZE; q++) {
                        c2_accum[j * VECTOR_SIZE + q] = 0;
                    }
                }
            }
        }

        #pragma unroll
        for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
        {
//            int4 c2_vec_val;
            TypeUtils<FINE_VECTOR_SIZE>::vector_type c2_vec_val;
            half* c2_vec_val_ptr = reinterpret_cast<half*>(&c2_vec_val);

            #pragma unroll
            for (int q = 0; q < VECTOR_SIZE; q++) {
                c2_vec_val_ptr[q] = c2_accum[j * VECTOR_SIZE + q];
            }

            reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&last_partial_sum[j * VECTOR_SIZE + threadIdx.x * TilingConfig::TILE_N2])[0] = c2_vec_val;
//            reinterpret_cast<int4*>(&last_partial_sum[j * VECTOR_SIZE + threadIdx.x * TilingConfig::TILE_N2])[0] = c2_vec_val;
        }

        thread_start_flag[threadIdx.x] = contain_start? 1 : 0;

    }

    __syncthreads();

    // has the end of one row && the first element is continue ones from previous thread
    // first element should not be first element of the whole matrix
    if(contain_end && first_element_not_start)
    {
            //propagation from previous thread
        for(int prev_t = 1; prev_t < 5; prev_t++)
        {
            #pragma unroll
            for(int j = 0; j < num_cols / VECTOR_SIZE; j++)
            {
//                int4 CFrag_vals = reinterpret_cast<int4*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
//                int CFrag_vals = reinterpret_cast<int*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];
                TypeUtils<FINE_VECTOR_SIZE>::vector_type CFrag_vals = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                        first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0];

                half* CFrag_val_ptr = reinterpret_cast<half*>(&CFrag_vals);

//                int4 last_partial_vec = reinterpret_cast<int4*>(&last_partial_sum[j * VECTOR_SIZE + (threadIdx.x - prev_t) * TilingConfig::TILE_N2])[0];
//                int last_partial_vec = reinterpret_cast<int*>(&last_partial_sum[j * VECTOR_SIZE + (threadIdx.x - prev_t) * TilingConfig::TILE_N2])[0];
                TypeUtils<FINE_VECTOR_SIZE>::vector_type last_partial_vec = reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&last_partial_sum[
                        j * VECTOR_SIZE + (threadIdx.x - prev_t) * TilingConfig::TILE_N2])[0];
                half* last_partial_vec_ptr = reinterpret_cast<half*>(&last_partial_vec);

                #pragma unroll
                for (int q = 0; q < VECTOR_SIZE; q++) {
                    CFrag_val_ptr[q] += last_partial_vec_ptr[q];
                }

//                reinterpret_cast<int4*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
//                reinterpret_cast<int*>(&C_smem_ptr[first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
                reinterpret_cast<TypeUtils<FINE_VECTOR_SIZE>::vector_type *>(&C_smem_ptr[
                        first_row_idx * (TilingConfig::TILE_N2 + PADDING_SHARED_MEM_FOR_C) + j * VECTOR_SIZE])[0] = CFrag_vals;
            }

            if(thread_start_flag[threadIdx.x - prev_t] == 1)
            {
                break;
            }
        }
    }

    return;
}


template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(half *smem_CFrag,
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;

                // // For FP32 accumulators
                // if (r % 2 > 0)
                //     col_offset += 1;
                // //
                // if (r % 4 >= 2)
                //     row_offset += 8;
                // if (r >= 4)
                //     col_offset += 8;
                //
                // (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] += __float2half(
                //         c[RegSetID][r]);
                
                // For FP16 accumulators
                if (r % 2 > 0)
                    // col_offset += 1;
                    row_offset += 8;
                //
                if (r % 4 >= 2)
                    // row_offset += 8;
                    col_offset += 8;
                
                half* smem = smem_CFrag + (Tensor_i_offset) * TilingConfig::TILE_N; // 16 for batch size 8

                int element_no = (row_offset) * TilingConfig::TILE_N + (Tensor_j_offset + col_offset);

                int new_i = element_no / (HALF_PER_SHMEM_BANK);
                int new_j = element_no % (HALF_PER_SHMEM_BANK);

                uint32_t mask = (new_i % (HALF_PER_128B / (HALF_PER_SHMEM_BANK / TilingConfig::TILE_N2))) << 3;

                // Permutation needed for avoiding bank conflicts
                new_j = new_j ^ mask;

                float smem_val; // = reinterpret_cast<float*>(smem + new_i * HALF_PER_SHMEM_BANK + new_j)[0];
                half* smem_val_ptr = reinterpret_cast<half*>(&smem_val);
                half* reg_val_ptr = reinterpret_cast<half*>(&c[RegSetID][r]);
                
                smem_val_ptr[0] = reg_val_ptr[0];
                smem_val_ptr[1] = reg_val_ptr[1];
                reinterpret_cast<float*>(smem + new_i * HALF_PER_SHMEM_BANK + new_j)[0] = smem_val;
            }
        }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister_FSP(float (*smem_CFrag)[TILE_M + PADDING_SHARED_MEM_FOR_C],
                                 half __restrict__ C_val1[][8],
                                 half __restrict__ C_val2[][8])
{
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int thx = lane_id % 8;
    int thy = lane_id / 8;

#pragma unroll
    for (int i = 0; i < 8; i++) {
//                printf("thx=%d ==> %d %f\n", threadIdx.x, i, __half2float(C_val2[0][i * 2]));
        (*(smem_CFrag + thx))[warpId * 32 + i * 4 + thy ] += __half2float(C_val1[0][i]);
//        (*(smem_CFrag + thx))[warpId * 32 + i * 8 + thy * 2 + 1] += __half2float(C_val1[0][i * 2 + 1]);
        (*(smem_CFrag + thx))[warpId * 32 + i * 4 + thy  + (TILE_M / 2)] +=
            __half2float(C_val2[0][i]);
//        (*(smem_CFrag + thx))[warpId * 32 + i * 8 + thy * 2 + (TILE_M / 2) + 1] +=
//            __half2float(C_val2[0][i * 2 + 1]);
        if (!TilingConfig::N8) {
            (*(smem_CFrag + thx + 8))[warpId * 32 + i * 4 + thy] += __half2float(C_val1[1][i]);
//            (*(smem_CFrag + thx + 8))[warpId * 32 + i * 8 + thy * 2 + 1] += __half2float(C_val1[1][i * 2 + 1]);
            (*(smem_CFrag + thx + 8))[warpId * 32 + i * 4 + thy + (TILE_M / 2)] +=
                __half2float(C_val2[1][i]);
//            (*(smem_CFrag + thx + 8))[warpId * 32 + i * 8 + thy * 2 + (TILE_M / 2) + 1] +=
//                __half2float(C_val2[1][i * 2 + 1]);
        }
    }
}

#endif