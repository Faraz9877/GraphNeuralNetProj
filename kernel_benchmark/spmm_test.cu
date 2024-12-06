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

// #define TRIM_B
// #define BREAKDOWN
#define SPARSE_LOAD_A

//#define USE_CUBLAS
#define USE_FLASH_LLM
// #define USE_SPUTNIK
// #define USE_CUSPARSE
//#define USE_SPARTA
//#define USE_CUSPARSELT

#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <iomanip>
#include "FSP_Computation.h"
#include "FSP_Batch8.h"
#include <cstdlib>


#ifdef USE_FLASH_LLM
#include "SpMM_API.cuh"
#include "spmm_decompose.h"
#endif

// #ifdef USE_SPUTNIK
#include "./sputnik_utils.h"
#include "sputnik/sputnik.h"
// #endif

#ifdef USE_SPARTA
#include "sparTA.h"
#endif

#ifdef USE_CUSPARSELT
#include "CuSparseLt.h"
#endif
//

// ITERATION wrongly used in SPMM

int main(int argc, char** argv)
{
    if (argc != 7) {
        printf("Wrong Inputs! Correct input format: ./spmm_test M K N Sparsity A1_SplitK A2_SplitK\n");
        return;
    }
    int M_GLOBAL                    = atoi(argv[1]);
    int K_GLOBAL                    = atoi(argv[2]);
    int N_GLOBAL                    = atoi(argv[3]);
    int MATRIX_A_PRUNING_PERCENTAGE = atoi(argv[4]);
    int A1_SPLIT_K                     = atoi(argv[5]);
    int A2_SPLIT_K                     = atoi(argv[6]);

    if (N_GLOBAL > 8 && A2_SPLIT_K != 1) {
        printf("N_GLOBAL > 8, A2_SPLIT_K must be 1\n");
        return 0;
    }
    //
    // printf("M: %d N: %d K: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    //
    cublasStatus_t cublas_status;
    // cusparseStatus_t  cusparse_status;
    // cudaError_t       cuda_error;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    half* A_h            = NULL;  // row major
    half* B_h            = NULL;  // col major
    half* B_Transposed_h = NULL;  // row major
    // Device memory
    half* A            = NULL;
    half* B            = NULL;
    half* B_Transposed = NULL;
    //
    A_h            = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h            = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(half) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    //
    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    // init_host_structure_sparsity(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);
    // half *debug_A1, *debug_A2;
    // debug_A1 = (half *) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    // debug_A2 = (half *) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    // printf("decompose\n");
    // decompose(A_h, debug_A1, debug_A2, M_GLOBAL * K_GLOBAL);
    // //use A2 part to check
    // memcpy(A_h, debug_A1, sizeof(half) * M_GLOBAL * K_GLOBAL);
//    A_h[143*K_GLOBAL + 62] = 0.0;

//    int row_idx = 62;

//    for(int i = 0; i < N_GLOBAL; i++)
//    {
//        printf("%f ", __half2float(B_h[row_idx + i * K_GLOBAL]));
//    }

//    getchar();

//    InitUnStructureSparseMatrix(A2, M_GLOBAL, K_GLOBAL, NULL, NULL, NULL);
//    return 0;

//    init_host_structure_sparsity(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);

    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++) {
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
            // printf("%f ", __half2float(B_Transposed_h[i * N_GLOBAL + j]));
        }
    //
    // printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    //#ifdef USE_CUBLAS
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuBlas...\n");
    half* D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    // Tensor core not enabled
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cudaDeviceSynchronize();
    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas = 0;
    cudaEventElapsedTime(&milliseconds_cublas, start, stop);
    milliseconds_cublas = milliseconds_cublas / BENCHMARK_ITERATION;
    float tflops_cublas =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas / 1000.))
        / 1e12;
    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16F,
                                     k,
                                     B,
                                     CUDA_R_16F,
                                     k,
                                     &beta,
                                     D_cublas,
                                     CUDA_R_16F,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16F,
                     k,
                     B,
                     CUDA_R_16F,
                     k,
                     &beta,
                     D_cublas,
                     CUDA_R_16F,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    milliseconds_cublas_tc = milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                 / (milliseconds_cublas_tc / 1000.))
                             / 1e12;
    half* D_cublas_h = NULL;  // col major
    D_cublas_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_cublas);
    /////////////////////////////////////////////////////////////////////////////////////////////////
//#endif
#ifdef USE_FLASH_LLM
    /////////////////////////////////////////////////////////////////////////////////////////////////
    half *A1, *A2;
    A1 = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    A2 = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    printf("decompose\n");
    decompose(A_h, A1, A2, M_GLOBAL * K_GLOBAL);
    // memset(A1, 0, sizeof(half) * M_GLOBAL * K_GLOBAL);
    // memcpy(A2, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL);

    half* D_SpMM2 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_SpMM2), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM2 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM2, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    half* NZWeights_CPU   = NULL;
    int*      TileOffsets_CPU = NULL;
//    int NumOffsets = InitSparseMatrixA_API_NoReorder_v1(A_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, &NZWeights_CPU, &TileOffsets_CPU);
    uint64_t* Ind_data_CPU = NULL; 
    uint64_t *Ind_data_GPU = NULL;
#ifdef SPARSE_LOAD_A
    int NumOffsets = InitSparseMatrixA1_Vec_Sparse_Load(A1, M_GLOBAL, K_GLOBAL, &NZWeights_CPU, &Ind_data_CPU, &TileOffsets_CPU);
#else
    int NumOffsets = InitSparseMatrixA_API_NoReorder_v4(A1, M_GLOBAL, N_GLOBAL, K_GLOBAL, &NZWeights_CPU, &TileOffsets_CPU);
#endif
    int NNZ        = TileOffsets_CPU[NumOffsets - 1] * 8;  // VectorSize = 4

    // bangtian: add meta data for using sparse tensor core
    int *h_meta, *d_meta;
    h_meta = (int *)malloc((M_GLOBAL/16) * (K_GLOBAL/16) * 8* sizeof(int));
    cudaMalloc((void **)&d_meta, (M_GLOBAL/16) * (K_GLOBAL/16) * 8 * sizeof(int));
    inspect_metadata(A1, h_meta, M_GLOBAL, K_GLOBAL);
    cudaMemcpy(d_meta, h_meta, (M_GLOBAL/16) * (K_GLOBAL/16) * 8*sizeof(int), cudaMemcpyHostToDevice);

//

    // printf("NumOffsets: %d, NNZ: %d\n", NumOffsets, NNZ);
    //
    half *NZWeights_GPU   = NULL;
    int *TileOffsets_GPU = NULL;
    cudaMalloc(&TileOffsets_GPU, sizeof(int) * NumOffsets);
    if (NNZ == 0)
        NNZ = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&NZWeights_GPU, sizeof(uint32_t) * NNZ);
    cudaMalloc(reinterpret_cast<void**>(&NZWeights_GPU), sizeof(half) * NNZ);

#ifdef SPARSE_LOAD_A
    cudaMalloc(reinterpret_cast<void**>(&Ind_data_GPU), sizeof(uint64_t) * (NNZ/8));
    cudaMemcpy(Ind_data_GPU, Ind_data_CPU, sizeof(uint64_t) * (NNZ/8), cudaMemcpyHostToDevice);
    free(Ind_data_CPU);
#endif

    if (TileOffsets_GPU == NULL || NZWeights_GPU == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
//    cudaMemcpy(NZWeights_GPU, NZWeights_CPU, sizeof(uint32_t) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(NZWeights_GPU, NZWeights_CPU, sizeof(half) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_GPU, TileOffsets_CPU, sizeof(int) * NumOffsets, cudaMemcpyHostToDevice);
    
    free(TileOffsets_CPU);
    free(NZWeights_CPU);
    // printf("Done! Compressed A matrix for GPU kernel: MM_Sparse_TC.\n");
    //
    //Call Sputnik for A2 part
    float* A2_float_h = (float*)malloc(sizeof(float) * m * k);
    for (int i = 0; i < m * k; i++)
        A2_float_h[i] = __half2float(A2[i]);

    half *h_A2_vals, *d_A2_vals;
    uint16_t *h_A2_idx, *d_A2_idx;
    int *h_A2_row_ptr, *d_A2_row_ptr;
    int *h_row_length, *d_row_length;
    // below two are for sputnik-sf version
    uint8_t *h_A2_tile_length, *d_A2_tile_length;
    int *h_A2_tile_ptr, *d_A2_tile_ptr;
//     int size = A2_inspection_matrix(A2, M_GLOBAL, K_GLOBAL, &h_A2_vals, &h_A2_idx, &h_A2_row_ptr);
    int size;
    if (N_GLOBAL == 8)
        size = A2_inspection_matrix_batch8(A2, M_GLOBAL, K_GLOBAL, A2_SPLIT_K, &h_A2_vals, &h_A2_idx, &h_A2_row_ptr);
    else
        size = A2_inspection_matrix_sputnik_spadding(A2, M_GLOBAL, K_GLOBAL, &h_A2_vals, &h_A2_idx, &h_A2_row_ptr);
    
//    int size = A2_inspection_matrix_sputnik_spadding_sf(A2, M_GLOBAL, K_GLOBAL, &h_A2_vals, &h_A2_idx,
//                                                        &h_A2_tile_length, &h_A2_tile_ptr, &h_A2_row_ptr);


//    int size = A2_inspection_matrix_sputnik(A2, M_GLOBAL, K_GLOBAL, &h_A2_vals, &h_A2_idx, &h_A2_row_ptr, &h_row_length);
//    int size = A2_inspection_matrix_kcut_v1(A2, M_GLOBAL, K_GLOBAL, SPLIT_K, &h_A2_vals, &h_A2_idx, &h_A2_row_ptr);
//    return 0;
    cudaMalloc((void **)&d_A2_vals, sizeof(half) * size);
    cudaMalloc((void **)&d_A2_idx, sizeof(uint16_t) * size);
//    cudaMalloc((void **)&d_A2_row_ptr, sizeof(int) * ((M_GLOBAL/32)*SPLIT_K + 1));
//    cudaMalloc((void **)&d_A2_row_ptr, sizeof(int) * ((M_GLOBAL/32) + 1));
//    cudaMalloc((void **)&d_A2_row_ptr, sizeof(int) * ((M_GLOBAL/4) + 1));
    cudaMalloc((void **)&d_A2_row_ptr, sizeof(int) * ((M_GLOBAL) + 1));
//    cudaMalloc((void **)&d_row_length, sizeof(int) * M_GLOBAL);

#ifdef TRIM_B
    // Compress fine-grained column indices
    int row_per_block = WARP_SIZE / (N_GLOBAL / HALF_PER_128B);
    for (int i = 0; i < size; i++) {
        int prev_col = h_A2_idx[i];

        int base_idx = prev_col - (prev_col % (row_per_block * 2));
        int remainder_idx = (prev_col % (row_per_block * 2));

        h_A2_idx[i] = base_idx + (remainder_idx - (2 - (remainder_idx % 2))) / 2;
    }
#endif

    cudaMemcpy(d_A2_vals, h_A2_vals, sizeof(half) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2_idx, h_A2_idx, sizeof(uint16_t) * size, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_A2_row_ptr, h_A2_row_ptr, sizeof(int) * ((M_GLOBAL/32)*SPLIT_K + 1), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_A2_row_ptr, h_A2_row_ptr, sizeof(int) * ((M_GLOBAL/4) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2_row_ptr, h_A2_row_ptr, sizeof(int) * ((M_GLOBAL) + 1), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_row_length, h_row_length, sizeof(int) * M_GLOBAL, cudaMemcpyHostToDevice);


    int *row_indices = (int *)malloc(sizeof(int) * M_GLOBAL);
    for(int i=0; i < M_GLOBAL; i++)
    {
        row_indices[i] = i;
    }

    {
        // Create our unsorted row indices.
        std::vector<int> swizzle_staging(M_GLOBAL);
//        std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);
        for (int i=0; i < M_GLOBAL; i++)
            swizzle_staging[i] = i;
        // Argsort the row indices based on their length.
        std::sort(swizzle_staging.begin(), swizzle_staging.end(), [&h_A2_row_ptr](int idx_a, int idx_b) {
            int length_a = h_A2_row_ptr[idx_a + 1] - h_A2_row_ptr[idx_a];
            int length_b = h_A2_row_ptr[idx_b + 1] - h_A2_row_ptr[idx_b];
            return length_a > length_b;
        });

        // Copy the ordered row indices to the output.
        std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * M_GLOBAL);
    }

    int *d_row_indices;
    cudaMalloc((void **)&d_row_indices, sizeof(int) * M_GLOBAL);
    cudaMemcpy(d_row_indices, row_indices, sizeof(int) * M_GLOBAL, cudaMemcpyHostToDevice);


//    cudaMalloc((void **)&d_A2_tile_length, sizeof(uint8_t) * (M_GLOBAL/4));
//    cudaMemcpy(d_A2_tile_length, h_A2_tile_length, sizeof(uint8_t) * (M_GLOBAL/4), cudaMemcpyHostToDevice);
//    cudaMalloc((void **)&d_A2_tile_ptr, sizeof(int) * (M_GLOBAL/4));
//    cudaMemcpy(d_A2_tile_ptr, h_A2_tile_ptr, sizeof(int) * (M_GLOBAL/4), cudaMemcpyHostToDevice);

    checkLastCudaError(__LINE__);
//    return 0;
//
//    sputnik_utils::SparseMatrix            sparse_matrix_A2(m, k, A2_float_h, sputnik_utils::SORTED, 4);
//    sputnik_utils::CudaSparseMatrix<half2> sparse_matrix_A2_gpu(sparse_matrix_A2);

    printf("Launching HeteroSparse without Ahead of Time Sparse Data Reordering...\n");
    
    half *Reduction_Workspace = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace), sizeof(half) * M_GLOBAL * N_GLOBAL * (A1_SPLIT_K + A2_SPLIT_K));
    cudaMemset(Reduction_Workspace, 0, sizeof(half) * M_GLOBAL * N_GLOBAL * (A1_SPLIT_K + A2_SPLIT_K));
    if (Reduction_Workspace == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }

    // Reduction kernel configuration
    dim3 grid2(M_GLOBAL / 32, N_GLOBAL / 8, 1);
    dim3 block2(32, 1, 1);

    // Initialize separate streams for A1 and A2
    cudaStream_t stream_A1, stream_A2, stream_reduction;
    cudaStreamCreate(&stream_A1);
    cudaStreamCreate(&stream_A2);
    cudaStreamCreate(&stream_reduction);

    cudaProfilerStart();
    for (int i = 0; i < WARM_UP_ITERATION; i++){
        SpMM_SplitK_API(0,
                       A,
                       reinterpret_cast<uint4 *>(NZWeights_GPU),
                       Ind_data_GPU,
                       d_meta,
                       TileOffsets_GPU,
                       B_Transposed,
                       D_SpMM2,
                       M_GLOBAL,
                       N_GLOBAL,
                       K_GLOBAL,
                       Reduction_Workspace,
                       A1_SPLIT_K);
        if (N_GLOBAL == 8)
            CUDA_CALL(FSP_Batch8(0,
                                M_GLOBAL,
                                N_GLOBAL,
                                K_GLOBAL,
                                A2_SPLIT_K,
                                d_A2_vals,
                                d_A2_idx,
                                d_A2_row_ptr,
                                // d_row_indices,
                                B_Transposed,
                                D_SpMM2,
                                Reduction_Workspace + M_GLOBAL * N_GLOBAL * A1_SPLIT_K
                                ));
        else
            CUDA_CALL(FSP_Computation_reorder(M_GLOBAL,
                                            N_GLOBAL,
                                            K_GLOBAL,
                                            d_A2_vals,
                                            d_A2_idx,
                                            d_A2_row_ptr,
                                            d_row_indices,
                                            B_Transposed,
                                            Reduction_Workspace + M_GLOBAL * N_GLOBAL * A1_SPLIT_K));
        // cudaDeviceSynchronize();
        Reduction<<<grid2, block2, 0, 0>>>(D_SpMM2, Reduction_Workspace, M_GLOBAL, N_GLOBAL, K_GLOBAL, (A1_SPLIT_K + A2_SPLIT_K));
        // cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        SpMM_SplitK_API(0,
                       A,
                       reinterpret_cast<uint4 *>(NZWeights_GPU),
                       Ind_data_GPU,
                       d_meta,
                       TileOffsets_GPU,
                       B_Transposed,
                       D_SpMM2,
                       M_GLOBAL,
                       N_GLOBAL,
                       K_GLOBAL,
                       Reduction_Workspace,
                       A1_SPLIT_K);
        if (N_GLOBAL == 8)
            CUDA_CALL(FSP_Batch8(0,
                                M_GLOBAL,
                                N_GLOBAL,
                                K_GLOBAL,
                                A2_SPLIT_K,
                                d_A2_vals,
                                d_A2_idx,
                                d_A2_row_ptr,
                                // d_row_indices,
                                B_Transposed,
                                D_SpMM2,
                                Reduction_Workspace + M_GLOBAL * N_GLOBAL * A1_SPLIT_K
                                ));
        else
            CUDA_CALL(FSP_Computation_reorder(M_GLOBAL,
                                            N_GLOBAL,
                                            K_GLOBAL,
                                            d_A2_vals,
                                            d_A2_idx,
                                            d_A2_row_ptr,
                                            d_row_indices,
                                            B_Transposed,
                                            Reduction_Workspace + M_GLOBAL * N_GLOBAL * A1_SPLIT_K));
        // cudaDeviceSynchronize();
        Reduction<<<grid2, block2, 0, 0>>>(D_SpMM2, Reduction_Workspace, M_GLOBAL, N_GLOBAL, K_GLOBAL, (A1_SPLIT_K + A2_SPLIT_K));
        // cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    checkLastCudaError(__LINE__);
    cudaProfilerStop();
    // Get the average of the times
    float milliseconds_SpMM2 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM2, start, stop);
    milliseconds_SpMM2 = milliseconds_SpMM2 / BENCHMARK_ITERATION;
    float tflops_SpMM2 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM2 / 1000.))
        / 1e12;
    half* D_SpMM_h2 = NULL;  // col major
    D_SpMM_h2       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_h2, D_SpMM2, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major

    half* D_SpMM_h2_col_maj = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    for(int i = 0; i < M_GLOBAL; i++)
    {
        for(int j = 0; j < N_GLOBAL; j++)
        {
            D_SpMM_h2_col_maj[i + j * M_GLOBAL] = D_SpMM_h2[j + i * N_GLOBAL];
        }
    }


#ifdef BREAKDOWN
    // Breakdown Times
    for (int i = 0; i < WARM_UP_ITERATION; i++){
        SpMM_SplitK_API(0,
                       A,
                       reinterpret_cast<uint4 *>(NZWeights_GPU),
                    //    Ind_data_GPU,
                       d_meta,
                       TileOffsets_GPU,
                       B_Transposed,
                       D_SpMM2,
                       M_GLOBAL,
                       N_GLOBAL,
                       K_GLOBAL,
                       A1_Reduction_Workspace,
                       A1_SPLIT_K);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        SpMM_SplitK_API(0,
                       A,
                       reinterpret_cast<uint4 *>(NZWeights_GPU),
                    //    Ind_data_GPU,
                       d_meta,
                       TileOffsets_GPU,
                       B_Transposed,
                       D_SpMM2,
                       M_GLOBAL,
                       N_GLOBAL,
                       K_GLOBAL,
                       A1_Reduction_Workspace,
                       A1_SPLIT_K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the average of the times
    float milliseconds_SpMM2_A1 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM2_A1, start, stop);
    milliseconds_SpMM2_A1 = milliseconds_SpMM2_A1 / BENCHMARK_ITERATION;
    float tflops_SpMM2_A1 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM2_A1 / 1000.))
        / 1e12;

    for (int i = 0; i < WARM_UP_ITERATION; i++){
        CUDA_CALL(FSP_Batch8(M_GLOBAL,
                            N_GLOBAL,
                            K_GLOBAL,
                            A2_SPLIT_K,
                            d_A2_vals,
                            d_A2_idx,
                            d_A2_row_ptr,
                            // d_row_indices,
                            B_Transposed,
                            D_SpMM2,
                            A2_Reduction_Workspace
                            ));
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        CUDA_CALL(FSP_Batch8(M_GLOBAL,
                            N_GLOBAL,
                            K_GLOBAL,
                            A2_SPLIT_K,
                            d_A2_vals,
                            d_A2_idx,
                            d_A2_row_ptr,
                            // d_row_indices,
                            B_Transposed,
                            D_SpMM2,
                            A2_Reduction_Workspace
                            ));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the average of the times
    float milliseconds_SpMM2_A2 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM2_A2, start, stop);
    milliseconds_SpMM2_A2 = milliseconds_SpMM2_A2 / BENCHMARK_ITERATION;
    float tflops_SpMM2_A2 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM2_A2 / 1000.))
        / 1e12;
#endif

    // std::cout.unsetf(std::ios::floatfield); std::cout << std::setprecision(4);
    // int stride = 2;
    // std::cout << "D_SpMM_h2:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < std::min(64, N_GLOBAL) / stride; j++) {
    //         std::cout << __half2float(D_SpMM_h2_col_maj[i + j * stride * M_GLOBAL]) << "\t";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
    // std::cout << "D_cublas_h:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < std::min(64, N_GLOBAL) / stride; j++) {
    //         std::cout << __half2float(D_cublas_h[i + j * stride * M_GLOBAL]) << "\t";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
    // std::cout.unsetf(std::ios::floatfield);

    // for (int i = 0; i < M_GLOBAL; i++) {
    //     for (int j = 0; j < N_GLOBAL; j += stride) {
    //         if (isnan(__half2float(D_SpMM_h2_col_maj[i + j * M_GLOBAL])) || fabs(__half2float(D_SpMM_h2_col_maj[i + j * M_GLOBAL]) - __half2float(D_cublas_h[i + j * M_GLOBAL])) > 0.9) {
    //             printf("i = %d, j = %d, D_SpMM_h2 = %f, D_cublas_h = %f\n", i, j, __half2float(D_SpMM_h2_col_maj[i + j * M_GLOBAL]), __half2float(D_cublas_h[i + j * M_GLOBAL]));
    //             getchar();
    //         }
    //     }
    // }

    free(D_SpMM_h2);
    cudaFree(D_SpMM2);
    cudaFree(NZWeights_GPU);
    cudaFree(TileOffsets_GPU);
    cudaFree(Reduction_Workspace);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif
#ifdef USE_CUSPARSE
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuSparse_ColMajor...\n");
    half* D_CuSparse = NULL;
    cudaMalloc(&D_CuSparse, N_GLOBAL * M_GLOBAL * sizeof(half));
    if (D_CuSparse == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_CuSparse, 0.0f, N_GLOBAL * M_GLOBAL * sizeof(half));
    //
    cusparseHandle_t sp_handle = 0;
    cusparseCreate(&sp_handle);
    cusparseSetStream(sp_handle, 0);
    cusparseSpMatDescr_t SpMatA;
    cusparseDnMatDescr_t DnMatA, DnMatB, DnMatC;
    // Create Dense Matrix
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatA,
                                       M_GLOBAL,
                                       K_GLOBAL,
                                       K_GLOBAL,
                                       A,
                                       CUDA_R_16F,
                                       CUSPARSE_ORDER_ROW))  // Very critical!!! Weight Matrix must be row major,
                                                             // otherwise causing significant performance problems
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatB, K_GLOBAL, N_GLOBAL, K_GLOBAL, B, CUDA_R_16F, CUSPARSE_ORDER_COL))
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatC, M_GLOBAL, N_GLOBAL, M_GLOBAL, D_CuSparse, CUDA_R_16F, CUSPARSE_ORDER_COL))
    // Create Sparse Matrix in CSR format
    int* csrRowPtr;
    cudaMalloc(&csrRowPtr, sizeof(int) * (M_GLOBAL + 1));
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMatA,
                                     M_GLOBAL,
                                     K_GLOBAL,
                                     0,
                                     csrRowPtr,
                                     NULL,
                                     NULL,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_16F))
    // execute Sparse to Dense conversion
    void*  Buffer     = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(
        cusparseDenseToSparse_bufferSize(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))
    cudaMalloc(&Buffer, bufferSize);
    CHECK_CUSPARSE(
        cusparseDenseToSparse_analysis(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer))
    //
    int64_t numRowTMP, numColTMP, NNZ_1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(SpMatA, &numRowTMP, &numColTMP, &NNZ_1))
    //
    int*  csrColInd;
    half* csrVal;
    cudaMalloc(&csrColInd, NNZ_1 * sizeof(int));
    cudaMalloc(&csrVal, NNZ_1 * sizeof(half));
    //
    CHECK_CUSPARSE(cusparseCsrSetPointers(SpMatA, csrRowPtr, csrColInd, csrVal))
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer))
    //
    cusparseSpMMAlg_t CuSparse_Algorithm;
    CuSparse_Algorithm = CUSPARSE_SPMM_ALG_DEFAULT;
    CuSparse_Algorithm =
        CUSPARSE_SPMM_CSR_ALG1;  // csrmm_kernel faster: Provide the best performance with column-major layout
    // CuSparse_Algorithm = CUSPARSE_SPMM_CSR_ALG2;    // csrmm_v2_kernel: Provide the best performance with row-major
    // layout!!! How about try row major of B&C? CuSparse_Algorithm = CUSPARSE_SPMM_CSR_ALG3;
    //
    // printf("CuSparse Algorithm: %d \n", CuSparse_Algorithm);
    //
    const float alpha_float = 1.0;
    const float beta_float  = 0.0;
    //
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_float,
                                           SpMatA,
                                           DnMatB,
                                           &beta_float,
                                           DnMatC,
                                           CUDA_R_32F,
                                           CuSparse_Algorithm,
                                           &bufferSize))
    cudaFree(Buffer);
    cudaMalloc(&Buffer, bufferSize);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA,
                                    DnMatB,
                                    &beta_float,
                                    DnMatC,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm,
                                    Buffer))
    cudaEventRecord(start);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA,
                                    DnMatB,
                                    &beta_float,
                                    DnMatC,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm,
                                    Buffer))
    cudaEventRecord(stop);
    //
    float milliseconds_CuSparse_ColMajor = 0.0f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_CuSparse_ColMajor, start, stop);
    milliseconds_CuSparse_ColMajor = milliseconds_CuSparse_ColMajor / CUSPARSE_ITERATION;
    float tflops_CuSparse_ColMajor = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                         / (milliseconds_CuSparse_ColMajor / 1000.))
                                     / 1e12;
    //
    half* D_CuSparse_h;
    D_CuSparse_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_CuSparse_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_CuSparse_h, D_CuSparse, N_GLOBAL * M_GLOBAL * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(D_CuSparse);
    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);
    cudaFree(Buffer);
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuSparse_RowMajor...\n");
    half* D_CuSparse_2;
    cudaMalloc(&D_CuSparse_2, N_GLOBAL * M_GLOBAL * sizeof(half));
    if (D_CuSparse_2 == NULL) {
        printf("Error in Test_SpMM_v2.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_CuSparse_2, 0.0f, N_GLOBAL * M_GLOBAL * sizeof(half));
    //
    cusparseHandle_t sp_handle_2 = 0;
    cusparseCreate(&sp_handle_2);
    cusparseSpMatDescr_t SpMatA_2;
    cusparseDnMatDescr_t DnMatA_2, DnMatB_2, DnMatC_2;
    // Create Dense Matrix
    CHECK_CUSPARSE(cusparseCreateDnMat(&DnMatA_2,
                                       M_GLOBAL,
                                       K_GLOBAL,
                                       K_GLOBAL,
                                       A,
                                       CUDA_R_16F,
                                       CUSPARSE_ORDER_ROW))  // Very critical!!! Weight Matrix must be row major,
                                                             // otherwise causing significant performance problems

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatB_2, K_GLOBAL, N_GLOBAL, N_GLOBAL, B_Transposed, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&DnMatC_2, M_GLOBAL, N_GLOBAL, N_GLOBAL, D_CuSparse_2, CUDA_R_16F, CUSPARSE_ORDER_ROW))
    // Create Sparse Matrix in CSR format
    int* csrRowPtr_2;
    cudaMalloc(&csrRowPtr_2, sizeof(int) * (M_GLOBAL + 1));
    CHECK_CUSPARSE(cusparseCreateCsr(&SpMatA_2,
                                     M_GLOBAL,
                                     K_GLOBAL,
                                     0,
                                     csrRowPtr_2,
                                     NULL,
                                     NULL,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_16F))
    // execute Sparse to Dense conversion
    void*  Buffer_2     = NULL;
    size_t bufferSize_2 = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
        sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize_2))
    cudaMalloc(&Buffer_2, bufferSize_2);
    CHECK_CUSPARSE(
        cusparseDenseToSparse_analysis(sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer_2))
    //
    int64_t numRowTMP_2, numColTMP_2, NNZ_2;
    CHECK_CUSPARSE(cusparseSpMatGetSize(SpMatA_2, &numRowTMP_2, &numColTMP_2, &NNZ_2))
    //
    int*  csrColInd_2;
    half* csrVal_2;
    cudaMalloc(&csrColInd_2, NNZ_2 * sizeof(int));
    cudaMalloc(&csrVal_2, NNZ_2 * sizeof(half));
    //
    CHECK_CUSPARSE(cusparseCsrSetPointers(SpMatA_2, csrRowPtr_2, csrColInd_2, csrVal_2))
    CHECK_CUSPARSE(
        cusparseDenseToSparse_convert(sp_handle_2, DnMatA_2, SpMatA_2, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer_2))
    //
    cusparseSpMMAlg_t CuSparse_Algorithm_2;
    CuSparse_Algorithm_2 = CUSPARSE_SPMM_ALG_DEFAULT;
    CuSparse_Algorithm_2 =
        CUSPARSE_SPMM_CSR_ALG1;  // csrmm_kernel faster: Provide the best performance with column-major layout
    CuSparse_Algorithm_2 = CUSPARSE_SPMM_CSR_ALG2;  // csrmm_v2_kernel: Provide the best performance with row-major
                                                    // layout!!! How about try row major of B&C?
    // CuSparse_Algorithm_2 = CUSPARSE_SPMM_CSR_ALG3;
    // printf("CuSparse Algorithm: %d \n", CuSparse_Algorithm_2);
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(sp_handle_2,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_float,
                                           SpMatA_2,
                                           DnMatB_2,
                                           &beta_float,
                                           DnMatC_2,
                                           CUDA_R_32F,
                                           CuSparse_Algorithm_2,
                                           &bufferSize_2))
    cudaFree(Buffer_2);
    cudaMalloc(&Buffer_2, bufferSize_2);
    //
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle_2,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA_2,
                                    DnMatB_2,
                                    &beta_float,
                                    DnMatC_2,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm_2,
                                    Buffer_2))
    cudaEventRecord(start);
    for (int i = 0; i < CUSPARSE_ITERATION; i++)
        CHECK_CUSPARSE(cusparseSpMM(sp_handle_2,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_float,
                                    SpMatA_2,
                                    DnMatB_2,
                                    &beta_float,
                                    DnMatC_2,
                                    CUDA_R_32F,
                                    CuSparse_Algorithm_2,
                                    Buffer_2))
    cudaEventRecord(stop);
    //
    float milliseconds_CuSparse_RowMajor;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_CuSparse_RowMajor, start, stop);
    milliseconds_CuSparse_RowMajor = milliseconds_CuSparse_RowMajor / CUSPARSE_ITERATION;
    float tflops_CuSparse_RowMajor = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                         / (milliseconds_CuSparse_RowMajor / 1000.))
                                     / 1e12;
    // transpose result to col-major
    half* D_CuSparse_h_2_row_major;
    half* D_CuSparse_h_2;
    D_CuSparse_h_2           = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    D_CuSparse_h_2_row_major = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_CuSparse_h_2 == NULL || D_CuSparse_h_2_row_major == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_CuSparse_h_2_row_major,
               D_CuSparse_2,
               N_GLOBAL * M_GLOBAL * sizeof(half),
               cudaMemcpyDeviceToHost);  // row major
    for (int i = 0; i < N_GLOBAL; i++)
        for (int j = 0; j < M_GLOBAL; j++)
            D_CuSparse_h_2[i * M_GLOBAL + j] = D_CuSparse_h_2_row_major[i + j * N_GLOBAL];
    free(D_CuSparse_h_2_row_major);
    cudaFree(D_CuSparse_2);
    cudaFree(csrRowPtr_2);
    cudaFree(csrColInd_2);
    cudaFree(csrVal_2);
    cudaFree(Buffer_2);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif
#ifdef USE_SPUTNIK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching Sputnik...\n");
    half* D_Sputnik = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_Sputnik), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_Sputnik == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_Sputnik, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    //
    float* A_float_h = NULL;
    A_float_h        = (float*)malloc(sizeof(float) * M_GLOBAL * K_GLOBAL);
    for (int i = 0; i < M_GLOBAL * K_GLOBAL; i++)
        A_float_h[i] = __half2float(A_h[i]);
    sputnik_utils::SparseMatrix            sparse_matrix(M_GLOBAL, K_GLOBAL, A_float_h, sputnik_utils::IDENTITY, 4);
    sputnik_utils::CudaSparseMatrix<half2> sparse_matrix_gpu(sparse_matrix);
    for (int i = 0; i < WARM_UP_ITERATION; i++)
        CUDA_CALL(sputnik::CudaSpmm(M_GLOBAL,
                                    K_GLOBAL,
                                    N_GLOBAL,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_Transposed),
                                    reinterpret_cast<half2*>(D_Sputnik),
                                    0));
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        CUDA_CALL(sputnik::CudaSpmm(M_GLOBAL,
                                    K_GLOBAL,
                                    N_GLOBAL,
                                    sparse_matrix_gpu.NumElementsWithPadding(),
                                    sparse_matrix_gpu.RowIndices(),
                                    sparse_matrix_gpu.Values(),
                                    sparse_matrix_gpu.RowOffsets(),
                                    sparse_matrix_gpu.ColumnIndices(),
                                    reinterpret_cast<half2*>(B_Transposed),
                                    reinterpret_cast<half2*>(D_Sputnik),
                                    0));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_Sputnik = 0;
    cudaEventElapsedTime(&milliseconds_Sputnik, start, stop);
    milliseconds_Sputnik = milliseconds_Sputnik / BENCHMARK_ITERATION;
    float tflops_Sputnik =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_Sputnik / 1000.))
        / 1e12;
    // transpose result to col-major
    half* D_Sputnik_h_row_major = NULL;
    half* D_Sputnik_h           = NULL;
    D_Sputnik_h_row_major       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    D_Sputnik_h                 = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(
        D_Sputnik_h_row_major, D_Sputnik, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(D_Sputnik);
    for (int i = 0; i < N_GLOBAL; i++)
        for (int j = 0; j < M_GLOBAL; j++)
            D_Sputnik_h[i * M_GLOBAL + j] = D_Sputnik_h_row_major[i + j * N_GLOBAL];
    free(D_Sputnik_h_row_major);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif
#ifdef USE_SPARTA
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching sparTA...\n");
    //
    float milliseconds_sparTA  = 0;
    half* D_sparTA_h_row_major = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_sparTA_h_row_major == NULL) {
        printf("Error in spmm_test.cu: line %d Malloc falied\n", __LINE__);
        exit(-1);
    }
    sparTA(A_h, B_Transposed_h, D_sparTA_h_row_major, M_GLOBAL, N_GLOBAL, K_GLOBAL, &milliseconds_sparTA);
    float tflops_sparTA =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_sparTA / 1000.))
        / 1e12;
    // transpose result to col-major
    half* D_sparTA_h = NULL;
    D_sparTA_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    for (int i = 0; i < N_GLOBAL; i++)
        for (int j = 0; j < M_GLOBAL; j++)
            D_sparTA_h[i * M_GLOBAL + j] = D_sparTA_h_row_major[i + j * N_GLOBAL];
    free(D_sparTA_h_row_major);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif
#ifdef USE_CUSPARSELT
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuSparseLt...\n");
    //
    float milliseconds_cusparselt  = 0;
    half* D_cusparselt_h_row_major = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cusparselt_h_row_major == NULL) {
        printf("Error in spmm_test.cu: line %d Malloc falied\n", __LINE__);
        exit(-1);
    }
    CuSparseLt_matmul(A_h, B_Transposed_h, D_cusparselt_h_row_major, M_GLOBAL, N_GLOBAL, K_GLOBAL, &milliseconds_cusparselt);
    float tflops_cusparselt =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cusparselt / 1000.))
        / 1e12;
    // transpose result to col-major
    half* D_cusparselt_h = NULL;
    D_cusparselt_h       = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    for (int i = 0; i < N_GLOBAL; i++)
        for (int j = 0; j < M_GLOBAL; j++)
            D_cusparselt_h[i * M_GLOBAL + j] = D_cusparselt_h_row_major[i + j * N_GLOBAL];
    free(D_cusparselt_h_row_major);
    /////////////////////////////////////////////////////////////////////////////////////////////////
#endif
    // printf("Verifying correctness of the computations...\n");
    //
#ifdef USE_FLASH_LLM
//    double totalError_SpMM  = ComputeTotalError(D_cublas_h, D_SpMM_h, M_GLOBAL, N_GLOBAL);
    double totalError_SpMM2 = ComputeTotalError(D_cublas_h, D_SpMM_h2_col_maj, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("MySpMM", 10, 0.5, D_cublas_h, D_SpMM_h, M_GLOBAL, N_GLOBAL);
    free(D_SpMM_h2_col_maj);
#endif
#ifdef USE_CUSPARSE
    double totalError_CuSparse   = Comput eTotalError(D_cublas_h, D_CuSparse_h, M_GLOBAL, N_GLOBAL);
    double totalError_CuSparse_2 = ComputeTotalError(D_cublas_h, D_CuSparse_h_2, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("CuSparse",   10, 0.5, D_cublas_h, D_CuSparse_h, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("CuSparse2",  10, 0.5, D_cublas_h, D_CuSparse_h_2, M_GLOBAL, N_GLOBAL);
    free(D_CuSparse_h);
    free(D_CuSparse_h_2);
#endif
#ifdef USE_SPUTNIK
    double totalError_Sputnik = ComputeTotalError(D_cublas_h, D_Sputnik_h, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("Sputnik", 10, 0.5, D_cublas_h, D_Sputnik_h, M_GLOBAL, N_GLOBAL);
    free(D_Sputnik_h);
#endif
#ifdef USE_SPARTA
    double totalError_sparTA = ComputeTotalError(D_cublas_h, D_sparTA_h, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("sparTA", 10, 0.5, D_cublas_h, D_sparTA_h, M_GLOBAL, N_GLOBAL);
    free(D_sparTA_h);
#endif
#ifdef USE_CUSPARSELT
    double totalError_cusparselt = ComputeTotalError(D_cublas_h, D_cusparselt_h, M_GLOBAL, N_GLOBAL);
    // PrintMismatch("CuSparseLt", 10, 0.5, D_cublas_h, D_cusparselt_h, M_GLOBAL, N_GLOBAL);
    free(D_cusparselt_h);
#endif
    printf("******************************************Problem Size******************************************\n");
    printf("M: %d N: %d K: %d Pruning Rate: %d A1_SplitK: %d A2_SplitK: %d\n",
           M_GLOBAL,
           N_GLOBAL,
           K_GLOBAL,
           MATRIX_A_PRUNING_PERCENTAGE,
           A1_SPLIT_K,
           A2_SPLIT_K);
// printf("******************************************Performance*******************************************\n");
#ifdef USE_CUSPARSE
    PrintPerformance("CuSparse_C", milliseconds_CuSparse_ColMajor, tflops_CuSparse_ColMajor, totalError_CuSparse);
    PrintPerformance("CuSparse_R", milliseconds_CuSparse_RowMajor, tflops_CuSparse_RowMajor, totalError_CuSparse_2);
#endif
#ifdef USE_SPUTNIK
    PrintPerformance("Sputnik", milliseconds_Sputnik, tflops_Sputnik, totalError_Sputnik);
#endif
#ifdef USE_CUBLAS
    PrintPerformance("CuBlas_SIMT", milliseconds_cublas, tflops_cublas, 0.0);
    PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);
#endif
#ifdef USE_SPARTA
    PrintPerformance("sparTA", milliseconds_sparTA, tflops_sparTA, totalError_sparTA);
#endif
#ifdef USE_CUSPARSELT
    PrintPerformance("CuSparseLt", milliseconds_cusparselt, tflops_cusparselt, totalError_cusparselt);
#endif
#ifdef USE_FLASH_LLM
    PrintPerformance("HeteroSparse_v1", milliseconds_SpMM2, tflops_SpMM2, totalError_SpMM2);
#ifdef BREAKDOWN
    PrintPerformance("HeteroSparse_v1_A1", milliseconds_SpMM2_A1, tflops_SpMM2_A1, NAN);
    PrintPerformance("HeteroSparse_v1_A2", milliseconds_SpMM2_A2, tflops_SpMM2_A2, NAN);
//    PrintPerformance("HeteroSparse_v2", milliseconds_SpMM, tflops_SpMM, totalError_SpMM);
#endif
#endif

    free(D_cublas_h);
    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
#ifdef USE_FLASH_LLM
    free(h_meta);
    cudaFree(d_meta);
//    cudaFree(A2_vals_gpu);
//    cudaFree(A2_idx_gpu);
//    free(A1);
//    free(A2);
#endif
    return 0;
}
