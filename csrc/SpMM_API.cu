/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
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

#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./SpMM_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

template<typename TilingConfig, typename SparseKernelConfig>
static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint4* Compressed_A,
                                  const uint64_t*   A1_Ind_data,
                                  const int*   metadata,
                                  const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    // extra TILE_M * (TILE_K / 2) is used for fine-grained sparsity implementation, may will be replaced by a better implementation
//    static int SHMEM_SZ =
//        max((TILE_M * (TILE_K / 2) + TilingConfig::TILE_N * TILE_K + TILE_M * (TILE_K / 2))
//                * sizeof(half) * 2,
//            (TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    static int SHMEM_SZ = max((TILE_M * (TILE_K / 2) + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2 +
                                WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8 * sizeof(int32_t) * 2, // For metadata
                              (TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(half));
//    static int SHMEM_SZ = (TILE_M * (TILE_K / 2) + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2 +
//                          TILE_M * (TilingConfig::TILE_N + PADDING_SHARED_MEM_FOR_C) * sizeof(half) +
//                          TilingConfig::BLOCK_THREADS * TilingConfig::TILE_N * sizeof(half) + // res_cache and last_partial_res for threads
//                          TilingConfig::BLOCK_THREADS * sizeof(uint16_t); // includes_start flag for threads
    cudaFuncSetAttribute(
        SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(A,
                                                                                           Compressed_A,
                                                                                           TileOffsets,
                                                                                           A1_Ind_data,
                                                                                           metadata,
                                                                                           B,
                                                                                           Reduction_Workspace,
                                                                                           M_Global,
                                                                                           N_Global,
                                                                                           K_Global,
                                                                                           Split_K);
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint4* Compressed_A,
                            const uint64_t*   A1_Ind_data,
                            const int*   metadata,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K)
{
#ifdef DEBUG_MODE
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    // if (Split_K == 1)
        // SpMM_SplitK_OutputPTR = C;
    // else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<64>>(stream,
                                                                                    A,
                                                                                    Compressed_A,
                                                                                    A1_Ind_data,
                                                                                    metadata,
                                                                                    TileOffsets,
                                                                                    B,
                                                                                    SpMM_SplitK_OutputPTR,
                                                                                    M_Global,
                                                                                    N_Global,
                                                                                    K_Global,
                                                                                    Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<64>>(stream,
                                                                                 A,
                                                                                 Compressed_A,
                                                                                 A1_Ind_data,
                                                                                 metadata,
                                                                                 TileOffsets,
                                                                                 B,
                                                                                 SpMM_SplitK_OutputPTR,
                                                                                 M_Global,
                                                                                 N_Global,
                                                                                 K_Global,
                                                                                 Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<64>>(stream,
                                                                                 A,
                                                                                 Compressed_A,
                                                                                 A1_Ind_data,
                                                                                 metadata,
                                                                                 TileOffsets,
                                                                                 B,
                                                                                 SpMM_SplitK_OutputPTR,
                                                                                 M_Global,
                                                                                 N_Global,
                                                                                 K_Global,
                                                                                 Split_K);
            break;
        case 64:
            // return SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64> >
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 4>, SparseKernelConfig<64>>(stream,
                                                                                 A,
                                                                                 Compressed_A,
                                                                                 A1_Ind_data,
                                                                                 metadata,
                                                                                 TileOffsets,
                                                                                 B,
                                                                                 SpMM_SplitK_OutputPTR,
                                                                                 M_Global,
                                                                                 N_Global,
                                                                                 K_Global,
                                                                                 Split_K);
            break;
        // case 128:
        //     SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(stream,
        //                                                                          A,
        //                                                                          Compressed_A,
                                                                                    // A1_Ind_data,
        //                                                                          metadata,
        //                                                                          TileOffsets,
        //                                                                          B,
        //                                                                          SpMM_SplitK_OutputPTR,
        //                                                                          M_Global,
        //                                                                          N_Global,
        //                                                                          K_Global,
        //                                                                          Split_K);
        //     break;
        default:
        //     if (N_Global % 128 == 0)
        //         SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<32>>(stream,
        //                                                                              A,
        //                                                                              Compressed_A,
                                                                                        // A1_Ind_data,
        //                                                                              metadata,
        //                                                                              TileOffsets,
        //                                                                              B,
        //                                                                              SpMM_SplitK_OutputPTR,
        //                                                                              M_Global,
        //                                                                              N_Global,
        //                                                                              K_Global,
        //                                                                              Split_K);
        //     else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            // }
            break;
    }
    //
    // cudaError_t Error = cudaGetLastError();
    // if (Error != cudaSuccess)
    //     return Error;

    // if (Split_K == 1)
    //     return Error;
    // dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    // dim3 BlockDim(WARP_SIZE, 1, 1);
    // SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

static int BankID_Minimum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MinItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() < MinItemCount) {
            ID           = i;
            MinItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

static int BankID_Maximum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MaxItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() > MaxItemCount) {
            ID           = i;
            MaxItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

/*
return: Number of Element in array TileOffsets
Note: TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
*/
// template<typename TilingConfig, typename SparseKernelConfig>
__host__ int InitSparseMatrixA_API(half*      A_h,
                                   int        M,
                                   int        N,
                                   int        K,
                                   uint32_t** Compressed_A,  // CPU PTR
                                   int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int       TotalNZCount = 0;
    uint32_t* Ptr_SubArray = *Compressed_A;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half*        CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int          TileNZCount       = 0;
            int          remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            unsigned int Item              = 0;
            // Processing each tile
            std::vector<unsigned int> ItemsInBank[32];
            int                       ZeroPositionForBank[32];
            for (int k = 0; k < 32; k++)
                ZeroPositionForBank[k] = -1;
            //
            // printf("Starting Processing Tile i:%d j:%d...\n", i, j);
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    // Row permutation for bank-conflict-free shared memory layout
                    int      row            = m;
                    int      col            = n;
                    uint32_t mask           = (row % 8) << 3;
                    int      col_permutated = col ^ mask;
                    int      bank_smem      = (col_permutated / 2) % 32;
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(&Item);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        *short_ptr       = static_cast<short>(row * TILE_K + col_permutated);
                        ItemsInBank[bank_smem].push_back(Item);
                        //
                        TileNZCount++;
                    }
                    else {
                        if (ZeroPositionForBank[bank_smem] == -1)
                            ZeroPositionForBank[bank_smem] = row * TILE_K + col_permutated;
                    }
                }
            }
            //
            // printf("Starting Weight Padding...\n");
            for (int k = 0; k < remainingPaddings; k++) {
                int BankID = BankID_Minimum(ItemsInBank);
                assert(BankID >= 0 && BankID < 32);
                int ZeroPosition = ZeroPositionForBank[BankID];
                assert(ZeroPosition != -1);
                //
                half* half_ptr   = reinterpret_cast<half*>(&Item);
                *half_ptr        = __float2half_rn(0.0f);
                short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                *short_ptr       = static_cast<short>(ZeroPosition);
                ItemsInBank[BankID].push_back(Item);
                //
                TileNZCount++;
            }
            /*
            if(i==0 && j==0)
            {
              printf("For tile i:%d j:%d\n",i,j);
              for(int h=0; h<32; h++)
                printf("%ld ", ItemsInBank[h].size());
              printf("\n");
            }
            */
            //
            // printf("Starting Weight Shuffle...\n");
            std::vector<unsigned int> MainPart[32];
            std::vector<unsigned int> TailPart[32];
            int                       TileVectorCount = TileNZCount / VECTOR_SIZE;
            assert(TileNZCount % VECTOR_SIZE == 0);
            int Repeat_Vector   = TileVectorCount / WARP_SIZE;
            int Remained_Vector = TileVectorCount % WARP_SIZE;
            // Filing the TailPart
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    int BankID = BankID_Maximum(ItemsInBank);
                    Item       = ItemsInBank[BankID].back();
                    ItemsInBank[BankID].pop_back();
                    TailPart[b].push_back(Item);
                }
            }
            // Filing the MainPart
            // printf("Starting Filing the MainPart...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < WARP_SIZE; b++) {
                        int BankID = BankID_Maximum(ItemsInBank);
                        Item       = ItemsInBank[BankID].back();
                        ItemsInBank[BankID].pop_back();
                        MainPart[b].push_back(Item);
                    }
                }
            }
            // Writing to the Sub-Array
            // printf("Starting Writing to the Sub-Array...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < 32; b++) {
                        Item = MainPart[b].back();
                        MainPart[b].pop_back();
                        int V_Size                                     = VECTOR_SIZE;
                        Ptr_SubArray[r * V_Size * 32 + b * V_Size + v] = Item;
                    }
                }
            }
            Ptr_SubArray += Repeat_Vector * VECTOR_SIZE * WARP_SIZE;
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    Item = TailPart[b].back();
                    TailPart[b].pop_back();
                    Ptr_SubArray[b * VECTOR_SIZE + v] = Item;
                }
            }
            Ptr_SubArray += VECTOR_SIZE * Remained_Vector;
            //
            TotalNZCount += TileNZCount;
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    //
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //
    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = (row % 8) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}


// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder_v1(half*      A_h,
                                                int        M,
                                                int        N,
                                                int        K,
                                                uint32_t** Compressed_A,  // CPU PTR
                                                int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];

            int total_counter = 0;
            int nnz_counter   = 0;

            for (int m = 0; m < _TILE_M; m++) {

                for (int n = 0; n < TILE_K; n++) {

                    if (total_counter == 4) {
                        total_counter = 0;
                        nnz_counter   = 0;
                    }


                    total_counter++;

                    half value = CurrentTilePTR[m * K + n];

                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {

                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int row = m;
                        int col = n;

                        int local_k = nnz_counter;
                        nnz_counter++;

                        int local_k_id = col / 4;
                        int mod_col    = local_k_id * 2 + local_k;
                        // assert(mod_col<(TILE_K/2));
                        //                        uint32_t mask           = (row % 8) << 3;
                        //                        int      col_permutated = col ^ mask;
                        *short_ptr = static_cast<short>(row * (TILE_K / 2) + mod_col);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }

                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int row        = m;
                            int col        = n;
                            int local_k    = col % 4;
                            int local_k_id = col / 4;
                            int mod_col    = local_k_id * 2 + local_k;
                            // assert(mod_col<(TILE_K/2));

                            //                            uint32_t mask           = (row % 8) << 3;
                            //                            int      col_permutated = col ^ mask;
                            *short_ptr = static_cast<short>(row * (TILE_K / 2) + mod_col);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}


// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder_v2(half*      A_h,
                                                int        M,
                                                int        N,
                                                int        K,
                                                half** Compressed_A,  // CPU PTR
                                                int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (half*)malloc(NNZ_AfterPadding * sizeof(half));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];

            int total_counter = 0;

            for (int m = 0; m < _TILE_M; m++) {

                for (int n = 0; n < TILE_K; n++) {

                    // Permutation needed for avoiding bank conflicts using ldmatrix
                    int      row            = m;
                    int      col            = n;
//                    uint32_t mask           = (row % 8) << 3;
//                    int      col_permutated = col ^ mask;

                    if (total_counter == 4) {
                        total_counter = 0;
                    }

                    total_counter++;

//                    half value = CurrentTilePTR[row * K + col_permutated];
                    half value = CurrentTilePTR[row * K + col];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr  = *Compressed_A + TotalNZCount;
                        *half_ptr       = value;
                        TileNZCount++;
                        TotalNZCount++;
                    }

                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr  = *Compressed_A + TotalNZCount;
                            *half_ptr       = value;
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % (2*VECTOR_SIZE) == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / (2*VECTOR_SIZE);
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / (2 * VECTOR_SIZE);  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding //
                                           // adding an empty tile at last
    //

    //
    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder_v3(half*      A_h,
                                                int        M,
                                                int        N,
                                                int        K,
                                                half** Compressed_A,  // CPU PTR
                                                int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;

    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;


    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }

    *Compressed_A = (half*)malloc(M * K / 2 * sizeof(half));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }

    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;

            int nnz_counter   = 0;

            for (int m = 0; m < _TILE_M; m++) {

                for (int n = 0; n < TILE_K; n+=4) {

                    nnz_counter = 0;

                    for (int index_in_tile = 0; index_in_tile < 4; index_in_tile++) {

                        half value = CurrentTilePTR[m * K + n + index_in_tile];

                        if (fabs(__half2float(value)) > ZERO_THRESHOLD) {

                            half* half_ptr  = *Compressed_A + TotalNZCount;
                            *half_ptr       = value;
                            nnz_counter++;
                            TotalNZCount++;
                            TileNZCount++;
                            assert(nnz_counter <= 2);
                        }
                    }

                    for (int remain = nnz_counter; remain < 2; remain++) {
                        half* half_ptr  = *Compressed_A + TotalNZCount;
                        *half_ptr       = __float2half_rn(0.0f);
                        nnz_counter++;
                        TotalNZCount++;
                        TileNZCount++;
                    }
                }
            }
            //
            assert(TileNZCount % (2*VECTOR_SIZE) == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / (2*VECTOR_SIZE);
        }
    }

    // assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / (2 * VECTOR_SIZE);  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding //
                                           // adding an empty tile at last
//   int global_padd_nnz = 0;
//   for(int i=0; i < M / 256; i++){
//        for(int j=0; j < K / TILE_K; j++){
//            for(int m = 0; m < 256; m++){
//                int local_nnz = 0;
//                for(int n = 0; n < TILE_K; n++)
//               {
//                    half value = A_h[(i * 256 + m) * K + (j * TILE_K + n)];
//                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
//                        ++local_nnz;
//                    }
//               }
//               local_nnz = ((local_nnz -1) / 2 + 1)*2;
//                global_padd_nnz += local_nnz;
//            }
//        }
//    }
//
//    printf("global_padd_nnz = %d\n", global_padd_nnz);
//   getchar();

    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}


// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder_v4(half*      A_h,
                                                int        M,
                                                int        N,
                                                int        K,
                                                half** Compressed_A,  // CPU PTR
                                                int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int _TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;

    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / _TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;


    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }

    *Compressed_A = (half*)malloc(M * K / 2 * sizeof(half));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }

    // Generating compressed format for A Matrix
    assert(M % _TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;

            int nnz_counter   = 0;

            half *Compressed_Row = (half*)malloc(TILE_K * sizeof(half));
            int RowNZCount       = 0;
            
            for (int m = 0; m < _TILE_M; m++) {

                // Pad the rows with enough zeros to fit 2:4
                for (int n = 0; n < TILE_K; n += 4) {

                    int      row            = m;
                    int      col            = n;

                    nnz_counter = 0;

                    for (int index_in_tile = 0; index_in_tile < 4; index_in_tile++) {

                        half value = CurrentTilePTR[row * K + col + index_in_tile];

                        if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                            Compressed_Row[RowNZCount++] = value;
                            nnz_counter++;
                            TileNZCount++;
                            assert(nnz_counter <= 2);
                        }
                    }

                    for (int remain = nnz_counter; remain < 2; remain++) {
                        Compressed_Row[RowNZCount++] = __float2half_rn(0.0f);
                        nnz_counter++;
                        TileNZCount++;
                    }
                }

                // Every two rows will make one row of length 64 to fill the shmem banks
                if (m % 2 == 1) {
                    // Permute the row to avoid bank conflicts
                    for (int new_n = 0; new_n < TILE_K; new_n++) {

                        // Permutation needed for avoiding bank conflicts using ldmatrix
                        int      new_row            = m / 2; // new_n and new_m correspond to the SMEM layout
                        int      col            = new_n;
                        uint32_t mask           = (new_row % 4) << 3; // Permutation by 4 is enough for TILE_K / 2
                        // uint32_t mask           = 2 << 3; // Permutation by 4 is enough for TILE_K / 2
                        int      col_permutated = col ^ mask;

                        (*Compressed_A)[TotalNZCount++] = Compressed_Row[col_permutated];
                        // (*Compressed_A)[TotalNZCount++] = Compressed_Row[col];
                    }
                    
                    free(Compressed_Row);
                    Compressed_Row = (half*)malloc(TILE_K * sizeof(half));
                    RowNZCount       = 0;
                }
            }
            //
            assert(TileNZCount % (2*VECTOR_SIZE) == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / (2*VECTOR_SIZE);
        }
    }

    // assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / (2 * VECTOR_SIZE);  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding //
                                           // adding an empty tile at last

    return (M / _TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}


// This function implemented by Bangtian, load Structured Sparse Matrix A1 as sparse w/o significant padding
__host__ int InitSparseMatrixA1_Vec_Sparse_Load(half *A1,
                                                int M,
                                                int K,
                                                half **Compressed_A,  // CPU PTR
                                                uint64_t **Ind_data,
                                                int **TileOffsets) {
    // Change this tile from 256 to 128 to decrease the size of TB's
    const int _TILE_M = TILE_M;
    const int VECTOR_SIZE = 8;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
    const int BITS_PER_ELEMENT = 7;

    float ZERO_THRESHOLD = 0.0;
    int NumRow_offsets = M / _TILE_M;
    int NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A1[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;

    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }

    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A1 + (i * _TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }


    *TileOffsets = (int *) malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }

    int TotalNZCount = 0;
    // create TileOffsets first
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half *CurrentTilePTR = A1 + (i * _TILE_M) * K + (j * TILE_K);
            int TileNZCount = 0;
            int tile_nnz = 0;
            for (int m = 0; m < _TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (__half2float(value) != ZERO_THRESHOLD) {
                        ++tile_nnz;
                    }
                }
//                assert(row_nnz !=0);
//                if(row_nnz == 0)
//                    TileNZCount += 0;
//                else
//                    TileNZCount += ((row_nnz - 1) / VECTOR_SIZE + 1);
            }
            TileNZCount = ((tile_nnz - 1) / VECTOR_SIZE + 1) ;
            TotalNZCount += TileNZCount;
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount;
        }
    }

    // create a mapping from column indices to local indices into smem
    int *col_map = (int *) malloc(M *K * sizeof(int));
    // step 1: initialize col_map
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            col_map[i * K + j] = -1;
        }
    }

    // step 2: fill in col_map
    for(int i = 0; i < M / _TILE_M; i++){
        for(int j = 0; j < K / TILE_K; j++){
            half *CurrentTilePTR = A1 + (i * _TILE_M) * K + (j * TILE_K);
            for(int m = 0; m < _TILE_M; m++){
                for(int n = 0; n < TILE_K; n+=4){
                    vector<int> global_ind;
                    for(int ind = 0; ind < 4; ind++){
                        half value = CurrentTilePTR[m * K + n + ind];
                        if(__half2float(value) != ZERO_THRESHOLD){
                            //should be global indices
                            global_ind.push_back(j * TILE_K + n + ind);
//                            local_ind.push_back(n + ind);
                        }
                    }
                    assert(global_ind.size() <= 2);
                    int local_tile = n/4;
                    for(int ind = 0; ind < global_ind.size(); ind++){
                        int global_col_idx = global_ind[ind];
                        col_map[(i * _TILE_M + m) * K + global_col_idx] = local_tile * 2 + ind;
                    }
                }
            }
        }
    }

    printf("INTO TotalNZCount = %d\n", TotalNZCount);

    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / _TILE_M) * (K / TILE_K) + 1] =
            TotalNZCount;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding //
    // adding an empty tile at last

    *Compressed_A = (half *) malloc(TotalNZCount * sizeof(half) * VECTOR_SIZE);
    *Ind_data = (uint64_t *) malloc(TotalNZCount * sizeof(uint64_t));
    int v_offset = 0;
    int d_offset = 0;
    int max_offset = 0;
    for (int i = 0; i < M / _TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half *CurrentTilePTR = A1 + (i * _TILE_M) * K + (j * TILE_K);
            vector<int> tile_offset;
            vector<half> value_vec;
            vector<uint64_t> index_vec;
            vector<int> row_vec;
            for (int m = 0; m < _TILE_M; m++) {
//                int row_nnz = 0;
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (__half2float(value) != ZERO_THRESHOLD) {
                        value_vec.push_back(value);
                        int local_col = col_map[(i * _TILE_M + m) * K + j * TILE_K + n];
                        assert(local_col != -1);
                        assert(local_col < 32);
                        uint64_t l_col = local_col;
                        index_vec.push_back(l_col);
                        row_vec.push_back(m);
                    }
                }
            }

            assert(value_vec.size()==index_vec.size());
            assert(value_vec.size()==row_vec.size());

            int ind = 0;
            for (; ind + VECTOR_SIZE <= value_vec.size(); ind += VECTOR_SIZE) {
                uint64_t index_data = 0;
                uint64_t row_ind = row_vec[ind];
                index_data |= (row_ind << (VECTOR_SIZE * BITS_PER_ELEMENT));
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    (*Compressed_A)[v_offset + k] = value_vec[ind + k];
                    if(k==0){
                        index_data |= (index_vec[ind]);
                    }
                    else{
                        if(row_vec[ind+k] == row_vec[ind+k-1]){
                            index_data |= ((index_vec[ind + k] - index_vec[ind + k - 1]) << (BITS_PER_ELEMENT * k));
                            assert(index_vec[ind + k-1] < index_vec[ind + k]);
                        }
                        else{
                            int prev_offset = row_vec[ind + k - 1] * (TILE_K / 2) + index_vec[ind + k - 1];
                            int curr_offset = row_vec[ind + k] * (TILE_K / 2) + index_vec[ind + k];
                            uint64_t diff = curr_offset - prev_offset;
                            index_data |= (diff << (BITS_PER_ELEMENT*k));
//                            printf("diff = %d %d %d\n", diff, row_vec[ind + k-1], row_vec[ind + k]);
                            tile_offset.push_back(diff);
                            assert(diff < (1<<BITS_PER_ELEMENT));
                        }
                    }

//                    uint64_t index = index_vec[ind + k];
//                    (*Ind_data)[d_offset + k] = index;
                }
                (*Ind_data)[d_offset] = index_data;
                v_offset += VECTOR_SIZE;
                d_offset += 1;
            }

            if(ind == value_vec.size()){
                continue;
            }

            assert(ind < value_vec.size());

            //padding
            int v_ind = 0;
            uint64_t index_data = 0;
            uint64_t row_ind = row_vec[ind];
            index_data |= (row_ind << VECTOR_SIZE * BITS_PER_ELEMENT);

            // uint64_t last_row_ind;
            // uint64_t last_col_ind;
            half last_value;
            for (; ind + v_ind < value_vec.size(); v_ind++) {
                (*Compressed_A)[v_offset ++] = value_vec[ind+v_ind];
                if(v_ind==0){
                    index_data |= (index_vec[ind+v_ind]);
                }
                else{
                    if(row_vec[ind+v_ind] == row_vec[ind+v_ind-1]){
                        index_data |= ((index_vec[ind + v_ind] - index_vec[ind + v_ind - 1])
                                << (BITS_PER_ELEMENT * v_ind));
                    }
                    else{
                        int prev_offset = row_vec[ind + v_ind - 1] * (TILE_K / 2) + index_vec[ind + v_ind - 1];
                        int curr_offset = row_vec[ind + v_ind] * (TILE_K / 2) + index_vec[ind + v_ind];
                        uint64_t diff = curr_offset - prev_offset;
                        index_data |= (diff << (BITS_PER_ELEMENT*v_ind));
                        assert(diff < (1 << BITS_PER_ELEMENT));
                        tile_offset.push_back(diff);
                    }
                }
                if(ind + v_ind == value_vec.size() - 1){
                    // last_row_ind = row_vec[ind + v_ind];
                    // last_col_ind = index_vec[ind + v_ind];
                    last_value = value_vec[ind + v_ind];
                }
            }

            int padding_size = VECTOR_SIZE - value_vec.size() % VECTOR_SIZE;
            // int last_offset = last_row_ind * (TILE_K / 2) + last_col_ind;
//            if(last_offset + padding_size >= 256 * (TILE_K / 2)){
//                printf("last_offset = %d, padding_size = %d\n", last_offset, padding_size);
//                printf("last_row_ind = %d, last_col_ind = %d\n", last_row_ind, last_col_ind);
//            }
//            assert(last_offset  + padding_size < 256 * (TILE_K / 2));
            for(int i = 0; i < padding_size; i++){
                (*Compressed_A)[v_offset ++] = last_value;
                index_data |= (( 0) << (BITS_PER_ELEMENT * (i + v_ind)));
            }
            (*Ind_data)[d_offset++] = index_data;


//            //compute min, max, mean, mean of tile_offset
            int min = 100000;
            int max = 0;
            int sum = 0;
            for(int i = 0; i < tile_offset.size(); i++){
                if(tile_offset[i] < min){
                    min = tile_offset[i];
                }
                if(tile_offset[i] > max){
                    max = tile_offset[i];
                }
                sum += tile_offset[i];
            }
            int mean = sum / tile_offset.size();
            // int mean_of_tile_offset = mean;
//             printf("%d %d min = %d, max = %d, mean = %d, mean_of_tile_offset = %d\n", i, j, min, max, mean, mean_of_tile_offset);
             if(max_offset < max){
                 max_offset = max;
             }
        }
    }

    printf("max_offset = %d\n", max_offset);
    printf("v_offset = %d, d_offset = %d\n", v_offset, d_offset);
    printf("Into Function: TotalNZCount = %d\n", TotalNZCount);


    return (M / _TILE_M) * (K / TILE_K) + 2; // number of Elements in array TileOffsets
}


__host__ void inspect_metadata(half* mat, int* meta, int M, int K)
{
    std::map<std::string, int> metaMap;
    metaMap["1100"] = 0x4;
    metaMap["1010"] = 0x8;
    metaMap["1001"] = 0xC;
    metaMap["0110"] = 0x9;
    metaMap["0101"] = 0xD;
    metaMap["0011"] = 0xE;

    metaMap["1000"] = 0x0;
    metaMap["0100"] = 0x1;
    metaMap["0010"] = 0x2;
    metaMap["0001"] = 0x3;

    metaMap["0000"] = 0xF;

    std::map<std::string, int> metaStats;

    metaStats["1100"] = 0;
    metaStats["1010"] = 0;
    metaStats["1001"] = 0;
    metaStats["0110"] = 0;
    metaStats["0101"] = 0;
    metaStats["0011"] = 0;
    metaStats["1000"] = 0;
    metaStats["0100"] = 0;
    metaStats["0010"] = 0;
    metaStats["0001"] = 0;
    metaStats["0000"] = 0;


    const int total_size = (M / 16) * (K / 16);

    int* buffer = new int[total_size * 8];

    int elem_id = 0;
    for (int m = 0; m < M / 16; m++) {
        for (int k = 0; k < K / 16; k++) {
            for (int m2 = 0; m2 < 8; m2++) {
                unsigned int metadata = 0;
                for (int k2 = 0; k2 < 4; k2++) {
                    std::string key     = "";
                    int         counter = 0;
                    for (int i = 0; i < 4; i++) {
                        int   index = (m * 16 + m2) * K + k * 16 + k2 * 4 + i;
                        float value = __half2float(mat[index]);

                        if (value != 0.0f) {
                            key += "1";
                            counter++;
                        }
                        else {
                            key += "0";
                        }
                    }

                    metadata |= metaMap[key] << (k2 * 4);
                    metaStats[key]++;
                }
                for (int k2 = 0; k2 < 4; k2++) {
                    std::string key     = "";
                    int         counter = 0;
                    for (int i = 0; i < 4; i++) {
                        int   index = (m * 16 + m2 + 8) * K + k * 16 + k2 * 4 + i;
                        float value = __half2float(mat[index]);

                        if (value != 0.0f) {
                            key += "1";
                            counter++;
                        }
                        else {
                            key += "0";
                        }
                    }

                    metadata |= metaMap[key] << (k2 * 4 + 16);
                    metaStats[key]++;
                }
                int blockId = m * K / 16 + k;

                buffer[blockId * 8 + m2] = metadata;
                // buffer[blockId * 8 + m2] = elem_id++;
            }
        }
    }

    // for (int i = 0; i < total_size; i += BLOCK_K_TENSORS) {
    //     for (int j = 0; j < 8; j++) {
    //         for (int k = 0; k < BLOCK_K_TENSORS; k++) {
    //             meta[i * 8 + j * BLOCK_K_TENSORS + k] = buffer[i * 8 + j + k * 8];
    //         }
    //     }
    // }

    // printf("INSPECTION Initial buffer:\n");
    // // for (int i = 0; i < TILE_M / 16 * TILE_K / 16 * 8; i++) {
    // for (int i = TILE_M / 16 * TILE_K / 16 * 8; i < TILE_M / 16 * TILE_K / 16 * 8 * 2; i++) {
    //     printf("%d ", buffer[i]);
    //     if (i % 16 == 15)
    //         printf("\n");
    // }
    // for (int i = 0; i < TILE_M / 16; i++)
    //     for (int k = 0; k < 8; k++) {
    //         for (int j = 0; j < TILE_K / 16; j++) {
    //             printf("%d ", buffer[i * K / 16 * 8 + j * 8 + k]);
    //         }
    //         printf("\n");
    //     }

    // reorder for vectorized loading
    int meta_id = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            for (int ii = 0; ii < TILE_M / 16; ii++) {
                for (int jj = 0; jj < TILE_K / 16 / BLOCK_K_TENSORS; jj++) {
                    for (int row = 0; row < 8; row++) {
                        for (int k = 0; k < BLOCK_K_TENSORS; k++) {
                            int original_tile_row = i * TILE_M / 16 + ii;
                            int original_tile_col = j * TILE_K / 16 + jj * BLOCK_K_TENSORS + k;
                            meta[meta_id++] =
                                // buffer[i * K / 16 * TILE_M / 16 * 8 + j * TILE_M / 16 * TILE_K / 16 * 8 +
                                //         ii * K / 16 * 8 + jj * BLOCK_K_TENSORS * 8 + k * 8 + row];
                                buffer[original_tile_row * K / 16 * 8 + original_tile_col * 8 + row];
                        }
                    }
                }
            }
        }
    }

    // printf("INSPECTION Sorted buffer:\n");
    // // for (int i = 0; i < TILE_M / 16 * TILE_K / 16 * 8; i++) {
    // for (int i = TILE_M / 16 * TILE_K / 16 * 8; i < TILE_M / 16 * TILE_K / 16 * 8 * 2; i++) {
    //     printf("%d ", meta[i]);
    //     if (i % 16 == 15)
    //         printf("\n");
    // }

    // Count all metadata
    int total_patterns = 0;
    for (auto& kv : metaStats) {
        total_patterns += kv.second;
    }

    // Show statistics for different metadata
    printf("Metadata Inspection:\n");
    for (auto& kv : metaStats) {
        std::cout << kv.first << " " << kv.second << " " << kv.second * 100.0 / total_patterns << "%" << std::endl;
    }

    delete[] buffer;
}


/*
input:    char* DenseMatrixFileName
          int   M
          int   N                   // N is used by void InitSparseMatrixA_API()
          int   K
          char* NZWeightsFileName
          char* TileOffsetsFileName
          char* OutputSizesFileName // NNZ -> NumOffsets
*/
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    unsigned int* NZWeights_CPU   = nullptr;
    int*          TileOffsets_CPU = nullptr;
    int           NumOffsets      = InitSparseMatrixA_API(host_array.data(), M, 0, K, &NZWeights_CPU, &TileOffsets_CPU);
    int           NNZ             = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // Step 3: Write to FILE(OutputSizesFileName)
    //         Write to FILE(NZWeightsFileName), FILE(TileOffsetsFileName)
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);
    std::ofstream out_NZWeightsFile(NZWeightsFileName, std::ios::out | std::ios::binary);
    std::ofstream out_TileOffsetsFile(TileOffsetsFileName, std::ios::out | std::ios::binary);
    if (!out_SizesFile.is_open() || !out_NZWeightsFile.is_open() || !out_TileOffsetsFile.is_open()) {
        printf("GenSparseMatrixBinFile() ERROR: file %s, %s, or %s cannot be opened or creaetd. \n",
               OutputSizesFileName,
               NZWeightsFileName,
               TileOffsetsFileName);
        exit(-1);
    }
    //
    // out_SizesFile << NNZ << NumOffsets;
    out_SizesFile.write((char*)&NNZ, sizeof(int));
    out_SizesFile.write((char*)&NumOffsets, sizeof(int));
    out_SizesFile.close();
    out_NZWeightsFile.write((char*)NZWeights_CPU, sizeof(uint32_t) * NNZ);
    out_NZWeightsFile.close();
    out_TileOffsetsFile.write((char*)TileOffsets_CPU, sizeof(int) * NumOffsets);
    out_TileOffsetsFile.close();
}