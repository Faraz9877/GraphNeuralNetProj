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

#include "MatMulUtilities.cuh"
#include <vector>


template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg_A1_Dense_Load(uint32_t*    Registers_GlobalToShared1,
                                                           uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                           const uint4* GlobalPTR1,
                                                           int          NNZ_VECTOR_ThisTile1,
                                                           uint32_t*    Registers_GlobalToShared2,
                                                           uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                           const uint4* GlobalPTR2,
                                                           int          NNZ_VECTOR_ThisTile2)
{
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
#pragma unroll
    for (int i = 0; i < (SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / (SparseKernelConfig::VECTOR_SIZE*4)); i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        const uint4 GlobalVal1 = GlobalPTR1[index];
        Registers_GlobalToShared1[i * 4 + 0] = GlobalVal1.x;
        Registers_GlobalToShared1[i * 4 + 1] = GlobalVal1.y;
        Registers_GlobalToShared1[i * 4 + 2] = GlobalVal1.z;
        Registers_GlobalToShared1[i * 4 + 3] = GlobalVal1.w;
    }

    if(NNZ_VECTOR_ThisTile2 != 0){
#pragma unroll
        for (int i = 0; i < (SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / (SparseKernelConfig::VECTOR_SIZE*4)); i++) {
            int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
            const uint4 GlobalVal2 = GlobalPTR2[index];
            Registers_GlobalToShared2[i * 4 + 0] = GlobalVal2.x;
            Registers_GlobalToShared2[i * 4 + 1] = GlobalVal2.y;
            Registers_GlobalToShared2[i * 4 + 2] = GlobalVal2.z;
            Registers_GlobalToShared2[i * 4 + 3] = GlobalVal2.w;
        }
    }
}


// used for loading structured sparse matrix A1 directly to avoid padding version 
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg_A1_Sparse_Load(uint32_t*    Registers_GlobalToShared,
                                                         uint64_t* Registers_Ind,   
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal,
                                                         const uint4* GlobalPTR_val,
                                                         const uint64_t* GlobalPTR_Ind,
                                                         int          NNZ_VECTOR_ThisTile)
{
    // here we assume the tile is 256(128) x 64, one thread block is composed of 4 warps
    // Load Global to registers
    int Num_NNZ_Vector = NNZ_VECTOR_ThisTile / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if(threadIdx.x < (NNZ_VECTOR_ThisTile % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector++;
    *NNZ_VECTOR_ThreadLocal = Num_NNZ_Vector;

    //TODO: may cause bugs here, will check it later
#pragma unroll
    for(int i=0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / (SparseKernelConfig::VECTOR_SIZE*2); i++){
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if(index >= NNZ_VECTOR_ThisTile)
            break;
        Registers_GlobalToShared[i * 4 + 0] = GlobalPTR_val[index].x;
        Registers_GlobalToShared[i * 4 + 1] = GlobalPTR_val[index].y;
        Registers_GlobalToShared[i * 4 + 2] = GlobalPTR_val[index].z;
        Registers_GlobalToShared[i * 4 + 3] = GlobalPTR_val[index].w;
        Registers_Ind[i] = GlobalPTR_Ind[index];
    }
    return;
}


template<typename TilingConfig>
__device__ __forceinline__ void Load_Meta_Data_to_Smem(const int*   meta_addr,
                                                       int32_t* __restrict__ sMeta)
{
    constexpr int VEC_SIZE = 4;
    // Load meta_data to smem
    int lane_id = threadIdx.x % WARP_SIZE;

    // Load WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8 int32_t values by one TB
    constexpr int TotalNumOfCopyUnit = WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * METADATA_PER_TENSOR;
    constexpr int MaxIteration = (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_THREADS * VEC_SIZE) + 1;
    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        if ((i * WARP_SIZE + lane_id) * VEC_SIZE >= WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR)
            break;
        cp_async<16>(reinterpret_cast<half*>(sMeta + (i * WARP_SIZE + lane_id) * VEC_SIZE),
                     reinterpret_cast<const half*>(meta_addr + (i * WARP_SIZE + lane_id) * VEC_SIZE));
    }
}


template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void Load_Meta_Data(const int* meta_a,
                                               int32_t __restrict__ meta_data[][BLOCK_K_TENSORS],
                                               int meta_stride)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int groupID = lane_id >> 2;

    for(int i=0; i < WARP_ROW_TENSORS; i++){
        // vectorized version
        if (BLOCK_K_TENSORS == 4) {
            int4 tmp = reinterpret_cast<const int4*>(meta_a + (i * meta_stride) * 8)[groupID];
            meta_data[i][0] = tmp.x;
            meta_data[i][1] = tmp.y;
            meta_data[i][2] = tmp.z;
            meta_data[i][3] = tmp.w;
        }
        else if (BLOCK_K_TENSORS == 2) {
            int2 tmp = reinterpret_cast<const int2*>(meta_a + (i * meta_stride) * 8)[groupID];
            meta_data[i][0] = tmp.x;
            meta_data[i][1] = tmp.y;
        }
    }
}


template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared_dense_load(half* __restrict__ SharedPTR1,
                                                                    uint32_t* Registers_For_SparseTiles1,
                                                                    uint32_t  NNZ_ThreadLocal1,
                                                                    half* __restrict__ SharedPTR2,
                                                                    uint32_t* Registers_For_SparseTiles2,
                                                                    uint32_t  NNZ_ThreadLocal2)
{
    #pragma unroll
    for (int i = 0; i < (SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL
                         / (SparseKernelConfig::VECTOR_SIZE * BLOCK_ROW_WARPS));
         i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));

        short base = i * TilingConfig::BLOCK_THREADS + threadIdx.x;

        uint4* half_ptr =
            reinterpret_cast<uint4*>(&(Registers_For_SparseTiles1[i * SparseKernelConfig::VECTOR_SIZE]));

        uint4 value      = *half_ptr;

        reinterpret_cast<uint4*>(&SharedPTR1[base * 8])[0] = value;
        
        // cp_async<16>(SharedPTR1 + base * 8, GlobalPTR1 + index * 8, AsyncCopyPredictor);
    }

    if (TILE_M == 256) {
#pragma unroll
        for (int i = 0; i < (SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL
                             / (SparseKernelConfig::VECTOR_SIZE * BLOCK_ROW_WARPS));
             i++) {
            int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));

            short base = i * TilingConfig::BLOCK_THREADS + threadIdx.x;

            uint4* half_ptr =
            reinterpret_cast<uint4*>(&(Registers_For_SparseTiles2[i * SparseKernelConfig::VECTOR_SIZE]));

            uint4 value      = *half_ptr;

            reinterpret_cast<uint4*>(&SharedPTR2[base * 8])[0] = value;


            // cp_async<16>(SharedPTR2 + base * 8, GlobalPTR2 + index * 8, AsyncCopyPredictor);
        }
    }
}


// Init Shared Memory to 0
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    // modified version
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    //
    static_assert(TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TILE_M / TilingConfig::BLOCK_WARPS;
    //
    // static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * (TILE_K/2) + HALF_PER_128B * lane_id;
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / (TILE_K/2)) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / (TILE_K/2));
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}


template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared_sparse_load(half* __restrict__ SharedPTR,
                                                                    uint64_t* Registers_For_IndData,
                                                                    uint32_t* Registers_For_SparseTiles,
                                                                    uint32_t  NNZ_ThreadLocal)
{
    constexpr int XOR_RANGE = (HALF_PER_128B / (HALF_PER_SHMEM_BANK / (TILE_K / 2)));

    #pragma unroll
    for(int i=0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / (SparseKernelConfig::VECTOR_SIZE*2); i++){
        if(i >= NNZ_ThreadLocal)
            break;
        int idx = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
//        __syncthreads();
        uint64_t A1_Ind_data = Registers_For_IndData[i];
//        //obtain row index
        uint16_t row = A1_Ind_data >> (56);
//        __syncthreads();
        uint16_t first_col_idx = (A1_Ind_data) & 0x7F;
        uint16_t smem_addr = row * (TILE_K/2) + first_col_idx;
        half *half_ptr = reinterpret_cast<half *>(&(Registers_For_SparseTiles[i * SparseKernelConfig::VECTOR_SIZE +
                                                                              0]));

        int new_i = smem_addr / (HALF_PER_SHMEM_BANK);
        int new_j = smem_addr % (HALF_PER_SHMEM_BANK);
        uint32_t mask = (new_i % XOR_RANGE) << 3;
        new_j = new_j ^ mask;
        // if (blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("GtoS: tid: %d, new_i: %d, new_j: %d, XOR_RANGE: %d\n", threadIdx.x, new_i, new_j, XOR_RANGE);
        // }

        half value_1 = *half_ptr;
        half value_2 = *(half_ptr + 1);

        SharedPTR[new_i * HALF_PER_SHMEM_BANK + new_j] = value_1;
        uint16_t second_offset = (A1_Ind_data >> (7)) & 0x7F;
        smem_addr += second_offset;

        new_i = smem_addr / (HALF_PER_SHMEM_BANK);
        new_j = smem_addr % (HALF_PER_SHMEM_BANK);
        mask = (new_i % XOR_RANGE) << 3;
        new_j = new_j ^ mask;

        SharedPTR[new_i * HALF_PER_SHMEM_BANK + new_j] = value_2;


#pragma unroll
        for(int j=1; j < SparseKernelConfig::VECTOR_SIZE; j++){
            uint16_t offset_1 = (A1_Ind_data >> (2 * j * 7)) & 0x7F;
            uint16_t offset_2 = (A1_Ind_data >> ((2 * j + 1) * 7)) & 0x7F;

            half *half_ptr = reinterpret_cast<half *>(&(Registers_For_SparseTiles[i * SparseKernelConfig::VECTOR_SIZE +
                                                                                  j]));
            half value_1 = *half_ptr;
            half value_2 = *(half_ptr + 1);
            smem_addr += offset_1;

            new_i = smem_addr / (HALF_PER_SHMEM_BANK);
            new_j = smem_addr % (HALF_PER_SHMEM_BANK);
            mask = (new_i % XOR_RANGE) << 3;
            new_j = new_j ^ mask;

            SharedPTR[new_i * HALF_PER_SHMEM_BANK + new_j] = value_1;
            smem_addr += offset_2;

            new_i = smem_addr / (HALF_PER_SHMEM_BANK);
            new_j = smem_addr % (HALF_PER_SHMEM_BANK);
            mask = (new_i % XOR_RANGE) << 3;
            new_j = new_j ^ mask;

            SharedPTR[new_i * HALF_PER_SHMEM_BANK + new_j] = value_2;
        }
    }

    return;
}


template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel(const half*  A,
                            const uint4* Compressed_A1,
                            const int*   TileOffsets,
                            const uint64_t* A1_Ind_data,
                            const int*   metadata,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    //
    const int BatchID     = blockIdx.y / (M_Global / TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    //
    const int* TileOffsets_ThisBlock1 = nullptr;
    uint32_t Registers_GlobalToShared[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL/2];
    uint32_t NNZ_ThreadLocal1 = 0;

#ifdef SPARSE_LOAD_A
    TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    uint64_t Registers_Ind[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / (8)];
#else
    const int* TileOffsets_ThisBlock2 = nullptr;
    if (TILE_M == 256) {
        TileOffsets_ThisBlock1 =
            TileOffsets + K_Global / TILE_K * y * 2
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
        TileOffsets_ThisBlock2 =
            TileOffsets + K_Global / TILE_K * (y * 2 + 1)
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration

    }
    else {
        TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
        TileOffsets_ThisBlock2 = TileOffsets_ThisBlock1;  // otherwise will cause problem when passing
                                                          // TileOffsets_ThisBlock2[0] to SpMM_CopyFromGlobalToReg()
    }
    uint32_t NNZ_ThreadLocal2 = 0;
#endif
    //
    //
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][2];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];

    int32_t __restrict__      metadata_a[WARP_ROW_TENSORS][BLOCK_K_TENSORS];

    // copying B tile from GlobalMemory to SharedMemory
    // const half* BTileGlobalPTR =
    //     B + Tile_Start_N * K_Global
    //     + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    
    // For row-major B
    const half* BTileGlobalPTR =
        B + BatchID * AverageNumKBlock * TILE_K * N_Global
        + Tile_Start_N;  // Address for matrix B, taking SplitK into consideration
    //

    int NNZ_ThisTile1 = TileOffsets_ThisBlock1[1] - TileOffsets_ThisBlock1[0];

#ifdef SPARSE_LOAD_A
    SpMM_CopyFromGlobalToReg_A1_Sparse_Load<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               Registers_Ind,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A1 + TileOffsets_ThisBlock1[0],
                                                               A1_Ind_data + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1);
    // Initialize Shared Memory to 0 for Structured Sparse A
    SpMM_InitSharedMemory<TilingConfig>(smem);
    cp_async_group_commit();
#else
    int NNZ_ThisTile2 = 0;
    if (TILE_M == 256)
        NNZ_ThisTile2 = TileOffsets_ThisBlock2[1] - TileOffsets_ThisBlock2[0];

    SpMM_CopyFromGlobalToReg_A1_Dense_Load<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A1 + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1,
                                                               Registers_GlobalToShared
                                                                   + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 4,
                                                               &NNZ_ThreadLocal2,
                                                               Compressed_A1 + TileOffsets_ThisBlock2[0],
                                                               NNZ_ThisTile2);
#endif

    const int* meta_addr = metadata + (y * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS) * NumKBlock +
                                        Warp_i * WARP_ROW_TENSORS +
                                        BatchID * AverageNumKBlock * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS)) *
                                    BLOCK_K_TENSORS * METADATA_PER_TENSOR;
    int32_t* __restrict__ sMeta = reinterpret_cast<int32_t*>(smem + (TILE_M * (TILE_K / 2) + TilingConfig::TILE_N * TILE_K) * 2);
    
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x % 32 == 0) {
    //     printf("warpId: %d, meta_addr + Warp_i * WARP_ROW_TENSORS * NumKBlock * BLOCK_K_TENSORS * METADATA_PER_TENSOR = %d\n", warpId,
    //         *(meta_addr));
    // }
    
    Load_Meta_Data_to_Smem<TilingConfig>(
        meta_addr,
        sMeta + Warp_i * WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR);
    cp_async_group_commit();

    // cp_async_wait_group<0>();
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     printf("sMeta:\n");
    //     // for (int i = 0; i < WARP_ROW_TENSORS * BLOCK_ROW_WARPS * 8; i++) {
    //     //     for (int j = 0; j < BLOCK_K_TENSORS; j++) {
    //     //         printf("%d ", sMeta[i * BLOCK_K_TENSORS + j]);
    //     //     }
    //     //     printf("\n");
    //     // }
    //     for (int i = 0; i < WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8; i++) {
    //         printf("%d ", sMeta[i]);
    //         if (i % 16 == 15)
    //             printf("\n");
    //     }
    //     printf("\n");
    //     printf("\n");
    // }
    // __syncthreads();

    // Initilzaing Shared Memory to 0 for fine-grained A2
//    SpMM_InitSharedMemory<TilingConfig>(smem + (TILE_M * TILE_K/2) + TilingConfig::TILE_N * TILE_K);
    // cp_async_group_commit();
    
    // CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
    //     smem + (TILE_M * TILE_K/2), BTileGlobalPTR, K_Global);
    
    // For row-major B
    CopyTileFromGlobalToShared_64_X<TilingConfig::TILE_N2, TilingConfig>(
        smem + TILE_M * (TILE_K/2), BTileGlobalPTR, N_Global);
    cp_async_group_commit();

    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            reinterpret_cast<half2*>(&c[i][j])[0] = __half2_raw();

    //
    cp_async_wait_group<1>();
    __syncthreads();
    
    // // Load meta_data
    // Load_Meta_Data<TilingConfig, SparseKernelConfig>(
    //     sMeta + Warp_i * WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR,
    //     metadata_a_prefetch, BLOCK_K_TENSORS);

#ifdef SPARSE_LOAD_A
    SpMM_DecompressFromRegisterToShared_sparse_load<TilingConfig, SparseKernelConfig>(smem,
        Registers_Ind,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1);
    
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     printf("sA:\n");
    //     for (int i = 0; i < TILE_M / 4; i++) {
    //         for (int j = 0; j < TilingConfig::TILE_N2; j++) {
    //             printf("%f ", __half2float(smem[i * TILE_K / 2 + j]));
    //         }
    //         printf("\n");
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     half* sB = smem + TILE_M * TILE_K / 2;
    //     printf("sB:\n");
    //     for (int i = 0; i < TILE_K / 2; i++) {
    //         for (int j = 0; j < TilingConfig::TILE_N2; j++) {
    //             printf("%f ", __half2float(sB[i * TilingConfig::TILE_N + j]));
    //         }
    //         printf("\n");
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

#else
    SpMM_DecompressFromRegisterToShared_dense_load<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1,
        smem + TILE_M * TILE_K / 4,
        Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 4,
        NNZ_ThreadLocal2);
#endif
    //
    cp_async_wait_group<0>();
    __syncthreads();
    // Prefetch to reduce stall_long_sb
    int StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[0 + 1];
    int NNZ_ThisTile_Prefetch1           = TileOffsets_ThisBlock1[0 + 2] - TileOffsets_ThisBlock1[0 + 1];

#ifndef SPARSE_LOAD_A
    int StartIndex_SparseTiles_Prefetch2 = 0;
    int NNZ_ThisTile_Prefetch2           = 0;
    if (TILE_M == 256) {
        StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[0 + 1];
        NNZ_ThisTile_Prefetch2           = TileOffsets_ThisBlock2[0 + 2] - TileOffsets_ThisBlock2[0 + 1];
    }
#endif

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first

    
    // for (int thid = 0; thid < 128; thid++) {
    //     __syncthreads();
    //     if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == thid) {
    //         printf("sMeta metadata_a_prefetch tid %d:\n", thid);
    //         for (int i = 0; i < WARP_ROW_TENSORS; i++) {
    //             for (int j = 0; j < BLOCK_K_TENSORS; j++) {
    //                 printf("%d ", metadata_a_prefetch[i][j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     __syncthreads();
    // }

#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter - 1; tile_id_k++) {
        // Using the previous prefetched value
        int StartIndex_SparseTiles1 = StartIndex_SparseTiles_Prefetch1;
        int NNZ_ThisTile1           = NNZ_ThisTile_Prefetch1;
        StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        if(tile_id_k < NumIter-1)
            NNZ_ThisTile_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 2] - TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        else
            NNZ_ThisTile_Prefetch1 = 0;

#ifndef SPARSE_LOAD_A
        int StartIndex_SparseTiles2 = 0;
        int NNZ_ThisTile2           = 0;
        if (TILE_M == 256) {
            StartIndex_SparseTiles2 = StartIndex_SparseTiles_Prefetch2;
            NNZ_ThisTile2           = NNZ_ThisTile_Prefetch2;
        }
        if (TILE_M == 256) {
            StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
            if(tile_id_k < NumIter-1)
                NNZ_ThisTile_Prefetch2 =
                    TileOffsets_ThisBlock2[tile_id_k + 1 + 2] - TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
            else
                NNZ_ThisTile_Prefetch2 = 0;
        }
#endif
        //
        // copying B tile from GlobalMemory to SharedMemory
        // BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        // For row-major B
        BTileGlobalPTR = B + (BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K)) * N_Global + Tile_Start_N;

        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TILE_M * (TILE_K/2) + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TILE_M * (TILE_K/2) + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

#ifdef SPARSE_LOAD_A
        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();

        const int* meta_addr = metadata + (y * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS) * NumKBlock +
                                            Warp_i * WARP_ROW_TENSORS +
                                            (tile_id_k + 1) * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS) +
                                            BatchID * AverageNumKBlock * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS)) *
                                        BLOCK_K_TENSORS * METADATA_PER_TENSOR;
        
        int32_t* __restrict__ sMeta_write_PTR = 
            reinterpret_cast<int32_t*>(sMeta + ((tile_id_k + 1) % 2) * WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8);
        int32_t* __restrict__ sMeta_read_PTR =
            reinterpret_cast<int32_t*>(sMeta + ((tile_id_k) % 2) * WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8);

        Load_Meta_Data_to_Smem<TilingConfig>(
            meta_addr,
            sMeta_write_PTR + Warp_i * WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR);
        cp_async_group_commit();

        // cp_async_wait_group<0>();
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        //     printf("sMeta 2:\n");
        //     for (int i = 0; i < WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8; i++) {
        //         printf("%d ", sMeta_write_PTR[i]);
        //         if (i % 16 == 15)
        //             printf("\n");
        //     }
        //     printf("\n");
        //     printf("\n");
        // }
        // __syncthreads();

        SpMM_CopyFromGlobalToReg_A1_Sparse_Load<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
            Registers_Ind,
            &NNZ_ThreadLocal1,
            Compressed_A1 + StartIndex_SparseTiles1,
            A1_Ind_data + StartIndex_SparseTiles1,
            NNZ_ThisTile1);
#else
        cp_async_group_commit();
        SpMM_CopyFromGlobalToReg_A1_Dense_Load<TilingConfig, SparseKernelConfig>(
            Registers_GlobalToShared,
            &NNZ_ThreadLocal1,
            Compressed_A1 + StartIndex_SparseTiles1,
            NNZ_ThisTile1,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 4,
            &NNZ_ThreadLocal2,
            Compressed_A1 + StartIndex_SparseTiles2,
            NNZ_ThisTile2);
#endif
        // Copying B Tile
        // CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        //     smem_write_PTR + TILE_M * (TILE_K/2), BTileGlobalPTR, K_Global, GlobalCopy);
        
        // For row-major B
        CopyTileFromGlobalToShared_64_X<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TILE_M * (TILE_K/2), BTileGlobalPTR, N_Global, GlobalCopy);
        cp_async_group_commit();

        // Load meta_data
        Load_Meta_Data<TilingConfig, SparseKernelConfig>(
            sMeta_read_PTR + Warp_i * WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR,
            metadata_a, BLOCK_K_TENSORS);
        
        // for (int thid = 0; thid < 128; thid++) {
        //     __syncthreads();
        //     if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == thid) {
        //         printf("sMeta_read_PTR metadata_a_prefetch tid %d:\n", thid);
        //         for (int i = 0; i < WARP_ROW_TENSORS; i++) {
        //             for (int j = 0; j < BLOCK_K_TENSORS; j++) {
        //                 printf("%d ", metadata_a_prefetch[i][j]);
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     __syncthreads();
        // }

        PipelinedSPCoreComputations<TilingConfig>(c, a, b, metadata_a, smem_read_PTR, warp_start_row, warp_start_col);

//        SegmentScan_CUDACoreComputations<TilingConfig>(
//                A2_tile_row_ptr,
//                A2_val,
//                A2_idx,
//                smem_read_PTR + TILE_M * (TILE_K / 2),
//                last_partial_sum,
//                includes_start,
//                C_smem_ptr);
//
//        A2_tile_row_ptr += 1;

        cp_async_wait_group<1>();
//        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet

#ifdef SPARSE_LOAD_A
        SpMM_DecompressFromRegisterToShared_sparse_load<TilingConfig, SparseKernelConfig>(smem_write_PTR,
            Registers_Ind,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1);
#else
        SpMM_DecompressFromRegisterToShared_dense_load<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1,
            smem_write_PTR + TILE_M * TILE_K / 4,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 4,
            NNZ_ThreadLocal2);
#endif
//        __syncthreads();
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory

        __syncthreads();
    }

#pragma unroll(1)
    for (int tile_id_k = NumIter - 1; tile_id_k < NumIter; tile_id_k++) {

        int32_t* __restrict__ sMeta_read_PTR =
            reinterpret_cast<int32_t*>(sMeta + ((tile_id_k) % 2) * WARP_ROW_TENSORS * BLOCK_ROW_WARPS * BLOCK_K_TENSORS * 8);

        // Load meta_data
        Load_Meta_Data<TilingConfig, SparseKernelConfig>(
            sMeta_read_PTR + Warp_i * WARP_ROW_TENSORS * BLOCK_K_TENSORS * METADATA_PER_TENSOR,
            metadata_a, BLOCK_K_TENSORS);

        // double buffer
        half* __restrict__ smem_read_PTR  = smem;
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TILE_M * (TILE_K/2) + TILE_K * TilingConfig::TILE_N);

        PipelinedSPCoreComputations<TilingConfig>(c, a, b, metadata_a, smem_read_PTR, warp_start_row, warp_start_col);
    }

    __syncthreads();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
//    float(*smem_CFrag)[TILE_M + PADDING_SHARED_MEM_FOR_C] =
//        reinterpret_cast<float(*)[TILE_M + PADDING_SHARED_MEM_FOR_C]>(C_smem_ptr);
    half *C_smem_ptr = smem;
    StoreToSharedMemoryFromRegister<TilingConfig>(C_smem_ptr, c);
//    __syncthreads();
//    StoreToSharedMemoryFromRegister_FSP<TilingConfig>(smem_CFrag, C_Val1, C_Val2);
    __syncthreads();
    // Now that shared memory contains all the C tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M * N_Global + Tile_Start_N;

    CopyCTileFromSharedToGlobal<TilingConfig::TILE_N2, TilingConfig>(C_smem_ptr, BlockGlobalPTR, N_Global);
}
