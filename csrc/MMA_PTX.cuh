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
#include "TilingConfig.h"

template<int NumOfTensors>
__device__ __forceinline__ void FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                              half* __restrict__ smem,
                                                              int warp_start_row,
                                                              int k_offset)
{
    //
    int lane_id = threadIdx.x % 32;
    int i       = lane_id % MMA_M;
    int j       = lane_id / MMA_M;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict
    uint32_t RowLane_RowPermutation = i % COPY_UNIT_FP16_ROWS;
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_M * sizeof(half);
    }
}

template<int NumOfTensors>
__device__ __forceinline__ void FragSPLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][2],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{

    //
    int lane_id = threadIdx.x % 32;
    int i       = lane_id % MMA_M;
    //
    smem += (TILE_K / 2) * (warp_start_row) + (k_offset);

    int pack_no = (i) * (TILE_K / 2 / HALF_PER_128B);

    int new_i = pack_no / (HALF_PER_SHMEM_BANK / HALF_PER_128B);
    int new_j = pack_no % (HALF_PER_SHMEM_BANK / HALF_PER_128B);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 16) {
    //     printf("tid: %d, new_i: %d, new_j: %d, XOR_RANGE: %d\n", threadIdx.x, new_i, new_j, (HALF_PER_128B / (HALF_PER_SHMEM_BANK / (TILE_K / 2))));
    // }

    smem += new_i * HALF_PER_SHMEM_BANK + new_j * HALF_PER_128B;

    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Permutation needed for avoiding bank conflicts using ldmatrix
    uint32_t RowLane_RowPermutation = new_i % (HALF_PER_128B / (HALF_PER_SHMEM_BANK / (TILE_K / 2)));
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;

    #pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        //        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        //                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
        //                     : "r"(smem_local_ptr));
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1])
                     : "r"(smem_local_ptr));

        smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
        smem_local_ptr += TILE_K / 2 * MMA_M * sizeof(half);
        smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
    }
}

template<int NumOfTensors, int N8>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    //
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id % 8;
    // uint32_t Mask_RowPermutation = i << 4;

    if (lane_id > 15)
        i += 8;
    int j = (lane_id % 16) >= 8 ? 1 : 0;

    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    // Permutation needed for avoiding bank conflicts using ldmatrix
//   smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // if(N8)
        //  asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        //              : "=r"(Registers[i][0]), "=r"(Registers[i][1])
        //              : "r"(smem_local_ptr));
        // else
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_N * sizeof(half);
    }
}

// B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
//     b_write, SharedMemoryPTR + TILE_M * (TILE_K / 2), warp_start_col, (k + 1) * MMA_K);

template<typename TilingConfig>
__device__ __forceinline__ void B_RowMajFragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_col,
                                                                int k_offset)
{
    constexpr int NumOfTensors = TilingConfig::WARP_COL_TENSORS;
    //
    int      lane_id             = threadIdx.x % 32;
    int i = lane_id % 16;
    int j = (lane_id > 15) ? 8 : 0;

    smem += (warp_start_col) + (k_offset) * TilingConfig::TILE_N;
    
    int pack_no = (i) * (TilingConfig::TILE_N2 / HALF_PER_128B) + (j) / HALF_PER_128B;

    int new_i;
    int new_j;

    if (TilingConfig::N8) {
        new_i = pack_no / (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2);
        new_j = (pack_no % (HALF_PER_SHMEM_BANK / HALF_PER_128B / 2)) * 2;
    }
    else {
        new_i = pack_no / (HALF_PER_SHMEM_BANK / HALF_PER_128B);
        new_j = pack_no % (HALF_PER_SHMEM_BANK / HALF_PER_128B);
    }

    smem += new_i * HALF_PER_SHMEM_BANK + new_j * HALF_PER_128B;
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    uint32_t RowLane_RowPermutation = new_i % (HALF_PER_128B / (HALF_PER_SHMEM_BANK / TilingConfig::TILE_N2));
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;

#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
        smem_local_ptr += MMA_N * sizeof(half);
        smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
    }
}

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),
                   "r"(b[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}

__device__ __forceinline__ void
MMA_SP_FP16_M16N8K16(uint *__restrict__ c, uint *__restrict__ a, uint *__restrict__ b, int *__restrict__ metadata) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4,  %5},"
        "{%6, %7}, %8, 0x0;"
        : "=r"(c[0]),
          "=r"(c[1])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]),
          "r"(c[1]),
          "r"(metadata[0]));
}