//
// Created by Bangtian on 1/10/24.
//

#ifndef FLASH_LLM_SPTC_FSP_COMPUTATION_H
#define FLASH_LLM_SPTC_FSP_COMPUTATION_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include "type_utils.h"


template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

template<int SizeInBytes>
__device__ __forceinline__ void
cp_async(unsigned short* smem_ptr, const unsigned short* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr),
    "l"(global_ptr),
    "n"(SizeInBytes));
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}


// This file is added to explore the implementation for Fine-grained Sparsity on CUDA Cores
int A2_inspection_matrix(
    half* A2, int M, int K, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    printf("A2_inspection_matrix\n");
    assert(M % 32 == 0);
    int* block_length = (int*)malloc(sizeof(int) * (M / 32));
    memset(block_length, 0, sizeof(int) * (M / 32));
    const int __KCUT    = 1;
    const int PADDING   = 8 * __KCUT;
    int       total_nnz = 0;
    for (int i = 0; i < M / 32; i++) {
        int tile_max_length = 0;
        for (int j = 0; j < 32; j++) {
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * 32 * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            total_nnz += row_nnz;
            //            printf("i=%d, j=%d, row_nnz = %d\n", i, j, row_nnz);

            if (row_nnz > tile_max_length) {
                tile_max_length = row_nnz;
            }
        }

        if ((tile_max_length % PADDING) != 0) {
            tile_max_length = tile_max_length + PADDING - (tile_max_length % PADDING);
        }
        block_length[i] = tile_max_length;
        //        printf("i=%d, tile_max_length = %d\n", i, tile_max_length);
    }

    int size = 0;
    for (int i = 0; i < M / 32; i++) {
        size += block_length[i];
    }

    half*           mat_vals = (half*)malloc(sizeof(half) * size * 32);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size * 32);

    printf("size = %d\n", size);
    int offset = 0;
    for (int i = 0; i < M / 32; ++i) {
        //        printf("i=%d, block_length = %d\n", i, block_length[i]);
        int                    tile_max_length = block_length[i];
        vector<half>           tile_vals(tile_max_length * 32);
        vector<unsigned short> tile_cols(tile_max_length * 32);

        for (int j = 0; j < 32; j++) {
            int counter  = 0;
            int last_col = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * 32 * K + j * K + k];
                if (__half2float(val) != 0) {
                    tile_vals[j * tile_max_length + counter] = val;
                    tile_cols[j * tile_max_length + counter] = k;
                    last_col                                 = k;
                    counter++;
                }
            }

            if (counter < tile_max_length) {
                for (int k = counter; k < tile_max_length; k++) {
                    tile_vals[j * tile_max_length + k] = __float2half(0.0);
                    tile_cols[j * tile_max_length + k] = last_col;
                }
            }
        }

        for (int k = 0; k < tile_max_length / 8; k++) {
            for (int j = 0; j < 32; j++) {
                for (int k2 = 0; k2 < 8; k2++) {
                    mat_vals[offset] = tile_vals[j * tile_max_length + k * 8 + k2];
                    mat_cols[offset] = tile_cols[j * tile_max_length + k * 8 + k2];
                    offset++;
                }
            }
        }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M / 32) + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M / 32; i++) {
        matrix_offset[i + 1] = matrix_offset[i] + block_length[i];
    }

    *h_matrix_vals   = mat_vals;
    *h_matrix_cols   = mat_cols;
    *h_matrix_offset = matrix_offset;

    int padd_nnz = matrix_offset[M / 32] * 32;

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    free(block_length);
    return (size * 32);
}

int A2_inspection_matrix_sputnik(
    half* A2, int M, int K, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset, int **h_real_block_length)
{
    const int  TILE_SIZE = 4;
    const int PADDING   = 8 * 8;

    assert(M % TILE_SIZE == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M / TILE_SIZE));
    memset(block_length, 0, sizeof(int) * (M / TILE_SIZE));

    int* real_block_length_ = (int*)malloc(sizeof(int) * (M));

    int total_nnz = 0;

    for (int i = 0; i < M / TILE_SIZE; i++) {
        int tile_max_length = 0;
        for (int j = 0; j < TILE_SIZE; j++) {
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            total_nnz += row_nnz;

            int real_row_nnz = row_nnz;
            if(row_nnz % 8 != 0)
            {
                real_row_nnz = row_nnz + 8 - (row_nnz % 8);
            }

            real_block_length_[i * TILE_SIZE + j] = real_row_nnz;




            if (row_nnz > tile_max_length) {
                tile_max_length = row_nnz;
            }

//            if(i==44)
//            {
//                printf("i=%d, j=%d, row_nnz = %d\n", i, j, row_nnz);
//                getchar();
//            }
            // printf("i=%d, j=%d, row_nnz = %d\n", i, j, row_nnz);
        }

        if ((tile_max_length % PADDING) != 0) {
            tile_max_length = tile_max_length + PADDING - (tile_max_length % PADDING);
        }
        block_length[i] = tile_max_length;
    }

    int size = 0;
    for (int i = 0; i < M / TILE_SIZE; i++) {
        size += block_length[i];
    }

    printf("size = %d\n", size);

    half*           mat_vals = (half*)malloc(sizeof(half) * size * TILE_SIZE);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size * TILE_SIZE);
    int offset = 0;
    for(int i=0; i < M / TILE_SIZE; i++)
    {
       int tile_max_length = block_length[i];
       vector<half> tile_vals(tile_max_length * TILE_SIZE);
       vector<unsigned short> tile_cols(tile_max_length * TILE_SIZE);

       for (int j = 0; j < TILE_SIZE; j++)
       {
            int counter = 0;
            int last_col = 0;
            for (int k = 0; k < K; k++)
            {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0)
                {
                    tile_vals[j * tile_max_length + counter] = val;
                    tile_cols[j * tile_max_length + counter] = k;
                    last_col = k;
                    counter++;
                }
            }

            if (counter < tile_max_length)
            {
                for (int k = counter; k < tile_max_length; k++)
                {
                    tile_vals[j * tile_max_length + k] = __float2half(0.0);
                    tile_cols[j * tile_max_length + k] = last_col;
                }
            }   
       }

       for (int k = 0; k < tile_max_length / (8 * 8); k++) {
           for (int j = 0; j < TILE_SIZE; j++) {
               for (int k2 = 0; k2 < (8 * 8); k2++) {
                   mat_vals[offset] = tile_vals[j * tile_max_length + k * 8 * 8 + k2];
                   mat_cols[offset] = tile_cols[j * tile_max_length + k * 8 * 8 + k2];
                   offset++;
               }
           }
       }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M / TILE_SIZE) + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M / TILE_SIZE; i++) {
        matrix_offset[i + 1] = matrix_offset[i] + block_length[i];
    }

    *h_matrix_vals   = mat_vals;
    *h_matrix_cols   = mat_cols;
    *h_matrix_offset = matrix_offset;
    *h_real_block_length = real_block_length_;

    int padd_nnz = matrix_offset[M / TILE_SIZE] * TILE_SIZE;

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    free(block_length);
    return (size * TILE_SIZE);
}

int A2_inspection_matrix_sputnik_spadding(
        half* A2, int M, int K, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    // In this inspector, I tried to avoid padding overhead.
    const int  TILE_SIZE = 4;
    const int PADDING   = 8;

    assert(M % TILE_SIZE == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M));
    memset(block_length, 0, sizeof(int) * (M));

    int total_nnz = 0;

    for (int i = 0; i < M / TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            int row_id = i * TILE_SIZE + j;
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            total_nnz += row_nnz;

            if(row_nnz % PADDING != 0)
            {
                row_nnz = row_nnz + PADDING - (row_nnz % PADDING);
            }

            block_length[row_id] = row_nnz;
        }
    }

    int size = 0;
    for (int i = 0; i < M; i++) {
        size += block_length[i];
    }

    printf("size = %d\n", size);

    half*           mat_vals = (half*)malloc(sizeof(half) * size);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size);

    int offset = 0;
    for(int i=0; i < M; i++)
    {
        int row_length = block_length[i];
        vector<half> tile_vals(row_length);
        vector<unsigned short> tile_cols(row_length);
        int t_offset = 0;
        int last_col = 0;
        for(int k=0; k<K; k++)
        {
            half val = A2[i * K + k];
            if(__half2float(val) != 0)
            {
                tile_vals[t_offset] = val;
                tile_cols[t_offset] = k;
                last_col = k;
                t_offset++;
            }
        }

        if(t_offset < row_length)
        {
            for(int k=t_offset; k<row_length; k++)
            {
                tile_vals[k] = __float2half(0.0);
                tile_cols[k] = last_col;
            }
        }

        for(int k=0; k<row_length; k++)
        {
            mat_vals[offset] = tile_vals[k];
            mat_cols[offset] = tile_cols[k];
            offset++;
        }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M) + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M; i++) {
        matrix_offset[i + 1] = matrix_offset[i] + block_length[i];
    }

    *h_matrix_vals   = mat_vals;
    *h_matrix_cols   = mat_cols;
    *h_matrix_offset = matrix_offset;

    int padd_nnz = matrix_offset[M];

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    free(block_length);
    return (size);
}

// add data reorganization to avoid uncoalced memory access
int A2_inspection_matrix_sputnik_spadding_sf(
        half *A2, int M, int K, half **h_matrix_vals, unsigned short **h_matrix_cols, u_int8_t **h_min_length,
        int **h_matrix_tile_offset, int **h_matrix_offset) {
    // In this inspector, I tried to avoid padding overhead.
    const int  TILE_SIZE = 4;
    const int PADDING   = 8;

    assert(M % TILE_SIZE == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M));
    memset(block_length, 0, sizeof(int) * (M));
    int total_nnz = 0;

    u_int8_t *block_min_length = (u_int8_t *)malloc(sizeof(u_int8_t) * (M / TILE_SIZE));

    for (int i = 0; i < M / TILE_SIZE; i++) {
        int tile_min_length = INT_MAX;
        for (int j = 0; j < TILE_SIZE; j++) {
            int row_id = i * TILE_SIZE + j;
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            if(row_nnz < tile_min_length)
            {
                tile_min_length = row_nnz;
            }
//            printf("i=%d, j=%d, row_nnz = %d\n", i, j, row_nnz);

            total_nnz += row_nnz;

            if(row_nnz % PADDING != 0)
            {
                row_nnz = row_nnz + PADDING - (row_nnz % PADDING);
            }

            block_length[row_id] = row_nnz;
        }

//        printf("i=%d, tile_min_length = %d %d\n", i, tile_min_length, tile_min_length / (8*8));
        assert(tile_min_length / (8*8) <= 255);
        block_min_length[i] = (u_int8_t)(tile_min_length / (8*8));
//        getchar();
    }


    int size = 0;
    for (int i = 0; i < M; i++) {
        size += block_length[i];
    }

//    printf("size = %d\n", size);

    half*           mat_vals = (half*)malloc(sizeof(half) * size);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size);

    half*          mat_vals_sf = (half*)malloc(sizeof(half) * size);
    unsigned short* mat_cols_sf = (unsigned short*)malloc(sizeof(unsigned short) * size);
    int* row_offset_sf = (int*)malloc(sizeof(int) * M);

    int offset = 0;
    for(int i=0; i < M; i++)
    {
        int row_length = block_length[i];
        vector<half> tile_vals(row_length);
        vector<unsigned short> tile_cols(row_length);
        int t_offset = 0;
        int last_col = 0;
        for(int k=0; k<K; k++)
        {
            half val = A2[i * K + k];
            if(__half2float(val) != 0)
            {
                tile_vals[t_offset] = val;
                tile_cols[t_offset] = k;
                last_col = k;
                t_offset++;
            }
        }

        if(t_offset < row_length)
        {
            for(int k=t_offset; k<row_length; k++)
            {
                tile_vals[k] = __float2half(0.0);
                tile_cols[k] = last_col;
            }
        }

        for(int k=0; k<row_length; k++)
        {
            mat_vals[offset] = tile_vals[k];
            mat_cols[offset] = tile_cols[k];
            offset++;
        }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M) + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M; i++) {
        matrix_offset[i + 1] = matrix_offset[i] + block_length[i];
    }

    offset = 0;
    int *row_block_offset = (int*)malloc(sizeof(int) * (M / TILE_SIZE));
    for(int i=0; i < M/TILE_SIZE; i++)
    {
        row_block_offset[i] = offset;
        int tile_min_length = block_min_length[i];
        for(int j = 0; j < tile_min_length; j++)
        {
            for(int k = 0; k < TILE_SIZE; k++)
            {
                int row_id = i * TILE_SIZE + k;
                int row_offset = matrix_offset[row_id];
                for(int k2 = 0; k2 < (8*8); k2++)
                {
                    mat_vals_sf[offset] = mat_vals[row_offset + j * (8*8) + k2];
                    mat_cols_sf[offset] = mat_cols[row_offset + j * (8*8) + k2];
                    offset++;
                }
            }
        }
    }

    row_offset_sf[0] = offset;
    for(int i=0; i < M; i++)
    {
        int tile_min_length = block_min_length[i / TILE_SIZE];
        int row_length = block_length[i] - tile_min_length * (8*8);
        assert(row_length >= 0);
        for(int j =0; j < row_length; j++)
        {
            mat_vals_sf[offset] = mat_vals[matrix_offset[i] + tile_min_length * (8*8) + j];
            mat_cols_sf[offset] = mat_cols[matrix_offset[i] + tile_min_length * (8*8) + j];
            offset++;
        }
        row_offset_sf[i+1] = row_offset_sf[i] + row_length;
    }

    assert(offset == size);


    *h_matrix_vals   = mat_vals_sf;
    *h_matrix_cols   = mat_cols_sf;
    *h_min_length = block_min_length;
    *h_matrix_offset = row_offset_sf;
    *h_matrix_tile_offset = row_block_offset;

    int padd_nnz = matrix_offset[M];

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    free(block_length);
    free(mat_vals);
    free(mat_cols);
    free(matrix_offset);
    return (size);
}

// try to improve memory locality and make better use of memory hiearchy
int A2_inspection_matrix_opt(
    half* A2, int M, int K, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    // right now, just use sputnik's tiling strategy
    const int   TILE_SIZE = 4;
    vector<int> vector_rnnz;

    const int PADDING = 8 * 8;

    assert(M % TILE_SIZE == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M / TILE_SIZE));
    memset(block_length, 0, sizeof(int) * (M / TILE_SIZE));

    for (int i = 0; i < M / TILE_SIZE; i++) {
        int tile_max_length = 0;
        for (int j = 0; j < TILE_SIZE; j++) {
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            vector_rnnz.push_back(row_nnz);

            if (row_nnz > tile_max_length) {
                tile_max_length = row_nnz;
            }
            // printf("i=%d, j=%d, row_nnz = %d\n", i, j, row_nnz);
        }

        if ((tile_max_length % PADDING) != 0) {
            tile_max_length = tile_max_length + PADDING - (tile_max_length % PADDING);
        }
        block_length[i] = tile_max_length;
    }

    int size = 0;
    for (int i = 0; i < M / TILE_SIZE; i++) {
        size += block_length[i];
    }

    half*           mat_vals = (half*)malloc(sizeof(half) * size * TILE_SIZE);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size * TILE_SIZE);

    printf("size = %d\n", size);

    int col_length = 0;
    for (int i = 0; i < M / TILE_SIZE; i++) {
        int tile_max_length = block_length[i];
        col_length += tile_max_length / (8 * 8);
    }

    int* matrix_row_offset_ucol = (int*)malloc(sizeof(int) * ((M / TILE_SIZE) + 1));
    matrix_row_offset_ucol[0]   = 0;
    for (int i = 0; i < M / TILE_SIZE; i++) {
        matrix_row_offset_ucol[i + 1] = matrix_row_offset_ucol[i] + block_length[i] / (8 * 8);
    }

    int  num_col_tiles          = matrix_row_offset_ucol[M / TILE_SIZE];
    int* matrix_col_offset_ucol = (int*)malloc(sizeof(int) * (num_col_tiles + 1));
    matrix_col_offset_ucol[0]   = 0;

    int                 offset  = 0;
    int                 tile_id = 0;
    vector<vector<int>> uniq_col_vec;
    for (int i = 0; i < M / TILE_SIZE; ++i) {
        int                    tile_max_length = block_length[i];
        vector<half>           tile_vals(tile_max_length * TILE_SIZE);
        vector<unsigned short> tile_cols(tile_max_length * TILE_SIZE);
        vector<int>            uniq_col;
        for (int j = 0; j < TILE_SIZE; j++) {
            int counter  = 0;
            int last_col = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * TILE_SIZE * K + j * K + k];
                if (__half2float(val) != 0) {
                    tile_vals[j * tile_max_length + counter] = val;
                    tile_cols[j * tile_max_length + counter] = k;
                    last_col                                 = k;
                    counter++;
                }
            }

            if (counter < tile_max_length) {
                for (int k = counter; k < tile_max_length; k++) {
                    tile_vals[j * tile_max_length + k] = __float2half(0.0);
                    tile_cols[j * tile_max_length + k] = last_col;
                }
            }
        }

        for (int k = 0; k < tile_max_length / (8 * 8); k++) {
            set<int> col_ind;
            int      data_reuse = 0;
            for (int j = 0; j < TILE_SIZE; j++) {
                for (int k2 = 0; k2 < (8 * 8); k2++) {
                    mat_vals[offset] = tile_vals[j * tile_max_length + k * 8 + k2];
                    mat_cols[offset] = tile_cols[j * tile_max_length + k * 8 + k2];
                    int col          = tile_cols[j * tile_max_length + k * 8 + k2];
                    if (col_ind.count(col) == 0) {
                        col_ind.insert(col);
                    }
                    else {
                        ++data_reuse;
                    }
                    offset++;
                }
            }
            printf("data reuse = %d col_len=%d ratio=%f\n",
                   data_reuse,
                   col_ind.size(),
                   (float)data_reuse / (col_ind.size()));
            matrix_col_offset_ucol[tile_id + 1] = matrix_col_offset_ucol[tile_id] + col_ind.size();
            uniq_col.insert(uniq_col.end(), col_ind.begin(), col_ind.end());
            tile_id++;
        }
    }

    // add a check for unique column indices in each tile
    for (int i = 0; i < M / TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {}
    }

    // int num_min = *std::min_element(vector_rnnz.begin(), vector_rnnz.end());
    // int num_max = *std::max_element(vector_rnnz.begin(), vector_rnnz.end());
    // int num_avg = std::accumulate(vector_rnnz.begin(), vector_rnnz.end(), 0.0) / vector_rnnz.size();

    // printf("min = %d, max = %d, avg = %d\n",num_min, num_max, num_avg);
    // Remove the unnecessary return statement
}

int A2_inspection_matrix_kcut(
    half* A2, int M, int K, int K_cut, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    assert(K % K_cut == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M / 32) * K_cut);
    memset(block_length, 0, sizeof(int) * (M / 32) * K_cut);
    const int PADDING   = 8;
    int       total_nnz = 0;
    for (int i = 0; i < M / 32; i++) {
        for (int k = 0; k < K_cut; k++) {
            int tile_max_length = 0;
            for (int j = 0; j < 32; j++) {
                int row_nnz = 0;
                for (int k2 = 0; k2 < K / K_cut; k2++) {
                    half val = A2[i * 32 * K + j * K + k * K / K_cut + k2];
                    if (__half2float(val) != 0) {
                        row_nnz++;
                    }
                }
                total_nnz += row_nnz;

                if (row_nnz > tile_max_length) {
                    tile_max_length = row_nnz;
                }
            }

            if ((tile_max_length % PADDING) != 0) {
                tile_max_length = tile_max_length + PADDING - (tile_max_length % PADDING);
            }

            block_length[i * K_cut + k] = tile_max_length;
        }
    }

    int size = 0;
    for (int i = 0; i < M / 32; i++) {
        for (int k = 0; k < K_cut; k++) {
            size += block_length[i * K_cut + k];
        }
    }

    half*           mat_vals = (half*)malloc(sizeof(half) * size * 32);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size * 32);

    printf("size = %d\n", size);
    int offset = 0;
    for (int i = 0; i < M / 32; ++i) {
        for (int j = 0; j < K_cut; ++j) {
            int                    tile_max_length = block_length[i * K_cut + j];
            vector<half>           tile_vals(tile_max_length * 32);
            vector<unsigned short> tile_cols(tile_max_length * 32);
            for (int k = 0; k < 32; ++k) {
                int counter  = 0;
                int last_col = 0;
                for (int k2 = 0; k2 < K / K_cut; ++k2) {
                    half val = A2[i * 32 * K + k * K + j * (K / K_cut) + k2];
                    if (__half2float(val) != 0) {
                        tile_vals[k * tile_max_length + counter] = val;
                        tile_cols[k * tile_max_length + counter] = j * (K / K_cut) + k2;
                        last_col                                 = j * (K / K_cut) + k2;
                        counter++;
                    }
                }

                if (counter < tile_max_length) {
                    for (int k2 = counter; k2 < tile_max_length; k2++) {
                        tile_vals[k * tile_max_length + k2] = __float2half(0.0);
                        tile_cols[k * tile_max_length + k2] = last_col;
                    }
                }
            }

            for (int k = 0; k < tile_max_length / 8; ++k) {
                for (int j = 0; j < 32; ++j) {
                    for (int k2 = 0; k2 < 8; ++k2) {
                        mat_vals[offset] = tile_vals[j * tile_max_length + k * 8 + k2];
                        mat_cols[offset] = tile_cols[j * tile_max_length + k * 8 + k2];
                        offset++;
                    }
                }
            }
        }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M / 32) * K_cut + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M / 32; i++) {
        for (int k = 0; k < K_cut; k++) {
            matrix_offset[i * K_cut + k + 1] = matrix_offset[i * K_cut + k] + block_length[i * K_cut + k];
        }
    }

    *h_matrix_vals   = mat_vals;
    *h_matrix_cols   = mat_cols;
    *h_matrix_offset = matrix_offset;

    int padd_nnz = matrix_offset[(M / 32) * K_cut] * 32;

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    printf("original sparsity ratio = %f, padded sparsity ratio = %f\n",
           (float)total_nnz / (M * K),
           (float)padd_nnz / (M * K));
    free(block_length);

    return (size * 32);
}

int A2_inspection_matrix_kcut_v1(
    half* A2, int M, int K, int K_cut, half** h_matrix_vals, unsigned short** h_matrix_cols, int** h_matrix_offset)
{
    assert(K % K_cut == 0);

    int* block_length = (int*)malloc(sizeof(int) * (M / 32));
    memset(block_length, 0, sizeof(int) * (M / 32));
    const int PADDING   = 8 * K_cut;
    int       total_nnz = 0;
    for (int i = 0; i < M / 32; i++) {
        int tile_max_length = 0;
        for (int j = 0; j < 32; j++) {
            int row_nnz = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * 32 * K + j * K + k];
                if (__half2float(val) != 0) {
                    row_nnz++;
                }
            }
            if (row_nnz > tile_max_length) {
                tile_max_length = row_nnz;
            }

            total_nnz += row_nnz;
        }

        if ((tile_max_length % PADDING) != 0) {
            tile_max_length = tile_max_length + PADDING - (tile_max_length % PADDING);
        }

        block_length[i] = tile_max_length;
    }

    int size = 0;
    for (int i = 0; i < M / 32; i++) {
        size += block_length[i];
    }

    half*           mat_vals = (half*)malloc(sizeof(half) * size * 32);
    unsigned short* mat_cols = (unsigned short*)malloc(sizeof(unsigned short) * size * 32);

    printf("size = %d\n", size);
    int offset = 0;
    for (int i = 0; i < M / 32; ++i) {
        int                    tile_max_length = block_length[i];
        vector<half>           tile_vals(tile_max_length * 32);
        vector<unsigned short> tile_cols(tile_max_length * 32);

        for (int j = 0; j < 32; j++) {
            int counter  = 0;
            int last_col = 0;
            for (int k = 0; k < K; k++) {
                half val = A2[i * 32 * K + j * K + k];
                if (__half2float(val) != 0) {
                    tile_vals[j * tile_max_length + counter] = val;
                    tile_cols[j * tile_max_length + counter] = k;
                    last_col                                 = k;
                    counter++;
                }
            }

            if (counter < tile_max_length) {
                for (int k = counter; k < tile_max_length; k++) {
                    tile_vals[j * tile_max_length + k] = __float2half(0.0);
                    tile_cols[j * tile_max_length + k] = last_col;
                }
            }
        }

        for (int k = 0; k < tile_max_length / 8; k++) {
            for (int j = 0; j < 32; j++) {
                for (int k2 = 0; k2 < 8; k2++) {
                    mat_vals[offset] = tile_vals[j * tile_max_length + k * 8 + k2];
                    mat_cols[offset] = tile_cols[j * tile_max_length + k * 8 + k2];
                    offset++;
                }
            }
        }
    }

    int* matrix_offset = (int*)malloc(sizeof(int) * ((M / 32) + 1));
    matrix_offset[0]   = 0;
    for (int i = 0; i < M / 32; i++) {
        matrix_offset[i + 1] = matrix_offset[i] + block_length[i];
    }

    *h_matrix_vals   = mat_vals;
    *h_matrix_cols   = mat_cols;
    *h_matrix_offset = matrix_offset;

    int padd_nnz = matrix_offset[(M / 32)] * 32;

    printf("total_nnz = %d, padd_nnz = %d, ratio=%f\n", total_nnz, padd_nnz, (float)padd_nnz / total_nnz);
    printf("original sparsity ratio = %f, padded sparsity ratio = %f\n",
           (float)total_nnz / (M * K),
           (float)padd_nnz / (M * K));
    free(block_length);

    return (size * 32);
}

__global__ void A2_spmm(const int M,
                        const int N,
                        const int K,
                        const half* __restrict__ d_A2_vals,
                        const unsigned short* __restrict__ d_A2_Idx,
                        const int* __restrict__ d_row_ptr,
                        const half* __restrict__ d_B,
                        half* d_C)
{
    int block_id = blockIdx.x;
    int row_nnz  = d_row_ptr[block_id + 1] - d_row_ptr[block_id];

    int                   len        = row_nnz / 8;
    const half*           A2_vals    = d_A2_vals + d_row_ptr[block_id] * 32 + threadIdx.x * 8;
    const unsigned short* A2_indices = d_A2_Idx + d_row_ptr[block_id] * 32 + threadIdx.x * 8;

    half C_val[16] = {0.0f};

    //    if(threadIdx.x==0 && block_id==0)
    //    {
    //        printf("row_nnz = %d, len=%d\n", row_nnz, len);
    //    }

    for (int i = 0; i < len; i++) {
        uint4 vals  = Load(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 vals    = *(reinterpret_cast<const uint4*>(A2_vals));
        uint4 indices = Load(reinterpret_cast<const uint4*>(A2_indices));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));

        const half*           val_ptr = reinterpret_cast<const half*>(&vals);
        const unsigned short* idx_ptr = reinterpret_cast<const unsigned short*>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];

            //            A2_idx = A2_idx/4 * 2 + A2_idx%4;

            //            if(threadIdx.x==0 && block_id==0)
            //            {
            //                printf("i=%d j=%d A2_val = %f, A2_idx=%d\n", i, j, __half2float(A2_val), A2_idx);
            //            }

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

//            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
//            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));
            uint4 B_vals_1 = Load(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = Load(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }

        A2_vals += 32 * 8;
        A2_indices += 32 * 8;
    }

    half* C_ptr = d_C + block_id * 32 * N;

    uint4 C_vals_1 = *(reinterpret_cast<uint4*>(C_val));
    uint4 C_vals_2 = *(reinterpret_cast<uint4*>(C_val + 8));

    *(reinterpret_cast<uint4*>(&(C_ptr[threadIdx.x * N])))     = C_vals_1;
    *(reinterpret_cast<uint4*>(&(C_ptr[threadIdx.x * N + 8]))) = C_vals_2;

    // #pragma unroll
    //     for(int i=0; i < 16; i++)
    //     {
    //         C_ptr[threadIdx.x * N + i] = C_val[i];
    //     }
}

__global__ void A2_spmm_spadding(const int M,
                        const int N,
                        const int K,
                        const half* __restrict__ d_A2_vals,
                        const unsigned short* __restrict__ d_A2_Idx,
                        const int* __restrict__ d_row_ptr,
                        const half* __restrict__ d_B,
                        half* d_C)
{
    int block_id = blockIdx.x;
    int row_nnz  = d_row_ptr[block_id + 1] - d_row_ptr[block_id];

    int num_wave = (row_nnz -1) / (32 * 8) + 1;

    const half*           A2_vals    = d_A2_vals + d_row_ptr[block_id] + threadIdx.x * 8;
    const unsigned short* A2_indices = d_A2_Idx + d_row_ptr[block_id] + threadIdx.x * 8;

    half C_val[16] = {0.0f};

    for (int i = 0; i < num_wave-1; i++) {
//        half2 vals  = Load(reinterpret_cast<const half2*>(A2_vals));
        uint4 vals    = *(reinterpret_cast<const uint4*>(A2_vals));
//        short2 indices = Load(reinterpret_cast<const short2*>(A2_indices));
        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));

        const half*           val_ptr = reinterpret_cast<const half*>(&vals);
        const unsigned short* idx_ptr = reinterpret_cast<const unsigned short*>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];

//            A2_idx = A2_idx/4 * 2 + A2_idx%4;

            //            if(threadIdx.x==0 && block_id==0)
            //            {
            //                printf("i=%d j=%d A2_val = %f, A2_idx=%d\n", i, j, __half2float(A2_val), A2_idx);
            //            }

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

//            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
//            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));
            uint4 B_vals_1 = Load(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = Load(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }

        A2_vals += 32 * 8;
        A2_indices += 32 * 8;
    }

    int start_idx = (num_wave-1) * 32 * 8;

    if(start_idx + threadIdx.x*8 < row_nnz) {

//        half2 vals = Load(reinterpret_cast<const half2 *>(A2_vals));
        uint4 vals    = *(reinterpret_cast<const uint4*>(A2_vals));
//        short2 indices = Load(reinterpret_cast<const short2 *>(A2_indices));
        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));

        const half *val_ptr = reinterpret_cast<const half *>(&vals);
        const unsigned short *idx_ptr = reinterpret_cast<const unsigned short *>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];
//            A2_idx = A2_idx/4 * 2 + A2_idx%4;

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

//            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
//            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));
            uint4 B_vals_1 = Load(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = Load(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }
    }


    half* C_ptr = d_C + block_id * N;

#pragma unroll
    for(int offset =16; offset > 0; offset /= 2)
    {
#pragma unroll
        for(int i=0; i<16; i++)
        {
            C_val[i] += __shfl_down_sync(0xffffffff, C_val[i], offset);
        }
    }

if(threadIdx.x==0)
    {
        uint4 C_vals_1 = *(reinterpret_cast<uint4*>(C_val));
        uint4 C_vals_2 = *(reinterpret_cast<uint4*>(C_val + 8));
        *(reinterpret_cast<uint4*>(&(C_ptr[threadIdx.x])))     = C_vals_1;
        *(reinterpret_cast<uint4*>(&(C_ptr[threadIdx.x + 8]))) = C_vals_2;
    }
}

__global__ void __launch_bounds__(32, 8) A2_spmm_sputnik(const int M,
                                const int N,
                                const int K,
                                const half* __restrict__ d_A2_vals,
                                const unsigned short* __restrict__ d_A2_Idx,
                                const int* __restrict__ d_row_ptr,
                                const int* __restrict__ d_row_length,
                                const half* __restrict__ d_B,
                                half* d_C)
{
    int block_id = blockIdx.x;
    int row_nnz  = d_row_ptr[block_id + 1] - d_row_ptr[block_id];

    int real_row_length = d_row_length[block_id*4 + threadIdx.y];

    const half *A2_vals = d_A2_vals + d_row_ptr[block_id] * 4 + threadIdx.y * (8*8) + threadIdx.x * 8;
    const unsigned short *A2_indices = d_A2_Idx + d_row_ptr[block_id] * 4 + threadIdx.y * (8*8) + threadIdx.x * 8;

    int len = row_nnz / (8 * 8);

    __shared__ half *A2_tile_vals[4*64];
    __shared__ unsigned short *A2_tile_indices[64*4];

    __align__(16) half2 B_val[64];

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4*>(A2_tile_vals);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4*>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};

    for (int i = 0; i < len; i++)
    {
//        if( i * 8 * 8 + threadIdx.x * 8 >= real_row_length)
//        {
//            break;
//        }
        Store(Load(reinterpret_cast<const uint4*>(A2_vals)), A2_tile_vals_ptr + threadIdx.y * 8 + threadIdx.x);
        Store(Load(reinterpret_cast<const uint4*>(A2_indices)), A2_tile_indices_ptr + threadIdx.y * 8 + threadIdx.x);

//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//        // sparse load from gmem to smem
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;

//        __syncthreads();

        //dense load from gmem to register
//        #pragma unroll
//        for(int k=0; k<64; k++)
//        {
//            unsigned short col_ind= A2_tile_indices[threadIdx.y * 64 + k];
//            half *B_val_ptr = d_B + col_ind * N + threadIdx.x * 2;
//            half2 B_val_tmp = *(reinterpret_cast<const half2*>(B_val_ptr));
//            B_val[k] = B_val_tmp;
//        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint4 col_ind = A2_tile_indices_ptr[threadIdx.y * 8 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = B_val_tmp;
            }
        }

        __syncthreads();

        //compute
        for (int k = 0; k < 8; k++) {
            uint4 vals = A2_tile_vals_ptr[threadIdx.y * 8 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * 8 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
//                d_C[(block_id * 32 + threadIdx.y * 8 + k) * N + threadIdx.x * 2 + k2] += A2_val * B_val_ptr[0];
//                d_C[(block_id * 32 + threadIdx.y * 8 + k) * N + threadIdx.x * 2 + k2 + 1] += A2_val * B_val_ptr[1];
            }
        }

        A2_vals += (8 * 8* 4);
        A2_indices += (8 * 8*4);
        __syncthreads();
    }


    half *C_ptr = d_C + block_id * 4 * N + threadIdx.y * N + threadIdx.x * 2;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));

    return;
}


__global__ void A2_spmm_sputnik_spadding(const int M,
                                                         const int N,
                                                         const int K,
                                                         const half* __restrict__ d_A2_vals,
                                                         const unsigned short* __restrict__ d_A2_Idx,
                                                         const int* __restrict__ d_row_ptr,
                                                         const half* __restrict__ d_B,
                                                         half* d_C)
{
    int block_id = blockIdx.x;
    int row_idx = block_id * 4 + threadIdx.y;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];

    int num_wave = (row_nnz-1) / (8 * 8) + 1;

    const half *A2_vals = d_A2_vals + d_row_ptr[row_idx]  + threadIdx.x * 8;
    const unsigned short *A2_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * 8;
    __shared__ half A2_tile_vals[4*64];
    __shared__ unsigned short A2_tile_indices[64*4];

    __align__(16) half2 B_val[64];

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * 8 * 8 + threadIdx.x * 8);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices + threadIdx.y * 8 * 8 + threadIdx.x * 8);

    uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals);
    uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};


    const uint4 *A2_vals_vec_ptr = reinterpret_cast<const uint4*>(A2_vals);
    const uint4 *A2_indices_vec_ptr = reinterpret_cast<const uint4*>(A2_indices);

//    if(threadIdx.x==0 && threadIdx.y==1 && block_id==0)
//    {
//        printf("row_nnz = %d, num_wave=%d\n", row_nnz, num_wave);
//    }

#pragma unroll
    for(int i=0; i < num_wave-1; i++)
    {
//        if(i>=num_wave-1)
//            break;

        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);

//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;

//        __syncthreads();

#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = B_val_tmp;
            }
        }

//        __syncthreads();

        //compute
#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * 8 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_vals_vec_ptr += (8 );
        A2_indices_vec_ptr += (8);
    }

    uint4 zero_val = {0,0,0,0};
    A2_tile_vals_ptr[0] = zero_val;

    int start_idx = (num_wave-1) * (8*8);

//    if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//    {
//        printf("start_idx = %d\n", start_idx);
//    }
    if(start_idx + threadIdx.x *8 < row_nnz) {
//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;
        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
//        if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//        {
//           half *A2_tile_vals_ptr_tmp = reinterpret_cast<half *>(A2_tile_vals_ptr);
//              printf("A2_tile_vals_ptr_tmp = %f %f %f %f\n", __half2float(A2_tile_vals_ptr_tmp[0]), __half2float(A2_tile_vals_ptr_tmp[1]), __half2float(A2_tile_vals_ptr_tmp[2]), __half2float(A2_tile_vals_ptr_tmp[3]));
//        }
        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);
    }


//
////    __syncthreads();
//
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
        uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
//            half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            B_val[k * 8 + k2] = B_val_tmp;
        }
    }
////
//    __syncthreads();
////    //compute
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
        uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            half A2_val = vals_ptr[k2];

            half2 B_val_tmp = B_val[k * 8 + k2];
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("A2_val = %f\n", __half2float(A2_val));
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
        }
    }
//
//
//
    half *C_ptr = d_C + block_id * 4 * N + threadIdx.y * N + threadIdx.x * 2;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
//    if(block_id == 0 && threadIdx.y == 1 && threadIdx.x == 0)
//    {
//        printf("C_val = %f %f\n", __half2float(C_val[0]), __half2float(C_val[1]));
//    }
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));


    return;
}

//sputnik replication with data reorganization
__global__ void A2_spmm_sputnik_spadding_sf(const int M,
                                         const int N,
                                         const int K,
                                         const half* __restrict__ d_A2_vals,
                                         const unsigned short* __restrict__ d_A2_Idx,
                                         const int* __restrict__ d_tile_ptr,
                                         const int* __restrict__ d_row_ptr,
                                         const u_int8_t * __restrict__ d_min_tile_length,
                                         const half* __restrict__ d_B,
                                         half* d_C)
{
    int block_id = blockIdx.x;
    int row_idx = block_id * 4 + threadIdx.y;

//    const half *A2_vals = d_A2_vals + d_row_ptr[row_idx]  + threadIdx.x * 8;
//    const unsigned short *A2_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * 8;
    __shared__ half A2_tile_vals[4*64];
    __shared__ unsigned short A2_tile_indices[64*4];

    __align__(16) half2 B_val[64];

    uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals);
    uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};


//    const uint4 *A2_vals_vec_ptr = reinterpret_cast<const uint4*>(A2_vals);
//    const uint4 *A2_indices_vec_ptr = reinterpret_cast<const uint4*>(A2_indices);

    u_int8_t min_tile_length = d_min_tile_length[block_id];
    int tile_offset = d_tile_ptr[block_id];
    int lane_id = threadIdx.y * 8 + threadIdx.x;

    const uint4 *A2_gmem_vals_ptr = reinterpret_cast<const uint4 *>(d_A2_vals + tile_offset + lane_id * 8);
    const uint4 *A2_gmem_indices_ptr = reinterpret_cast<const uint4 *>(d_A2_Idx + tile_offset + lane_id * 8);

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * 8 * 8 + threadIdx.x * 8);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices + threadIdx.y * 8 * 8 + threadIdx.x * 8);

    for(u_int8_t i=0; i<min_tile_length; i++) {
        Store(Load(A2_gmem_vals_ptr), A2_tile_vals_ptr);
        Store(Load(A2_gmem_indices_ptr), A2_tile_indices_ptr);
#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * 8 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }
        A2_gmem_vals_ptr += (4 * 8);
        A2_gmem_indices_ptr += (4 * 8);
    }


    uint4 zero_val = {0,0,0,0};
    A2_tile_vals_ptr[0] = zero_val;

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * 8;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * 8;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];
    int num_wave = (row_nnz-1) / (8 * 8) + 1;

    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * 8 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_res_vals += (8 * 8);
        A2_res_indices += (8 * 8);
    }
    int start_idx = (num_wave-1) * (8*8);
    if(start_idx + threadIdx.x *8 < row_nnz) {
//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;
        Store(Load(reinterpret_cast<const uint4*>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4*>(A2_res_indices)), A2_tile_indices_ptr);
//        if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//        {
//           half *A2_tile_vals_ptr_tmp = reinterpret_cast<half *>(A2_tile_vals_ptr);
//              printf("A2_tile_vals_ptr_tmp = %f %f %f %f\n", __half2float(A2_tile_vals_ptr_tmp[0]), __half2float(A2_tile_vals_ptr_tmp[1]), __half2float(A2_tile_vals_ptr_tmp[2]), __half2float(A2_tile_vals_ptr_tmp[3]));
//        }
//        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);
    }


//
////    __syncthreads();
//
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
        uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
//            half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            B_val[k * 8 + k2] = B_val_tmp;
        }
    }
////
//    __syncthreads();
////    //compute
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
        uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            half A2_val = vals_ptr[k2];

            half2 B_val_tmp = B_val[k * 8 + k2];
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);

            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
        }
    }
//
//
//
    half *C_ptr = d_C + block_id * 4 * N + threadIdx.y * N + threadIdx.x * 2;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
//    if(block_id == 0 && threadIdx.y == 1 && threadIdx.x == 0)
//    {
//        printf("C_val = %f %f\n", __half2float(C_val[0]), __half2float(C_val[1]));
//    }
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));


    return;
}

//sputnik replication with data reorganization, our fastest version
__global__ void A2_spmm_sputnik_spadding_sf_ws(const int M,
                                            const int N,
                                            const int K,
                                            const half* __restrict__ d_A2_vals,
                                            const unsigned short* __restrict__ d_A2_Idx,
                                            const int* __restrict__ d_tile_ptr,
                                            const int* __restrict__ d_row_ptr,
                                            const u_int8_t * __restrict__ d_min_tile_length,
                                            const half* __restrict__ d_B,
                                            half* d_C)
{
    //The below code is added with the implicit assumption about the batch size
    const int K_SIZE = 8;
    const int M_SIZE = 4;
    const int NUM_ELEMENTS_PER_THREAD = 8;
    const int NUM_ELEMENTS_PER_32 = 2; // Since our scalar type is FP16, we can load 2 FP16 values


    int block_id = blockIdx.x;
    int row_idx = block_id * M_SIZE + threadIdx.y;
    __shared__ half A2_tile_vals[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];
    __shared__ unsigned short A2_tile_indices[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];

    __align__(16) half2 B_val[K_SIZE*NUM_ELEMENTS_PER_THREAD]; // maybe need to change for different batch size

    uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals);
    uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};

    u_int8_t min_tile_length = d_min_tile_length[block_id];
    int tile_offset = d_tile_ptr[block_id];
    int lane_id = threadIdx.y * K_SIZE + threadIdx.x;

    const uint4 *A2_gmem_vals_ptr = reinterpret_cast<const uint4 *>(d_A2_vals + tile_offset + lane_id * 8);
    const uint4 *A2_gmem_indices_ptr = reinterpret_cast<const uint4 *>(d_A2_Idx + tile_offset + lane_id * 8);

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                                      NUM_ELEMENTS_PER_THREAD) +
                                                        threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices +
                                                           threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                          NUM_ELEMENTS_PER_THREAD) +
                                                           threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    // This for loop is for data-reorganization part, the data fectched by each thread inside the warp are coallesced
    // in the global memory
    for (u_int8_t i = 0; i < min_tile_length; i++) {
        Store(Load(A2_gmem_vals_ptr), A2_tile_vals_ptr);
        Store(Load(A2_gmem_indices_ptr), A2_tile_indices_ptr);
#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * K_SIZE + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }
        A2_gmem_vals_ptr += (M_SIZE * K_SIZE);
        A2_gmem_indices_ptr += (M_SIZE * K_SIZE);
    }

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];
    int num_wave = (row_nnz-1) / (K_SIZE * NUM_ELEMENTS_PER_THREAD) + 1;

    //The following code is for the remaining part of the row, which is not covered by the data-reorganization part
    // Same as sputnik's implementation
    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * NUM_ELEMENTS_PER_THREAD + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_res_vals += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
        A2_res_indices += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    }
    int start_idx = (num_wave - 1) * (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    if (start_idx + threadIdx.x * NUM_ELEMENTS_PER_THREAD < row_nnz) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);
    }

#pragma unroll
    for (int k = 0; k < K_SIZE; k++) {
        if(start_idx + k * NUM_ELEMENTS_PER_THREAD >= row_nnz)
            break;
        uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
            B_val[k * 8 + k2] = B_val_tmp;
        }
    }

////    //compute
#pragma unroll
    for (int k = 0; k < K_SIZE; k++) {
        if (start_idx + k * NUM_ELEMENTS_PER_THREAD >= row_nnz)
            break;
        uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * (K_SIZE + 1) + k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
            half A2_val = vals_ptr[k2];
            half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);

            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
        }
    }
//
//
//
    half *C_ptr = d_C + block_id * M_SIZE * N + threadIdx.y * N + threadIdx.x * NUM_ELEMENTS_PER_32;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));
    return;
}

//sputnik replication with data reorganization, explore some ideas
__global__ void A2_spmm_sputnik_spadding_sf_ws_v1(const int M,
                                               const int N,
                                               const int K,
                                               const half* __restrict__ d_A2_vals,
                                               const unsigned short* __restrict__ d_A2_Idx,
                                               const int* __restrict__ d_tile_ptr,
                                               const int* __restrict__ d_row_ptr,
                                               const u_int8_t * __restrict__ d_min_tile_length,
                                               const half* __restrict__ d_B,
                                               half* d_C)
{
    //The below code is added with the implicit assumption about the batch size
    const int K_SIZE = 8;
    const int M_SIZE = 4;
    const int NUM_ELEMENTS_PER_THREAD = 8;
    const int NUM_ELEMENTS_PER_32 = 2; // Since our scalar type is FP16, we can load 2 FP16 values


    int block_id = blockIdx.x;
    int row_idx = block_id * M_SIZE + threadIdx.y;
    __shared__ half A2_tile_vals[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];
    __shared__ unsigned short A2_tile_indices[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];

    __align__(16) half2 B_val[K_SIZE*NUM_ELEMENTS_PER_THREAD]; // maybe need to change for different batch size

    const uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<const uint4 *>(A2_tile_vals + threadIdx.y * (K_SIZE + 1) *
                                                                                         NUM_ELEMENTS_PER_THREAD);
    const uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<const uint4 *>(A2_tile_indices +
                                                                            threadIdx.y * (K_SIZE + 1) *
                                                                            NUM_ELEMENTS_PER_THREAD);

    __align__(16) half C_val[2] = {0.0f};

    u_int8_t min_tile_length = d_min_tile_length[block_id];
    int tile_offset = d_tile_ptr[block_id];
    int lane_id = threadIdx.y * K_SIZE + threadIdx.x;

    const uint4 *A2_gmem_vals_ptr = reinterpret_cast<const uint4 *>(d_A2_vals + tile_offset + lane_id * 8);
    const uint4 *A2_gmem_indices_ptr = reinterpret_cast<const uint4 *>(d_A2_Idx + tile_offset + lane_id * 8);

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                                      NUM_ELEMENTS_PER_THREAD) +
                                                        threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices +
                                                           threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                          NUM_ELEMENTS_PER_THREAD) +
                                                           threadIdx.x * NUM_ELEMENTS_PER_THREAD);

    const half2 *B_matrix_base_ = reinterpret_cast<const half2 *>(d_B) + threadIdx.x;
    const int rhs_column_ = N * sizeof(half);
    // This for loop is for data-reorganization part, the data fectched by each thread inside the warp are coallesced
    // in the global memory
    for (u_int8_t i = 0; i < min_tile_length; i++) {
        Store(Load(A2_gmem_vals_ptr), A2_tile_vals_ptr);
        Store(Load(A2_gmem_indices_ptr), A2_tile_indices_ptr);
#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 col_ind = A2_tile_indices_smem_ptr[k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
//                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * K_SIZE + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 vals = A2_tile_valls_smem_ptr[k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }
        A2_gmem_vals_ptr += (M_SIZE * K_SIZE);
        A2_gmem_indices_ptr += (M_SIZE * K_SIZE);
    }

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];
    int num_wave = (row_nnz-1) / (K_SIZE * NUM_ELEMENTS_PER_THREAD) + 1;

    //The following code is for the remaining part of the row, which is not covered by the data-reorganization part
    // Same as sputnik's implementation
    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 col_ind = A2_tile_indices_smem_ptr[k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
//                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_THREAD + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 vals = A2_tile_valls_smem_ptr[k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_res_vals += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
        A2_res_indices += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    }
    int start_idx = (num_wave - 1) * (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    if (start_idx + threadIdx.x * NUM_ELEMENTS_PER_THREAD < row_nnz) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);
    }

#pragma unroll
    for (int k = 0; k < K_SIZE; k++) {
        if(start_idx + k * NUM_ELEMENTS_PER_THREAD >= row_nnz)
            break;
        uint4 col_ind = A2_tile_indices_smem_ptr[k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
        uint4 vals = A2_tile_valls_smem_ptr[k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
//            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
            half A2_val = vals_ptr[k2];
            const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
            half2 B_val_tmp = Load(Matrix);
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);

            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
//            B_val[k * 8 + k2] = B_val_tmp;
        }
    }

////    //compute
//#pragma unroll
//    for (int k = 0; k < K_SIZE; k++) {
//        if (start_idx + k * NUM_ELEMENTS_PER_THREAD >= row_nnz)
//            break;
//        uint4 vals = A2_tile_valls_smem_ptr[k];
//        half *vals_ptr = reinterpret_cast<half *>(&vals);
//#pragma unroll
//        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//            half A2_val = vals_ptr[k2];
//            half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
//            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
//
//            C_val[0] += A2_val * B_val_ptr[0];
//            C_val[1] += A2_val * B_val_ptr[1];
//        }
//    }
//
//
//
    half *C_ptr = d_C + block_id * M_SIZE * N + threadIdx.y * N + threadIdx.x * NUM_ELEMENTS_PER_32;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));
    return;
}

//sputnik replication with data reorganization, explore some ideas
// remove data reorganization
template<int _DenseVecSize, typename _DenseVecDataType>
__global__ void A2_spmm_sputnik_spadding_sf_ws_v2(const int M,
                                                  const int N,
                                                  const int K,
                                                  const half* __restrict__ d_A2_vals,
                                                  const unsigned short* __restrict__ d_A2_Idx,
//                                                  const int* __restrict__ d_tile_ptr,
                                                  const int* __restrict__ d_row_ptr,
                                                  const int* __restrict__ d_row_indices,
//                                                  const u_int8_t * __restrict__ d_min_tile_length,
                                                  const half* __restrict__ d_B,
                                                  half* d_C)
{
    //The below code is added with the implicit assumption about the batch size
    const int K_SIZE = 8;
    const int M_SIZE = 4;
    const int NUM_ELEMENTS_PER_THREAD = 8;
    const int NUM_ELEMENTS_PER_32 = 2; // Since our scalar type is FP16, we can load 2 FP16 values


    int block_id = blockIdx.x;
    int row_idx = d_row_indices[block_id * M_SIZE + threadIdx.y];
    __shared__ half A2_tile_vals[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];
    __shared__ unsigned short A2_tile_indices[M_SIZE*(K_SIZE+1)*NUM_ELEMENTS_PER_THREAD];

    __align__(16) _DenseVecDataType B_val[K_SIZE*NUM_ELEMENTS_PER_THREAD]; // maybe need to change for different batch size

    const uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<const uint4 *>(A2_tile_vals + threadIdx.y * (K_SIZE + 1) *
                                                                                         NUM_ELEMENTS_PER_THREAD);
    const uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<const uint4 *>(A2_tile_indices +
                                                                            threadIdx.y * (K_SIZE + 1) *
                                                                            NUM_ELEMENTS_PER_THREAD);

    __align__(16) half C_val[_DenseVecSize] = {0.0f};

    int lane_id = threadIdx.y * K_SIZE + threadIdx.x;

//    const uint4 *A2_gmem_vals_ptr = reinterpret_cast<const uint4 *>(d_A2_vals + tile_offset + lane_id * 8);
//    const uint4 *A2_gmem_indices_ptr = reinterpret_cast<const uint4 *>(d_A2_Idx + tile_offset + lane_id * 8);

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                                      NUM_ELEMENTS_PER_THREAD) +
                                                        threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices +
                                                           threadIdx.y * (K_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                          NUM_ELEMENTS_PER_THREAD) +
                                                           threadIdx.x * NUM_ELEMENTS_PER_THREAD);

    const _DenseVecDataType *B_matrix_base_ = reinterpret_cast<const _DenseVecDataType *>(d_B) + threadIdx.x;
    const int rhs_column_ = N * sizeof(half);

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];
    int num_wave = (row_nnz-1) / (K_SIZE * NUM_ELEMENTS_PER_THREAD) + 1;

    //The following code is for the remaining part of the row, which is not covered by the data-reorganization part
    // Same as sputnik's implementation
    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 col_ind = A2_tile_indices_smem_ptr[k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
//                _DenseVecDataType B_val_tmp = *(reinterpret_cast<const _DenseVecDataType *>(B_val_ptr));
                const _DenseVecDataType *Matrix = OffsetCast<const _DenseVecDataType>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
                _DenseVecDataType B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_THREAD + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SIZE; k++) {
            uint4 vals = A2_tile_valls_smem_ptr[k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
                half A2_val = vals_ptr[k2];
                _DenseVecDataType B_val_tmp = B_val[k * NUM_ELEMENTS_PER_THREAD + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
#pragma unroll
                for (int k3 = 0; k3 < _DenseVecSize; k3++) {
                    C_val[k3] += A2_val * B_val_ptr[k3];
                }
            }
        }

        A2_res_vals += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
        A2_res_indices += (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    }
    int start_idx = (num_wave - 1) * (NUM_ELEMENTS_PER_THREAD * K_SIZE);
    if (start_idx + threadIdx.x * NUM_ELEMENTS_PER_THREAD < row_nnz) {
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const uint4 *>(A2_res_indices)), A2_tile_indices_ptr);
    }

#pragma unroll
    for (int k = 0; k < K_SIZE; k++) {
        if(start_idx + k * NUM_ELEMENTS_PER_THREAD >= row_nnz)
            break;
        uint4 col_ind = A2_tile_indices_smem_ptr[k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
        uint4 vals = A2_tile_valls_smem_ptr[k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_THREAD; k2++) {
//            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
//            _DenseVecDataType B_val_tmp = Load(reinterpret_cast<const _DenseVecDataType *>(B_val_ptr));
            half A2_val = vals_ptr[k2];
            const _DenseVecDataType *Matrix = OffsetCast<const _DenseVecDataType>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
            _DenseVecDataType B_val_tmp = Load(Matrix);
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);

#pragma unroll
            for (int k3 = 0; k3 < _DenseVecSize; k3++) {
                C_val[k3] += A2_val * B_val_ptr[k3];
            }
        }
    }

    // Accumulate the results from the shared memory
    _DenseVecDataType global_C; // = Load(reinterpret_cast<_DenseVecDataType *>(d_C + (row_idx) * N + threadIdx.x * _DenseVecSize));
    half *C_ptr = reinterpret_cast<half *>(&global_C);
    for (int i = 0; i < _DenseVecSize; i++) {
        C_ptr[i] = C_val[i];
    }
    Store(global_C, reinterpret_cast<_DenseVecDataType *>(d_C + (row_idx) * N + threadIdx.x * _DenseVecSize));
}

//sputnik replication with data reorganization for exploring register usage
__global__ void __launch_bounds__(32) A2_spmm_sputnik_spadding_sf_regopt(const int M,
                                               const int N,
                                               const int K,
                                               const half* __restrict__ d_A2_vals,
                                               const unsigned short* __restrict__ d_A2_Idx,
                                               const int* __restrict__ d_tile_ptr,
                                               const int* __restrict__ d_row_ptr,
                                               const u_int8_t * __restrict__ d_min_tile_length,
                                               const half* __restrict__ d_B,
                                               half* d_C)
{
    //The below code is added with the implicit assumption about the batch size
    const int K_VEC_SIZE = 8;
    const int K_SCALAR_SIZE = 32;

    const int M_SIZE = 4;
    const int NUM_ELEMENTS_PER_THREAD = 8;
    const int NUM_ELEMENTS_PER_32 = 2; // Since our scalar type is FP16, we can load 2 FP16 values


    int block_id = blockIdx.x;
    int row_idx = block_id * M_SIZE + threadIdx.y;
    __shared__ half A2_tile_vals[M_SIZE*(K_VEC_SIZE+1)*NUM_ELEMENTS_PER_THREAD];
    __shared__ unsigned short A2_tile_indices[M_SIZE*(K_VEC_SIZE+1)*NUM_ELEMENTS_PER_THREAD];

    __align__(16) half2 B_val[K_VEC_SIZE*NUM_ELEMENTS_PER_THREAD]; // maybe need to change for different batch size

    half2 *A2_tile_valls_smem_ptr = reinterpret_cast<half2 *>(A2_tile_vals);
    short2 *A2_tile_indices_smem_ptr = reinterpret_cast<short2 *>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};
    float2 C_val_FP32;
    C_val_FP32.x = 0.0f;
    C_val_FP32.y = 0.0f;

    u_int8_t min_tile_length = d_min_tile_length[block_id];

    int tile_offset = d_tile_ptr[block_id];
    int lane_id = threadIdx.y * K_VEC_SIZE + threadIdx.x;

    const half8 *A2_gmem_vals_ptr = reinterpret_cast<const half8 *>(d_A2_vals + tile_offset + lane_id * 8);
    const short8 *A2_gmem_indices_ptr = reinterpret_cast<const short8 *>(d_A2_Idx + tile_offset + lane_id * 8);

    half8 *A2_tile_vals_ptr = reinterpret_cast<half8 *>(A2_tile_vals + threadIdx.y * (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                                      NUM_ELEMENTS_PER_THREAD) +
                                                        threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    short8 *A2_tile_indices_ptr = reinterpret_cast<short8 *>(A2_tile_indices +
                                                           threadIdx.y * (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                          NUM_ELEMENTS_PER_THREAD) +
                                                           threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    // This for loop is for data-reorganization part, the data fectched by each thread inside the warp are coallesced
    // in the global memory
    const half2 *B_matrix_base_ = reinterpret_cast<const half2 *>(d_B) + threadIdx.x;
    const int rhs_column_ = N * sizeof(half);
//    short2 *column_indices_tile = A2_tile_indices_smem_ptr + threadIdx.y * (K_VEC_SIZE + 1);
//    half2 *values_tile = A2_tile_valls_smem_ptr + threadIdx.y * (K_VEC_SIZE + 1);
    const short2 *column_indices_tile = MaybeOffset(A2_tile_indices_smem_ptr, threadIdx.y * (K_VEC_SIZE + 1) * 4);
    const half2 *values_tile = MaybeOffset(A2_tile_valls_smem_ptr, threadIdx.y * (K_VEC_SIZE + 1) * 4);
//    if(non_zeros > 64)
    for(u_int8_t i=0; i<min_tile_length; i++)
    {
        Store(Load(A2_gmem_vals_ptr), A2_tile_vals_ptr);
        Store(Load(A2_gmem_indices_ptr), A2_tile_indices_ptr);
#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            int scaled_indices[2];
            FSP_Convert(column_indices_tile+k, scaled_indices);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                scaled_indices[k2] *= rhs_column_;
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_32 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            const half2 *vals = values_tile + k;
            float lhs_values[2];
            FSP_Convert(vals, lhs_values);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_32 + k2];
                FMA(lhs_values[k2], B_val_tmp, &C_val_FP32);
            }
        }
        A2_gmem_vals_ptr += (M_SIZE * K_VEC_SIZE);
        A2_gmem_indices_ptr += (M_SIZE * K_VEC_SIZE);
    }

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];

    int num_wave = (row_nnz-1) / (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD) + 1;

//    const float ZeroValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
//    const int kZerosIndices[4] = {0, 0, 0, 0};
////    __syncthreads();
//    Store(*reinterpret_cast<const half8 *>(ZeroValues), A2_tile_vals_ptr);
//    Store(*reinterpret_cast<const short8 *>(kZerosIndices), A2_tile_indices_ptr);
    //The following code is for the remaining part of the row, which is not covered by the data-reorganization part
    // Same as sputnik's implementation
    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const half8 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const short8 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            int scaled_indices[2];
            FSP_Convert(column_indices_tile+k, scaled_indices);
//            short2 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * (K_SCALAR_SIZE + 4) + k];
//            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                scaled_indices[k2] *= rhs_column_;
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_32 + k2] = B_val_tmp;
//                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * NUM_ELEMENTS_PER_32;
//                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
//                B_val[k * NUM_ELEMENTS_PER_32 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            const half2 *vals = values_tile + k;
            float lhs_values[2];
            FSP_Convert(vals, lhs_values);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_32 + k2];
                FMA(lhs_values[k2], B_val_tmp, &C_val_FP32);
            }
        }

        A2_res_vals += (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
        A2_res_indices += (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
    }

    int start_idx = (num_wave - 1) * (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
    if (start_idx + threadIdx.x * NUM_ELEMENTS_PER_THREAD < row_nnz) {
        Store(Load(reinterpret_cast<const half8 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const short8 *>(A2_res_indices)), A2_tile_indices_ptr);
    }

//    __syncthreads();
#pragma unroll
    for (int k = 0; k < K_SCALAR_SIZE; k++) {
        if(start_idx + k * NUM_ELEMENTS_PER_32 >= row_nnz)
            break;
        int scaled_indices[2];
        float lhs_values[2];
        FSP_Convert(column_indices_tile+k, scaled_indices);
        FSP_Convert(values_tile+k, lhs_values);

#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
            scaled_indices[k2] *= rhs_column_;

            const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
            half2 B_val_tmp = Load(Matrix);

            FMA(lhs_values[k2], B_val_tmp, &C_val_FP32);


        }
    }

//
    half *C_ptr = d_C + block_id * M_SIZE * N + threadIdx.y * N + threadIdx.x * NUM_ELEMENTS_PER_32;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
    half2 *C_Val2 = reinterpret_cast<half2 *>(C_val);
    C_Val2[0] = __float22half2_rn(C_val_FP32);

    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));
//    return;
}

//sputnik replication with data reorganization for exploring register usage
__global__ void __launch_bounds__(32) A2_spmm_sputnik_spadding_sf_regopt_v1(const int M,
                                                                         const int N,
                                                                         const int K,
                                                                         const half* __restrict__ d_A2_vals,
                                                                         const unsigned short* __restrict__ d_A2_Idx,
                                                                         const int* __restrict__ d_tile_ptr,
                                                                         const int* __restrict__ d_row_ptr,
                                                                         const u_int8_t * __restrict__ d_min_tile_length,
                                                                         const half* __restrict__ d_B,
                                                                         half* d_C)
{
    //The below code is added with the implicit assumption about the batch size
    const int K_VEC_SIZE = 8;
    const int K_SCALAR_SIZE = 32;

    const int M_SIZE = 4;
    const int NUM_ELEMENTS_PER_THREAD = 8;
    const int NUM_ELEMENTS_PER_32 = 2; // Since our scalar type is FP16, we can load 2 FP16 values


    int block_id = blockIdx.x;
    int row_idx = block_id * M_SIZE + threadIdx.y;
    __shared__ half A2_tile_vals[M_SIZE*(K_VEC_SIZE+1)*NUM_ELEMENTS_PER_THREAD];
    __shared__ unsigned short A2_tile_indices[M_SIZE*(K_VEC_SIZE+1)*NUM_ELEMENTS_PER_THREAD];

    __align__(16) half2 B_val[K_VEC_SIZE*NUM_ELEMENTS_PER_THREAD]; // maybe need to change for different batch size

    half2 *A2_tile_valls_smem_ptr = reinterpret_cast<half2 *>(A2_tile_vals);
    short2 *A2_tile_indices_smem_ptr = reinterpret_cast<short2 *>(A2_tile_indices);

    __align__(16) half C_val[2] = {0.0f};
    float2 C_val_FP32;
    C_val_FP32.x = 0.0f;
    C_val_FP32.y = 0.0f;

    u_int8_t min_tile_length = d_min_tile_length[block_id];

    int tile_offset = d_tile_ptr[block_id];
    int lane_id = threadIdx.y * K_VEC_SIZE + threadIdx.x;

    const half8 *A2_gmem_vals_ptr = reinterpret_cast<const half8 *>(d_A2_vals + tile_offset + lane_id * 8);
    const short8 *A2_gmem_indices_ptr = reinterpret_cast<const short8 *>(d_A2_Idx + tile_offset + lane_id * 8);

    half8 *A2_tile_vals_ptr = reinterpret_cast<half8 *>(A2_tile_vals + threadIdx.y * (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                                      NUM_ELEMENTS_PER_THREAD) +
                                                        threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    short8 *A2_tile_indices_ptr = reinterpret_cast<short8 *>(A2_tile_indices +
                                                             threadIdx.y * (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD +
                                                                            NUM_ELEMENTS_PER_THREAD) +
                                                             threadIdx.x * NUM_ELEMENTS_PER_THREAD);
    // This for loop is for data-reorganization part, the data fectched by each thread inside the warp are coallesced
    // in the global memory
    const half2 *B_matrix_base_ = reinterpret_cast<const half2 *>(d_B) + threadIdx.x;
    const int rhs_column_ = N * sizeof(half);
//    short2 *column_indices_tile = A2_tile_indices_smem_ptr + threadIdx.y * (K_VEC_SIZE + 1);
//    half2 *values_tile = A2_tile_valls_smem_ptr + threadIdx.y * (K_VEC_SIZE + 1);
    const short2 *column_indices_tile = MaybeOffset(A2_tile_indices_smem_ptr, threadIdx.y * (K_VEC_SIZE + 1) * 4);
    const half2 *values_tile = MaybeOffset(A2_tile_valls_smem_ptr, threadIdx.y * (K_VEC_SIZE + 1) * 4);
//    if(non_zeros > 64)
    for(u_int8_t i=0; i<min_tile_length; i++)
    {
        Store(Load(A2_gmem_vals_ptr), A2_tile_vals_ptr);
        Store(Load(A2_gmem_indices_ptr), A2_tile_indices_ptr);
#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
//            int scaled_indices[2];
//            FSP_Convert(column_indices_tile+k, scaled_indices);
            short2 col_ind = column_indices_tile[k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
//                scaled_indices[k2] *= rhs_column_;
//                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_32 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            const half2 *vals = values_tile + k;
//            float lhs_values[2];
//            FSP_Convert(vals, lhs_values);
            const half *vals_ptr = reinterpret_cast<const half *>(vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_32 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                half A2_val = vals_ptr[k2];
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
//                FMA(lhs_values[k2], B_val_tmp, &C_val_FP32);
            }
        }
        A2_gmem_vals_ptr += (M_SIZE * K_VEC_SIZE);
        A2_gmem_indices_ptr += (M_SIZE * K_VEC_SIZE);
    }

    const half *A2_res_vals = d_A2_vals + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    const unsigned short *A2_res_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * NUM_ELEMENTS_PER_THREAD;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];

    int num_wave = (row_nnz-1) / (K_VEC_SIZE * NUM_ELEMENTS_PER_THREAD) + 1;

//    const float ZeroValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
//    const int kZerosIndices[4] = {0, 0, 0, 0};
////    __syncthreads();
//    Store(*reinterpret_cast<const half8 *>(ZeroValues), A2_tile_vals_ptr);
//    Store(*reinterpret_cast<const short8 *>(kZerosIndices), A2_tile_indices_ptr);
    //The following code is for the remaining part of the row, which is not covered by the data-reorganization part
    // Same as sputnik's implementation
    for(int i=0; i < num_wave-1; i++) {
        Store(Load(reinterpret_cast<const half8 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const short8 *>(A2_res_indices)), A2_tile_indices_ptr);

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
//            int scaled_indices[2];
//            FSP_Convert(column_indices_tile+k, scaled_indices);
            short2 col_ind = column_indices_tile[k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
//                scaled_indices[k2] *= rhs_column_;
//                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
                const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
                half2 B_val_tmp = Load(Matrix);
                B_val[k * NUM_ELEMENTS_PER_32 + k2] = B_val_tmp;
            }
        }

#pragma unroll
        for (int k = 0; k < K_SCALAR_SIZE; k++) {
            const half2 *vals = values_tile + k;
//            float lhs_values[2];
//            FSP_Convert(vals, lhs_values);
            const half *vals_ptr = reinterpret_cast<const half *>(vals);
#pragma unroll
            for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
                half2 B_val_tmp = B_val[k * NUM_ELEMENTS_PER_32 + k2];
//                FMA(lhs_values[k2], B_val_tmp, &C_val_FP32);
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                half A2_val = vals_ptr[k2];
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_res_vals += (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
        A2_res_indices += (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
    }

    int start_idx = (num_wave - 1) * (NUM_ELEMENTS_PER_THREAD * K_VEC_SIZE);
    if (start_idx + threadIdx.x * NUM_ELEMENTS_PER_THREAD < row_nnz) {
        Store(Load(reinterpret_cast<const half8 *>(A2_res_vals)), A2_tile_vals_ptr);
        Store(Load(reinterpret_cast<const short8 *>(A2_res_indices)), A2_tile_indices_ptr);
    }

//    __syncthreads();
#pragma unroll
    for (int k = 0; k < K_SCALAR_SIZE; k++) {
        if(start_idx + k * NUM_ELEMENTS_PER_32 >= row_nnz)
            break;
//        int scaled_indices[2];
//        float lhs_values[2];
//        FSP_Convert(column_indices_tile+k, scaled_indices);
//        FSP_Convert(values_tile+k, lhs_values);
        short2 col_ind = column_indices_tile[k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
        const half2 *vals = values_tile + k;
        const half *vals_ptr = reinterpret_cast<const half *>(vals);
#pragma unroll
        for (int k2 = 0; k2 < NUM_ELEMENTS_PER_32; k2++) {
//            scaled_indices[k2] *= rhs_column_;

//            const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, scaled_indices[k2]);
            const half2 *Matrix = OffsetCast<const half2>(B_matrix_base_, col_ind_ptr[k2] * rhs_column_);
            half2 B_val_tmp = Load(Matrix);
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
            half A2_val = vals_ptr[k2];
            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
        }
    }

//
    half *C_ptr = d_C + block_id * M_SIZE * N + threadIdx.y * N + threadIdx.x * NUM_ELEMENTS_PER_32;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
//    half2 *C_Val2 = reinterpret_cast<half2 *>(C_val);
//    C_Val2[0] = __float22half2_rn(C_val_FP32);

    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));
//    return;
}

__global__ void A2_spmm_sputnik_spadding_async(const int M,
                                         const int N,
                                         const int K,
                                         const half* __restrict__ d_A2_vals,
                                         const unsigned short* __restrict__ d_A2_Idx,
                                         const int* __restrict__ d_row_ptr,
                                         const half* __restrict__ d_B,
                                         half* d_C)
{
    int block_id = blockIdx.x;
    int row_idx = block_id * 4 + threadIdx.y;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];

    int num_wave = (row_nnz-1) / (8 * 8) + 1;

    const half *A2_vals = d_A2_vals + d_row_ptr[row_idx]  + threadIdx.x * 8;
    const unsigned short *A2_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * 8;
    __shared__ half A2_tile_vals[4*64*2];
    __shared__ unsigned short A2_tile_indices[64*4*2];

    __align__(16) half2 B_val[64];

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * 8 * 8 + threadIdx.x * 8);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices + threadIdx.y * 8 * 8 + threadIdx.x * 8);

    __align__(16) half C_val[2] = {0.0f};

    uint4 zero_val = {0,0,0,0};
    A2_tile_vals_ptr[0] = zero_val;

    const uint4 *A2_vals_vec_ptr = reinterpret_cast<const uint4*>(A2_vals);
    const uint4 *A2_indices_vec_ptr = reinterpret_cast<const uint4*>(A2_indices);

    bool can_copy = threadIdx.x * 8 < row_nnz;

//    if(block_id==0 && threadIdx.y==0 && threadIdx.x==0)
//    {
//        printf("row_nnz = %d, num_wave=%d\n", row_nnz, num_wave);
//    }

    cp_async<16>(A2_tile_indices + threadIdx.y * 8 * 8 + threadIdx.x * 8, A2_indices, can_copy);
    cp_async_group_commit();

    cp_async<16>(A2_tile_vals + threadIdx.y * 8 * 8 + threadIdx.x * 8, A2_vals, can_copy);
    cp_async_group_commit();

//    if(can_copy)
//    {
//        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
//        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);
//    }

#pragma unroll
    for(int i=0; i < num_wave-1; i++)
    {
//        if(i>=num_wave-1)
//            break;
        bool can_copy =( (i + 1) < num_wave - 1);


//        half *__restrict__ A2_tile_vals_read_ptr = A2_tile_vals;
//        unsigned short *__restrict__ A2_tile_indices_read_ptr = A2_tile_indices;
//
//        half *__restrict__ A2_tile_vals_write_ptr = A2_tile_vals ;
//        unsigned short *__restrict__ A2_tile_indices_write_ptr = A2_tile_indices;

        half *A2_tile_vals_read_ptr = A2_tile_vals + (i%2) * 4 * 64  + threadIdx.y * 8 * 8 + threadIdx.x * 8;
        unsigned short *A2_tile_indices_read_ptr = A2_tile_indices + (i%2) * 4 * 64  + threadIdx.y * 8 * 8 + threadIdx.x * 8;

        half *A2_tile_vals_write_ptr = A2_tile_vals + ((i+1)%2) * 4 * 64  + threadIdx.y * 8 * 8 + threadIdx.x * 8;
        unsigned short *A2_tile_indices_write_ptr = A2_tile_indices + ((i+1)%2) * 4 * 64  + threadIdx.y * 8 * 8 + threadIdx.x * 8;

        int start_idx = (num_wave - 1) * (8 * 8);

//        if(can_copy || ((threadIdx.x * 8 + start_idx < row_nnz) && i == (num_wave-2)))
//        {
//            Store(Load(A2_indices_vec_ptr + 8), reinterpret_cast<uint4*>(A2_tile_indices_write_ptr));
//            Store(Load(A2_vals_vec_ptr + 8), reinterpret_cast<uint4*>(A2_tile_vals_write_ptr));
//        }

        can_copy = can_copy || ((threadIdx.x * 8 + start_idx < row_nnz) && i == (num_wave - 2));

        cp_async < 16 > (A2_tile_indices_write_ptr, A2_indices + 8 * 8, can_copy);
        cp_async_group_commit();

        cp_async < 16 > (A2_tile_vals_write_ptr, A2_vals + 8 * 8, can_copy);
        cp_async_group_commit();

//        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
//        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);

//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;

//        __syncthreads();
        cp_async_wait_group<0>();
        __syncthreads();

//        cp_async_wait_group<1>();
        uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals+ (i%2) * 4 * 64);
        uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices + (i%2) * 4 * 64);

#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = B_val_tmp;
            }
        }

//        __syncthreads();

        //compute
#pragma unroll
        for (int k = 0; k < 8; k++) {
//            uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half2 B_val_tmp = B_val[k * 8 + k2];
                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val[0] += A2_val * B_val_ptr[0];
                C_val[1] += A2_val * B_val_ptr[1];
            }
        }
//
//        A2_vals_vec_ptr += (8 );
//        A2_indices_vec_ptr += (8);

        A2_vals += (8 * 8);
        A2_indices += (8 * 8);
    }

//    uint4 zero_val = {0,0,0,0};
//    A2_tile_vals_ptr[0] = zero_val;
//
//
//    cp_async_wait_group<1>();
//    __syncthreads();
    int start_idx = (num_wave-1) * (8*8);
//
////    if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
////    {
////        printf("start_idx = %d\n", start_idx);
////    }
//    if(start_idx + threadIdx.x *8 < row_nnz) {
////        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
////        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
////        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
////        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;
//        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
////        if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
////        {
////           half *A2_tile_vals_ptr_tmp = reinterpret_cast<half *>(A2_tile_vals_ptr);
////              printf("A2_tile_vals_ptr_tmp = %f %f %f %f\n", __half2float(A2_tile_vals_ptr_tmp[0]), __half2float(A2_tile_vals_ptr_tmp[1]), __half2float(A2_tile_vals_ptr_tmp[2]), __half2float(A2_tile_vals_ptr_tmp[3]));
////        }
//        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);
//    }
//
//
////
//////    __syncthreads();
////
    uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals+ ((num_wave-1)%2) * 4 * 64);
    uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices+ ((num_wave-1)%2) * 4 * 64);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
        uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 8 + k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x * 2;
//            half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            B_val[k * 8 + k2] = B_val_tmp;
        }
    }
//////
////    __syncthreads();
//////    //compute
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
        uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 8 + k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            half A2_val = vals_ptr[k2];

            half2 B_val_tmp = B_val[k * 8 + k2];
            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
//            if(block_id==0 && threadIdx.y==0 && threadIdx.x==0)
//            {
//                printf("A2_val = %f\n", __half2float(A2_val));
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            C_val[0] += A2_val * B_val_ptr[0];
            C_val[1] += A2_val * B_val_ptr[1];
        }
    }
//
//
//
    half *C_ptr = d_C + block_id * 4 * N + threadIdx.y * N + threadIdx.x * 2;
    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
//    if(block_id == 0 && threadIdx.y == 1 && threadIdx.x == 0)
//    {
//        printf("C_val = %f %f\n", __half2float(C_val[0]), __half2float(C_val[1]));
//    }
    C_ptr_2[0] = *(reinterpret_cast<half2 *>(C_val));


    return;
}

__global__ void A2_spmm_sputnik_spadding_2_16(const int M,
                                         const int N,
                                         const int K,
                                         const half* __restrict__ d_A2_vals,
                                         const unsigned short* __restrict__ d_A2_Idx,
                                         const int* __restrict__ d_row_ptr,
                                         const half* __restrict__ d_B,
                                         half* d_C)
{
    int block_id = blockIdx.x;
    int row_idx = block_id * 2 + threadIdx.y;
    int row_nnz  = d_row_ptr[row_idx+1] - d_row_ptr[row_idx];

    int num_wave = (row_nnz-1) / (16 * 8) + 1;

    const half *A2_vals = d_A2_vals + d_row_ptr[row_idx]  + threadIdx.x * 8;
    const unsigned short *A2_indices = d_A2_Idx + d_row_ptr[row_idx] + threadIdx.x * 8;
    __shared__ half A2_tile_vals[2*128];
    __shared__ unsigned short A2_tile_indices[64*4];

    __align__(16) half B_val[128];

    uint4 *A2_tile_vals_ptr = reinterpret_cast<uint4 *>(A2_tile_vals + threadIdx.y * 16 * 8 + threadIdx.x * 8);
    uint4 *A2_tile_indices_ptr = reinterpret_cast<uint4 *>(A2_tile_indices + threadIdx.y * 16 * 8 + threadIdx.x * 8);

    uint4 *A2_tile_valls_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_vals);
    uint4 *A2_tile_indices_smem_ptr = reinterpret_cast<uint4 *>(A2_tile_indices);

    __align__(16) half C_val = __float2half(0.0f);


    const uint4 *A2_vals_vec_ptr = reinterpret_cast<const uint4*>(A2_vals);
    const uint4 *A2_indices_vec_ptr = reinterpret_cast<const uint4*>(A2_indices);

//    if(threadIdx.x==0 && threadIdx.y==1 && block_id==0)
//    {
//        printf("row_nnz = %d, num_wave=%d\n", row_nnz, num_wave);
//    }

#pragma unroll
    for(int i=0; i < num_wave-1; i++)
    {
//        if(i>=num_wave-1)
//            break;

        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);

//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;

//        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; k++) {
//            uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
            uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 16 + k];
            unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x;
//                half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
                B_val[k * 8 + k2] = *B_val_ptr;
            }
        }

//        __syncthreads();

        //compute
#pragma unroll
        for (int k = 0; k < 16; k++) {
//            uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
            uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 16 + k];
            half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
            for (int k2 = 0; k2 < 8; k2++) {
                half A2_val = vals_ptr[k2];
                half B_val_tmp = B_val[k * 8 + k2];
//                half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
                C_val += A2_val * B_val_tmp;
//                C_val[1] += A2_val * B_val_ptr[1];
            }
        }

        A2_vals_vec_ptr += (16 );
        A2_indices_vec_ptr += (16);
    }

    uint4 zero_val = {0,0,0,0};
    A2_tile_vals_ptr[0] = zero_val;

    int start_idx = (num_wave-1) * (16*8);

//    if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//    {
//        printf("start_idx = %d\n", start_idx);
//    }
    if(start_idx + threadIdx.x *8 < row_nnz) {
//        uint4 vals = *(reinterpret_cast<const uint4*>(A2_vals));
//        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));
//        A2_tile_vals_ptr[threadIdx.y * 8 + threadIdx.x] = vals;
//        A2_tile_indices_ptr[threadIdx.y * 8 + threadIdx.x] = indices;
        Store(Load(A2_vals_vec_ptr), A2_tile_vals_ptr);
//        if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//        {
//           half *A2_tile_vals_ptr_tmp = reinterpret_cast<half *>(A2_tile_vals_ptr);
//              printf("A2_tile_vals_ptr_tmp = %f %f %f %f\n", __half2float(A2_tile_vals_ptr_tmp[0]), __half2float(A2_tile_vals_ptr_tmp[1]), __half2float(A2_tile_vals_ptr_tmp[2]), __half2float(A2_tile_vals_ptr_tmp[3]));
//        }
        Store(Load(A2_indices_vec_ptr), A2_tile_indices_ptr);
    }


//
////    __syncthreads();
//
#pragma unroll
    for (int k = 0; k < 16; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 col_ind = Load(A2_tile_indices_smem_ptr + threadIdx.y * 8 + k);
        uint4 col_ind = A2_tile_indices_smem_ptr[threadIdx.y * 16 + k];
        unsigned short *col_ind_ptr = reinterpret_cast<unsigned short *>(&col_ind);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            const half *B_val_ptr = d_B + col_ind_ptr[k2] * N + threadIdx.x;
//            half2 B_val_tmp = *(reinterpret_cast<const half2 *>(B_val_ptr));
//            half2 B_val_tmp = Load(reinterpret_cast<const half2 *>(B_val_ptr));
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            B_val[k * 8 + k2] = *B_val_ptr;
        }
    }
////
//    __syncthreads();
////    //compute
#pragma unroll
    for (int k = 0; k < 16; k++) {
        if(start_idx + k *8 >= row_nnz)
            break;
//        uint4 vals = Load(A2_tile_valls_smem_ptr + threadIdx.y * 8 + k);
        uint4 vals = A2_tile_valls_smem_ptr[threadIdx.y * 16 + k];
        half *vals_ptr = reinterpret_cast<half *>(&vals);
#pragma unroll
        for (int k2 = 0; k2 < 8; k2++) {
            half A2_val = vals_ptr[k2];

            half B_val_tmp = B_val[k * 8 + k2];
//            half *B_val_ptr = reinterpret_cast<half *>(&B_val_tmp);
//            if(block_id==0 && threadIdx.y==1 && threadIdx.x==0)
//            {
//                printf("A2_val = %f\n", __half2float(A2_val));
//                printf("B_val_tmp = %f %f\n", __half2float(B_val_tmp.x), __half2float(B_val_tmp.y));
//            }
            C_val += A2_val * B_val_tmp;
//            C_val[1] += A2_val * B_val_ptr[1];
        }
    }
//
//
//
    half *C_ptr = d_C + block_id * 2 * N + threadIdx.y * N + threadIdx.x;
//    half2 *C_ptr_2 = reinterpret_cast<half2 *>(C_ptr);
//    if(block_id == 0 && threadIdx.y == 1 && threadIdx.x == 0)
//    {
//        printf("C_val = %f %f\n", __half2float(C_val[0]), __half2float(C_val[1]));
//    }
    C_ptr[0] = C_val;


    return;
}


__global__ void A2_spmm_kcut(const int M,
                             const int N,
                             const int K,
                             const int K_cut,
                             const half* __restrict__ d_A2_vals,
                             const unsigned short* __restrict__ d_A2_Idx,
                             const int* __restrict__ d_row_ptr,
                             const half* __restrict__ d_B,
                             half* Reduction_Workspace)
{
    const int block_id = blockIdx.x;
    const int batch_id = blockIdx.y;

    int row_nnz = d_row_ptr[block_id * K_cut + batch_id + 1] - d_row_ptr[block_id * K_cut + batch_id];

    int                   len        = row_nnz / 8;
    const half*           A2_vals    = d_A2_vals + d_row_ptr[block_id * K_cut + batch_id] * 32 + threadIdx.x * 8;
    const unsigned short* A2_indices = d_A2_Idx + d_row_ptr[block_id * K_cut + batch_id] * 32 + threadIdx.x * 8;

    half C_val[16] = {0.0f};

    //    if(threadIdx.x==0 && block_id==0)
    //    {
    //        printf("row_nnz = %d, len=%d\n", row_nnz, len);
    //    }

    for (int i = 0; i < len; i++) {
        uint4 vals    = *(reinterpret_cast<const uint4*>(A2_vals));
        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));

        const half*           val_ptr = reinterpret_cast<const half*>(&vals);
        const unsigned short* idx_ptr = reinterpret_cast<const unsigned short*>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];

            //            A2_idx = A2_idx/4 * 2 + A2_idx%4;

            //            if(threadIdx.x==0 && block_id==0)
            //            {
            //                printf("i=%d j=%d A2_val = %f, A2_idx=%d\n", i, j, __half2float(A2_val), A2_idx);
            //            }

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }

        A2_vals += 32 * 8;
        A2_indices += 32 * 8;
    }

    half* C_ptr = Reduction_Workspace + batch_id * M * N + block_id * 32 * N;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        C_ptr[threadIdx.x * N + i] = C_val[i];
    }
}

__global__ void A2_spmm_kcut_v1(const int M,
                                const int N,
                                const int K,
                                const int K_cut,
                                const half* __restrict__ d_A2_vals,
                                const unsigned short* __restrict__ d_A2_Idx,
                                const int* __restrict__ d_row_ptr,
                                const half* __restrict__ d_B,
                                half* Reduction_Workspace)
{
    const int block_id = blockIdx.x;
    const int batch_id = blockIdx.y;

    int row_nnz = d_row_ptr[block_id + 1] - d_row_ptr[block_id];

    int len = row_nnz / (8 * K_cut);

    const half*           A2_vals    = d_A2_vals + d_row_ptr[block_id] * 32 + batch_id * len * 8 * 32 + threadIdx.x * 8;
    const unsigned short* A2_indices = d_A2_Idx + d_row_ptr[block_id] * 32 + batch_id * len * 32 + threadIdx.x * 8;

    half C_val[16] = {0.0f};

    //    if(threadIdx.x==0 && block_id==0)
    //    {
    //        printf("row_nnz = %d, len=%d\n", row_nnz, len);
    //    }

    for (int i = 0; i < len; i++) {
        uint4 vals    = *(reinterpret_cast<const uint4*>(A2_vals));
        uint4 indices = *(reinterpret_cast<const uint4*>(A2_indices));

        const half*           val_ptr = reinterpret_cast<const half*>(&vals);
        const unsigned short* idx_ptr = reinterpret_cast<const unsigned short*>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];

            //            A2_idx = A2_idx/4 * 2 + A2_idx%4;

            //            if(threadIdx.x==0 && block_id==0)
            //            {
            //                printf("i=%d j=%d A2_val = %f, A2_idx=%d\n", i, j, __half2float(A2_val), A2_idx);
            //            }

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }

        A2_vals += 32 * 8;
        A2_indices += 32 * 8;
    }

    half* C_ptr = Reduction_Workspace + batch_id * M * N + block_id * 32 * N;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        C_ptr[threadIdx.x * N + i] = C_val[i];
    }
}

__global__ void A2_spmm_async(const int M,
                              const int N,
                              const int K,
                              const half* __restrict__ d_A2_vals,
                              const unsigned short* __restrict__ d_A2_Idx,
                              const int* __restrict__ d_row_ptr,
                              const half* __restrict__ d_B,
                              half* d_C)
{

    __shared__ half           A_shared_val[32 * 8 * 2];
    __shared__ unsigned short A_shared_idx[32 * 8 * 2];

    int block_id = blockIdx.x;
    int row_nnz  = d_row_ptr[block_id + 1] - d_row_ptr[block_id];

    int                   len        = row_nnz / 8;
    const half*           A2_vals    = d_A2_vals + d_row_ptr[block_id] * 32 + threadIdx.x * 8;
    const unsigned short* A2_indices = d_A2_Idx + d_row_ptr[block_id] * 32 + threadIdx.x * 8;

    half C_val[16] = {0.0f};

    // From Global Mem to Shared Mem
    cp_async<16>(A_shared_val + threadIdx.x * 8, A2_vals);
    cp_async_group_commit();

    // From Global Mem to Shared Mem
    cp_async<16>(A_shared_idx + threadIdx.x * 8, A2_indices);
    cp_async_group_commit();

    //    if(threadIdx.x==0 && block_id==1)
    //    {
    //        printf("row_nnz = %d, len=%d\n", row_nnz, len);
    //    }

#pragma unroll(1)
    for (int i = 0; i < len; i++) {
        bool can_copy = (i + 1) < len;

        half* __restrict__ A_val_smem_read_ptr  = A_shared_val;
        half* __restrict__ A_val_smem_write_ptr = A_shared_val;

        A_val_smem_read_ptr  = A_shared_val + (i % 2) * 32 * 8 + threadIdx.x * 8;
        A_val_smem_write_ptr = A_shared_val + ((i + 1) % 2) * 32 * 8 + threadIdx.x * 8;

        unsigned short* __restrict__ A_idx_smem_read_ptr  = A_shared_idx;
        unsigned short* __restrict__ A_idx_smem_write_ptr = A_shared_idx;

        A_idx_smem_read_ptr  = A_shared_idx + (i % 2) * 32 * 8 + threadIdx.x * 8;
        A_idx_smem_write_ptr = A_shared_idx + ((i + 1) % 2) * 32 * 8 + threadIdx.x * 8;

        cp_async_wait_group<0>();
        __syncthreads();

        cp_async<16>(A_val_smem_write_ptr, A2_vals + 32 * 8, can_copy);
        cp_async_group_commit();

        cp_async<16>(A_idx_smem_write_ptr, A2_indices + 32 * 8, can_copy);
        cp_async_group_commit();

        // below is fetch from shared memory to registers

        uint4 vals    = *(reinterpret_cast<const uint4*>(A_val_smem_read_ptr));
        uint4 indices = *(reinterpret_cast<const uint4*>(A_idx_smem_read_ptr));

        const half*           val_ptr = reinterpret_cast<const half*>(&vals);
        const unsigned short* idx_ptr = reinterpret_cast<const unsigned short*>(&indices);

        cp_async_wait_group<0>();
        __syncthreads();

        //        unsigned short *__restrict__ A_shared_idx_mem_ptr = A_shared_idx;
        //
        //
        //
        //        uint4 vals = *(reinterpret_cast<const uint4 *>(A2_vals));
        //        uint4 indices = *(reinterpret_cast<const uint4 *>(A2_indices));
        //
        //        const half *val_ptr = reinterpret_cast<const half *>(&vals);
        //        const unsigned short *idx_ptr = reinterpret_cast<const unsigned short *>(&indices);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            half           A2_val = val_ptr[j];
            unsigned short A2_idx = idx_ptr[j];

            //            if(threadIdx.x==0 && block_id==0)
            //            {
            //                printf("i=%d j=%d A2_val = %f, A2_idx=%d\n", i, j, __half2float(A2_val), A2_idx);
            //            }

            // right now, we only support batch size 16
            const half* B_ptr_1 = d_B + A2_idx * N;
            const half* B_ptr_2 = d_B + A2_idx * N + 8;

            uint4 B_vals_1 = *(reinterpret_cast<const uint4*>(B_ptr_1));
            uint4 B_vals_2 = *(reinterpret_cast<const uint4*>(B_ptr_2));

            const half* B_val_ptr_1 = reinterpret_cast<const half*>(&B_vals_1);
            const half* B_val_ptr_2 = reinterpret_cast<const half*>(&B_vals_2);

#pragma unroll
            for (int k = 0; k < 8; k++) {
                C_val[k] += A2_val * B_val_ptr_1[k];
                C_val[k + 8] += A2_val * B_val_ptr_2[k];
            }
        }

        A2_vals += 32 * 8;
        A2_indices += 32 * 8;

        //        cp_async_wait_group<0>();
        //        __syncthreads();
    }

    half* C_ptr = d_C + block_id * 32 * N;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        C_ptr[threadIdx.x * N + i] = C_val[i];
    }
}

cudaError_t
FSP_Computation(int M, int N, int K, half *d_A2_vals, unsigned short *d_A2_Idx, int *d_row_ptr, half *d_B, half *d_C) {
    dim3 gridDim(M / 4, 1, 1);
    dim3 blockDim(8, 4, 1);

//    dim3 gridDim(M / 2, 1, 1);
//    dim3 blockDim(16, 2, 1);

//    dim3 gridDim(M, 1, 1);
//    dim3 blockDim(32, 1, 1);

//    A2_spmm_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_sputnik_spadding_async<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_sputnik_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_sputnik_spadding_sf_ws<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_sputnik<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);

//    printf("A2_spmm_sputnik_spadding\n");
    return cudaGetLastError();
}


cudaError_t
FSP_Computation_reorder(int M, int N, int K, half *d_A2_vals, unsigned short *d_A2_Idx, int *d_row_ptr,
                        int *d_row_indices, half *d_B, half *d_C) {
    dim3 gridDim(M / 4, 1, 1);
    dim3 blockDim(8, 4, 1);

    if (N == 16)
        A2_spmm_sputnik_spadding_sf_ws_v2<2, half2><<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_row_indices, d_B, d_C);
    else if (N == 32)
        A2_spmm_sputnik_spadding_sf_ws_v2<4, half4><<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_row_indices, d_B, d_C);
    else if (N == 64)
        A2_spmm_sputnik_spadding_sf_ws_v2<8, half8><<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_row_indices, d_B, d_C);



//    printf("A2_spmm_sputnik_spadding\n");
    return cudaGetLastError();
}

cudaError_t
FSP_Computation_sputnik(int M, int N, int K, half *d_A2_vals, unsigned short *d_A2_Idx, int *d_row_ptr, int *d_row_length, half *d_B, half *d_C) {
    dim3 gridDim(M / 4, 1, 1);
    dim3 blockDim(8, 4, 1);

//    dim3 gridDim(M / 2, 1, 1);
//    dim3 blockDim(16, 2, 1);

//    dim3 gridDim(M, 1, 1);
//    dim3 blockDim(32, 1, 1);

//    A2_spmm_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
    A2_spmm_sputnik_spadding<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, d_C);
//    A2_spmm_sputnik<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_row_ptr, d_row_length, d_B, d_C);

//    printf("A2_spmm_sputnik_spadding\n");
    return cudaGetLastError();
}


cudaError_t
FSP_Computation_sputnik_sf(int M, int N, int K, half *d_A2_vals, unsigned short *d_A2_Idx, int *d_tile_ptr,
                           int *d_row_ptr, u_int8_t *d_min_tile_length, half *d_B, half *d_C) {
    dim3 gridDim(M / 4, 1, 1);
    dim3 blockDim(8, 4, 1);

    A2_spmm_sputnik_spadding_sf_ws_v1<<<gridDim, blockDim>>>(M, N, K, d_A2_vals, d_A2_Idx, d_tile_ptr, d_row_ptr, d_min_tile_length, d_B, d_C);

//    printf("A2_spmm_sputnik_spadding\n");
    return cudaGetLastError();
}


cudaError_t FSP_Computation_Kcut(int             M,
                                 int             N,
                                 int             K,
                                 int             Kcut,
                                 half*           d_A2_vals,
                                 unsigned short* d_A2_Idx,
                                 int*            d_row_ptr,
                                 half*           d_B,
                                 half*           Reduction_Space)
{
    dim3 gridDim(M / 32, Kcut, 1);
    dim3 blockDim(32, 1, 1);

    A2_spmm_kcut_v1<<<gridDim, blockDim>>>(M, N, K, Kcut, d_A2_vals, d_A2_Idx, d_row_ptr, d_B, Reduction_Space);

    return cudaGetLastError();
}

#endif  // FLASH_LLM_SPTC_FSP_COMPUTATION_H
