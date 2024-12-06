//
// Created by paramath on 11/30/23.
//

#ifndef FLASH_LLM_SPTC_SPMM_DECOMPOSE_H
#define FLASH_LLM_SPTC_SPMM_DECOMPOSE_H
#include <vector>
#include <algorithm>
#include <numeric>
#include "../csrc/TilingConfig.h"

void decompose(half* A, half*A1, half*A2, int length){
    // split the matrix A into A1 and A2
    // A1 is for the saprse tensor core, A2 is for the finegrained sparse kernel
    memset(A1, 0, sizeof(half)*length);
    memset(A2, 0, sizeof(half)*length);
    assert(length % 4 == 0);

    int nnz1 = 0;
    int nnz2 = 0;
    for (int i = 0; i < length; i += 4)
    {
        int index_a1 = 0;
        for (int index = 0; index < 4; index++)
        {
            if (__half2float(A[i + index]) != 0.0f)
            {
                if (index_a1 < 2)
                {
//                    A1[i / 2 + index_a1] = A[i + index];
                    A1[i + index] = A[i + index];
                    index_a1++;
                    nnz1++;
                }
                else
                {
//                    printf("check i=%d index=%d index_a1=%d\n", i, index, index_a1);
//                    getchar();
//                    if((i*4) / 64 ==143)
//                        continue ;
                    A2[i + index] = A[i + index];
                    index_a1++;
                    nnz2++;
                }
            }
        }
    }
//    assert(__half2float(A1[9214])==0.0f);
    printf("NNZ1= %d ratio = %f pad_nnz=%d \nNNZ2= %d ratio = %f\n", nnz1, nnz1 * 1.0/(nnz1+nnz2), length * 2/4,  nnz2, nnz2 * 1.0/(nnz1+nnz2) );
    printf("A1 padding ratio = %f\n", 1-nnz1 * 1.0/(length * 2/4));
}

void decompose_v1(half* A, half*A1, half*A2, int length){
    // split the matrix A into A1 and A2
    // A1 is for the saprse tensor core, A2 is for the finegrained sparse kernel
    memset(A1, 0, sizeof(half)*length);
    memset(A2, 0, sizeof(half)*length);
    assert(length % 4 == 0);

    int nnz1 = 0;
    int nnz2 = 0;
    for (int i = 0; i < length; i += 4)
    {
        int index_a1 = 0;

        half a0 = A[i ];
        if(__half2float(a0) != 0.0f)
        {
            ++index_a1;
        }
        half a1 = A[i + 1];
        if(__half2float(a1) != 0.0f)
        {
            ++index_a1;
        }
        half a2 = A[i + 2];
        if (__half2float(a2) != 0.0f)
        {
            ++index_a1;
        }
        half a3 = A[i + 3];
        if(__half2float(a3) != 0.0f)
        {
            ++index_a1;
        }

        if(index_a1 > 1)
        {
            int index_c = 0;
            for (int index = 0; index < 4; index++)
            {

                if (__half2float(A[i + index]) != 0.0f)
                {
                    if (index_c < 2)
                    {
                        //                    A1[i / 2 + index_a1] = A[i + index];
                        A1[i + index] = A[i + index];
                        index_c++;
                        nnz1++;
                    }
                    else
                    {
                        A2[i + index] = A[i + index];
                        nnz2++;
                    }
                }
            }

        }
        else{
//            int index_c = 0;
            for (int index = 0; index < 4; index++)
            {

                if (__half2float(A[i + index]) != 0.0f)
                {

                    A2[i + index] = A[i + index];
                    nnz2++;
//                    if (index_a1 < 2)
//                    {
//                        //                    A1[i / 2 + index_a1] = A[i + index];
//                        A1[i + index] = A[i + index];
//                        index_a1++;
//                        nnz1++;
//                    }
//                    else
//                    {
//                        A2[i + index] = A[i + index];
//                        nnz2++;
//                    }
                }
            }


        }


    }
    printf("NNZ1= %d ratio = %f pad_nnz=%d \nNNZ2= %d ratio = %f\n", nnz1, nnz1 * 1.0/(nnz1+nnz2), length * 2/4,  nnz2, nnz2 * 1.0/(nnz1+nnz2) );
}


__host__ void InitUnStructureSparseMatrix(half * A2, int M, int K, half *vals, uint8_t *col_Ind, int *row_ptr)
{
    const int FUNC_TILE_M = 128;
    const int FUNC_TILE_K = 64;

    for(int i = 0; i < M/FUNC_TILE_M; i++)
    {
        for (int j = 0; j < K/FUNC_TILE_K; ++j) {
//            int nnz = 0;
            for(int m = 0; m < FUNC_TILE_M; m++)
            {
                int nnz = 0;
                for (int k = 0; k < FUNC_TILE_K; ++k) {
                    int index = i * FUNC_TILE_M * K + j * FUNC_TILE_K + m * K + k;
                    if(__half2float(A2[index]) != 0.0f)
                    {
                        ++nnz;
                    }
                }
                printf("row=%d col=%d lr=%d nnz=%d\n", i, j, m, nnz);

            }

//            printf("row=%d col=%d nnz=%d ratio=%f ratio_v1=%f\n", i, j, nnz, nnz* 1.0/(TILE_M), nnz * 1.0/(128));

        }
    }


}

__host__ void InitUnStructureSparseMatrix_V1(half * A2, int M, int K, half *vals, uint8_t *col_Ind, int *row_ptr)
{
    const int FUNC_TILE_M = 128;
    const int FUNC_TILE_K = 64;

    for(int i = 0; i < M/FUNC_TILE_M; i++)
    {
        for (int j = 0; j < K/FUNC_TILE_K; ++j) {


            for(int k = 0; k < FUNC_TILE_K; k++)
            {
                int nnz = 0;
                for (int m = 0; m < FUNC_TILE_M; ++m) {
                    int index = i * FUNC_TILE_M * K + j * FUNC_TILE_K + m * K + k;
                    if(__half2float(A2[index]) != 0.0f)
                    {
                        ++nnz;
                    }
                }
                printf("row=%d col=%d lc=%d nnz=%d\n", i, j, k, nnz);
            }
//            for(int m = 0; m < TILE_M; m++)
//            {
//                int row = i * TILE_M + m;
//                //                int nnz = 0;
//                for (int k = 0; k < TILE_K; ++k) {
//                    int index = i * TILE_M * K + j * TILE_K + m * K + k;
//                    if(__half2float(A2[index]) != 0.0f)
//                    {
//                        ++nnz;
//                    }
//                }
//                //                printf("row=%d col=%d lr=%d nnz=%d\n", i, j, m, nnz);
//
//            }
//
//            //            printf("row=%d col=%d nnz=%d ratio=%f\n", i, j, nnz, nnz* 1.0/(TILE_K/2));

        }
    }


}

__host__ void InitUnStructureSparseMatrix_Pattern(half*     A2,
                                                 int       M,
                                                 int       K,
                                                 half**    A2_vals,
                                                 uint8_t** A2_col_Ind,
                                                 int**     A2_row_ptr,
                                                 int&      len_vals,
                                                 int&      len_col_Ind)
{
    const int FUNC_TILE_M = 128;
    const int FUNC_TILE_K = 64;

//    const int Pattern_M = 2;
//
//    static_assert(Pattern_M == 2, "Pattern_M must be 2");
//
//    int num_pattern = int(pow(2, Pattern_M))-1;

    int num_blocks = M/FUNC_TILE_M * K/FUNC_TILE_K;

    *A2_row_ptr = (int *)malloc(sizeof(int)*(num_blocks*FUNC_TILE_M + 1));
//    *A2_row_col_ptr = (int *)malloc(sizeof(int)*(num_blocks*num_pattern*TILE_M/Pattern_M + 1));
//    len_row_ptr = num_blocks*num_pattern*TILE_M/Pattern_M + 1;
//    len_row_col_ptr = num_blocks*num_pattern*TILE_M/Pattern_M + 1;

    int tile_id = 0;
    int offset = 0;
//    int col_offset = 0;

    vector<int> nnz_per_tile;
    //generate row_ptr and row_col_ptr
    for(int i = 0; i < M/FUNC_TILE_M; i++)
    {
        for (int j = 0; j < K/FUNC_TILE_K; ++j) {
            int nnz_tile = 0;
            for(int m = 0; m < FUNC_TILE_M; m++)
            {
                vector<uint8_t> col_Ind;
                vector<half> vals;
//
                (*A2_row_ptr)[tile_id++] = offset; // first pattern: 0

                for(int k = 0; k < FUNC_TILE_K; k++)
                {
                    int index = (i * FUNC_TILE_M + m) * K + j * FUNC_TILE_K + k;
                    if(__half2float(A2[index]) != 0.0f)
                    {
                        vals.push_back(A2[index]);
                        col_Ind.push_back(k);
                    }
                }

                assert(vals.size() == col_Ind.size());
                nnz_tile += vals.size();
                offset += vals.size();
            }
            nnz_per_tile.push_back(nnz_tile);
        }
    }

    int min_tile_size = *min_element(nnz_per_tile.begin(), nnz_per_tile.end());
    int max_tile_size = *max_element(nnz_per_tile.begin(), nnz_per_tile.end());
    int avg_tile_size = std::accumulate(nnz_per_tile.begin(), nnz_per_tile.end(), 0) / nnz_per_tile.size();
    std::sort(nnz_per_tile.begin(), nnz_per_tile.end());
    int median_tile_size = nnz_per_tile[nnz_per_tile.size() / 2];
    printf("min_tile_size=%d max_tile_size=%d avg_tile_size=%d median_tile_size=%d\n", min_tile_size, max_tile_size, avg_tile_size, median_tile_size);
//    getchar();
//    (*A2_row_ptr)[tile_id] = offset;
    (*A2_row_ptr)[tile_id] = offset;
    assert(tile_id == num_blocks*FUNC_TILE_M);

    *A2_vals = (half *)malloc(sizeof(half)*(offset));
    *A2_col_Ind = (uint8_t *)malloc(sizeof(uint8_t)*(offset));
    len_vals = offset;
    len_col_Ind = offset;
    //copy the data
    tile_id = 0;
    for(int i = 0; i < M/FUNC_TILE_M; i++)
    {
        for (int j = 0; j < K/FUNC_TILE_K; ++j) {
            for(int m = 0; m < FUNC_TILE_M; m++)
            {
                vector<uint8_t> col_Ind;
                vector<half> vals;

                for(int k = 0; k < FUNC_TILE_K; k++)
                {
                    int index = (i * FUNC_TILE_M + m) * K + j * FUNC_TILE_K + k;
                    if(__half2float(A2[index]) != 0.0f)
                    {
                        vals.push_back(A2[index]);
                        col_Ind.push_back(k);
                    }
                }

                memcpy(*A2_vals + (*A2_row_ptr)[tile_id], vals.data(), sizeof(half)*vals.size());
                memcpy(*A2_col_Ind + (*A2_row_ptr)[tile_id], col_Ind.data(), sizeof(uint8_t)*col_Ind.size());
                assert((*A2_row_ptr)[tile_id+1] - (*A2_row_ptr)[tile_id] == vals.size());
//                assert((*A2_row_col_ptr)[tile_id+1] - (*A2_row_col_ptr)[tile_id] == col_Ind.size());

                tile_id ++;

            }
        }
    }
}

__host__ int InitUnStructureSparseMatrix_SegScan_COO(half* A2,
                                                     int M,
                                                     int K,
                                                     half** A2_vals,
                                                     u_int16_t **A2_Ind,
                                                    int** A2_row_ptr){
    const int FUNC_TILE_M = 256;
    const int FUNC_TILE_K = 64;
    const int VEC_SIZE = FINE_VECTOR_SIZE;

    int offset = 0;
    int tile_id = 0;

    int num_blocks = M/FUNC_TILE_M * K/FUNC_TILE_K;
    *A2_row_ptr = (int *)malloc(sizeof(int)*(num_blocks*FUNC_TILE_M + 1));

    vector <half> all_vals;
    vector <u_int16_t> all_inds;
    vector<bool> all_start_flags; // per element, indicating whether current element is the first one of one specific row
    vector<bool> all_end_flags; // per element, indicating whether current element is the last one of one specific row

    int longest_tile_size = 0;

    for(int m1 = 0; m1 < M / FUNC_TILE_M; m1++)
    {
        for(int k1 = 0; k1 < K / FUNC_TILE_K; k1++)
        {
            int local_nnz = 0;
            (*A2_row_ptr)[tile_id++] = offset;
            for(int m2 = 0; m2 < FUNC_TILE_M; m2++)
            {
                bool first_element = false;
                int row_nnz= 0;
                for(int k2 = 0; k2 < FUNC_TILE_K; k2++)
                {
                    int index = (m1 * FUNC_TILE_M + m2) * K + k1 * FUNC_TILE_K + k2;
                    if(__half2float(A2[index]) != 0.0f)
                    {
                        all_vals.push_back(A2[index]);
                        all_inds.push_back(m2 * TILE_K + k2);

                        if(!first_element)
                        {
                            first_element = true;
                            all_start_flags.push_back(true);
                        }
                        else
                        {
                            all_start_flags.push_back(false);
                        }

                        if((local_nnz % (128 * VEC_SIZE)) == 0 && local_nnz != 0)
                        {
                            all_start_flags[all_start_flags.size()-1] = true;
                        }

                        local_nnz++;
                        row_nnz++;

//                        if(m2==241)
//                            printf("k2=%d\n", k2);
                    }
                }

//                if(m2==241){
//                    printf("row_nnz=%d local_nnz=%d\n", row_nnz, local_nnz);
//                    getchar();
//                }

                longest_tile_size = max(longest_tile_size, row_nnz);
            }

            if(local_nnz % VEC_SIZE != 0)
            {
                int pad_size = VEC_SIZE - local_nnz % VEC_SIZE;
                for(int i = 0; i < pad_size; i++)
                {
                    all_vals.push_back(__float2half_rn(0.0f));
                    all_start_flags.push_back(false);
                    all_inds.push_back(all_inds[all_inds.size()-1]);
                }
                local_nnz += pad_size;
            }
            offset += local_nnz;
//            printf("row=%d col=%d nnz=%d\n", m1, k1, local_nnz);
        }
    }
    (*A2_row_ptr)[tile_id] = offset;
    printf("%d %d\n", tile_id, offset);
    printf("longest tile=%d\n", longest_tile_size);

    if(offset==0)
        return 0;
    assert(offset == all_vals.size());
    assert(all_vals.size() == all_start_flags.size());

    *A2_vals = (half *)malloc(sizeof(half)*(offset));
    memcpy(*A2_vals, all_vals.data(), sizeof(half)*offset);
    all_end_flags.resize(all_start_flags.size());

    for(int i=0; i < all_start_flags.size(); i += VEC_SIZE)
    {
        for(int j = 0; j < VEC_SIZE; j++)
        {
            if(i + j < all_start_flags.size()-1)
            {
                if(all_start_flags[i+j+1])
                {
                    all_end_flags[i+j] = true;
                }
                else
                {
                    all_end_flags[i+j] = false;
                }
            }
        }
    }
    all_end_flags[all_end_flags.size()-1] = true;
    assert(all_start_flags.size() == all_end_flags.size());
    int total_nnz = all_start_flags.size();
    *A2_Ind = (u_int16_t *)malloc(sizeof(u_int16_t)*total_nnz);

    for(int i=0; i < total_nnz; i++)
    {
        u_int16_t ind = all_inds[i];
        bool start_flag = all_start_flags[i];
        bool end_flag = all_end_flags[i];

        if(start_flag){
            ind = ind | 0x8000;
        }

        if(end_flag){
            ind = ind | 0x4000;
        }

        (*A2_Ind)[i] = ind;
    }

    return total_nnz;
}

__host__ void InitUnStructureSparseMatrix_Pattern_V4(half * A2, int M, int K, half *vals, uint8_t *col_Ind, int *row_ptr)
{
    const int FUNC_TILE_M = 128;
    const int FUNC_TILE_K = 64;

    const int Pattern_M = 4;

    static_assert(Pattern_M == 4, "Pattern_M must be 4");

    for(int i = 0; i < M/FUNC_TILE_M; i++)
    {
        for (int j = 0; j < K/FUNC_TILE_K; ++j) {
//            int cnt_pattern0 = 0;
//            int cnt_pattern1 = 0;
//            int cnt_pattern2 = 0;
//            int cnt_pattern3 = 0;
            vector<int> cnt_pattern(16, 0);

            for(int k = 0; k < FUNC_TILE_K; k++)
            {
                for(int gm = 0; gm < FUNC_TILE_M/Pattern_M; gm++)
                {
                    int index = i * FUNC_TILE_M * K + j * FUNC_TILE_K + gm * Pattern_M * K + 0 + k;
                    int index1 = i * FUNC_TILE_M * K + j * FUNC_TILE_K + gm * Pattern_M * K + K + k;
                    int index2 = i * FUNC_TILE_M * K + j * FUNC_TILE_K + gm * Pattern_M * K + 2*K+ k;
                    int index3 = i * FUNC_TILE_M * K + j * FUNC_TILE_K + gm * Pattern_M * K + 3*K + k;

                    int pattern0 = __half2float(A2[index]) != 0.0f ? 1 : 0;
                    int pattern1 = __half2float(A2[index1]) != 0.0f ? 1 : 0;
                    int pattern2 = __half2float(A2[index2]) != 0.0f ? 1 : 0;
                    int pattern3 = __half2float(A2[index3]) != 0.0f ? 1 : 0;
                    int pattern = pattern3*8 + pattern2*4 + pattern1 * 2 + pattern0;

                    cnt_pattern[pattern] += 1;
                }
            }
//            printf("row=%d col=%d p0=%d p1=%d p2=%d p3=%d %d\n", i, j, cnt_pattern0, cnt_pattern1, cnt_pattern2, cnt_pattern3, cnt_pattern0 + cnt_pattern1 + cnt_pattern2 + cnt_pattern3);
            printf("row=%d col=%d ", i, j);
            for (int p = 0; p < 16; ++p) {
                printf("p%d=%d ", p, cnt_pattern[p]);
            }
            printf("\n");
        }
    }


}
#endif  // FLASH_LLM_SPTC_SPMM_DECOMPOSE_H
