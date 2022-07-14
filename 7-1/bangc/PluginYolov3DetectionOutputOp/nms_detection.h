/*copyright (C) [2020] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef _NMS_DETECTION_H_
#define _NMS_DETECTION_H_
// TODO：完成NMS BANGC算子的编写


#define NMS_SIZE 64 // 向量运算对齐大小
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y

enum Addr { NRAM, SRAM, GDRAM };
enum SplitMode { NMS_BLOCK = 1, NMS_U1 = 4 };

// max(x, y) ~ max(x - y, 0) + y
template <typename NMS_DT>
__mlu_func__ void __svmax_relu(NMS_DT* dst, NMS_DT* src0, NMS_DT* src1, int len) {
  __bang_sub(dst, src0, src1, len);
  __bang_active_relu(dst, dst, len);
  __bang_add(dst, dst, src1, len);
}

// min(x, y) ~ y - max(y - x, 0)
template <typename NMS_DT>
__mlu_func__ void __svmin_relu(NMS_DT* dst, NMS_DT* src0, NMS_DT* src1, int len) {
  __bang_sub(dst, src1, src0, len);
  __bang_active_relu(dst, dst, len);
  __bang_sub(dst, src1, dst, len);
}


/*!
* 实现非极大值抑制（NMS），支持输入和输出地址空间的多样化选择（包括GDRAM/SRAM/NRAM）
    输入、输出、中间计算结果数据类型：half/float

* @param[out] output_box_num    筛选出的边界框的总个数
* @param[out] output_data       计算结果存放地址
* @param[out] dst               计算结果存放地址类型：GDRAM/SRAM/NRAM
* @param[in] input_data_score   待筛选边界框的score的地址
* @param[in] input_data_box     待筛选边界框的坐标的地址 - 存储顺序 [x1, y1, x2, y2]
* @param[in] src                输入数据存放地址类型：GDRAM/SRAM/NRAM
* @param[in] buffer             计算使用的NRAM空间首地址
* @param[in] buffer_size        计算使用的NRAM空间大小 - 单位：字节
* @param[in] sram               在函数外部声明的SRAM的地址空间，在UNION1模式下通过对SRAM的读写完成核间通信，找到score的最大值
* @param[in] split_mode         拆分模式: NMS_BLOCK(BLOCK) / NMS_U1(UNION1)
* @param[in] input_box_num      输入的待筛选边界框的数量
* @param[in] input_stride       输入数据的步长
* @param[out] output_stride     输出数据的步长
* @param[in] keepNum            根据概率排序选择保留概率最高的边界框个数
* @param[in] thresh_iou         交并比阈值
* @param[in] thresh_score       score阈值
* @param[in] save_method        存储模式: 0/1/2
    0: 
    score, x1, y1, x2, y2 | score x1, y1, x2, y2 | ...
    1:
    |---output_stride---|
    score, score, ...0000
    x1, x1, x1, .....0000
    y1, y1, y1, .....0000
    x2, x2, x2, .....0000
    y2, y2, y2, .....0000
    2: 
    when dst == NRAM, save selected box score only at original location.
*/

template <typename NMS_DT>
__mlu_func__ void nms_detection(int &output_box_num,            // 筛选出的边界框的总个数
                                NMS_DT* output_data,            // 计算结果存放地址
                                Addr dst,                       // 计算结果存放地址类型：GDRAM/SRAM/NRAM
                                NMS_DT* input_data_score,       // 待筛选边界框的score的地址
                                NMS_DT* input_data_box,         // 待筛选边界框的坐标的地址 - 存储顺序 [x1, y1, x2, y2]
                                Addr src,                       // 输入数据存放地址类型：GDRAM/SRAM/NRAM
                                NMS_DT* buffer,                 // 计算使用的NRAM空间首地址
                                int buffer_size,                // 计算使用的NRAM空间大小 - 单位：字节
                                NMS_DT* sram,                   // 在函数外部声明的SRAM的地址空间，for UNION1
                                SplitMode split_mode,           // 拆分模式: NMS_BLOCK(BLOCK) / NMS_U1(UNION1)
                                int input_box_num,              // 输入的待筛选边界框的数量
                                int input_stride,               // 输入数据的步长
                                int output_stride,              // 输出数据的步长
                                int keepNum,                    // 根据概率排序选择保留概率最高的边界框个数
                                NMS_DT thresh_iou,              // 交并比阈值
                                NMS_DT thresh_score,            // confidence score 阈值
                                int save_method                 // 存储格式：0 / 1 / 2
                                ) {
    /*====== PREPARATORY  ======*/
    /*------ 变量声明 ------*/
    int core_limit = split_mode;    // 启用的核数
    int32_t* loop_end_flag = (int32_t *)(sram + 28);  // for U1: 结束标识符
    loop_end_flag[0] = 0;
    int nms_buffer_count1 = 9;
    int nms_buffer_count2 = 4;
    int nram_save_limit_count;      // NRAM上临时存储待筛选边界框数量
    nram_save_limit_count = dst == NRAM ? 0 : 256;

    /* 数据调度模式
    0: load data to NRAM buffer first; 
    1: compute directly, when
        input data is on NRAM,
        buffer size is enough, 
        and input_box_num is aligned to 64
    */
    int MODE = 0;
    if (src == NRAM) {
        int flag1 = (input_box_num == NMS_UP(input_box_num, NMS_SIZE));  // input_box_num must be pad
        int flag2 = (buffer_size > (nms_buffer_count2 * input_box_num + 64 /*max_box*/ +
                                    (nram_save_limit_count * 5) * (dst != NRAM)) *
                                    sizeof(NMS_DT));  // buffer is enough
        if (flag1 && flag2)
        MODE = 1;
    }

    // input data ptr
    NMS_DT* input_score_ptr;
    NMS_DT* input_x1_ptr;
    NMS_DT* input_y1_ptr;
    NMS_DT* input_x2_ptr;
    NMS_DT* input_y2_ptr;
    input_score_ptr = input_data_score;
    input_x1_ptr = input_data_box;
    input_y1_ptr = input_x1_ptr + input_stride;
    input_x2_ptr = input_y1_ptr + input_stride;
    input_y2_ptr = input_x2_ptr + input_stride;

    // nram data ptr
    NMS_DT* x1;             // buffer空间，存放x1
    NMS_DT* y1;
    NMS_DT* x2;
    NMS_DT* y2;
    NMS_DT* score;          // buffer空间，存放score
    NMS_DT* inter_x1;       // buffer空间，IoU筛选临时空间
    NMS_DT* inter_y1;
    NMS_DT* inter_x2;
    NMS_DT* inter_y2;
    NMS_DT* max_box;        // buffer空间，存放置信度最高的边界框信息 [score, x1, y1, x2, y2]
    NMS_DT* nram_save;      // buffer空间，待筛选边界框的临时存储空间


    int limit = 0;          // 根据片上的buffer size计算一次最多能处理的输入框的个数
    int len_core = 0;       // 每个核需要处理的框的个数
    int max_seg_pad = 0;    // 每次处理输入框的个数，根据limit进行下补齐，满足硬件限制 the max length every repeat
    int repeat = 0;         // 整数段，需要处理几次max_seg_pad
    int remain = 0;         // 余数段，剩下的需要处理的框的个数
    int remain_pad = 0;     // 余数段进行补齐后的框个数，满足向量计算函数的对齐限制
    int input_offset = 0;   // 当前核处理的输入数据的起始地址偏移 offset of input_data for current core
    int nram_save_count = 0;// NRAM临时空间存储的框个数 -> 缓存+批量拷贝机制

    /*------ 数据划分 ------*/
    // 片上空间划分
    if (src == NRAM) {
        if (MODE != 0) {
            repeat = 0;
            remain = input_box_num;
            remain_pad = remain;
        } else {
            limit = (buffer_size - 64 * sizeof(NMS_DT) /*reserve for max score box*/ -
                    nram_save_limit_count * 5 * sizeof(NMS_DT)) /
                    (nms_buffer_count1 * sizeof(NMS_DT));
            len_core = input_box_num;
            input_offset = 0;
            max_seg_pad = NMS_DOWN(limit, NMS_SIZE);
            repeat = len_core / max_seg_pad;
            remain = len_core % max_seg_pad;
            remain_pad = NMS_UP(remain, NMS_SIZE);
        }
    }
    // src == SRAM or GDRAM
    else {
        limit = (buffer_size - 64 * sizeof(NMS_DT) -
                nram_save_limit_count * 5 * sizeof(NMS_DT)) /
                (nms_buffer_count1 * sizeof(NMS_DT));
        if (core_limit == 1) {
            len_core = input_box_num;
            input_offset = 0;
        } else {
            // 多核拆分 (尽量平均)
            if (input_box_num % core_limit == 0) {
                len_core = input_box_num / core_limit;
                input_offset = coreId * len_core;
            } else {
                // 
                int avg_core = input_box_num / core_limit;
                int tmp = input_box_num % core_limit;
                coreId < tmp ? len_core = avg_core + 1 : len_core = avg_core;
                input_offset = avg_core * coreId + (coreId <= tmp ? coreId : tmp);
            }
        }
        max_seg_pad = NMS_DOWN(limit, NMS_SIZE);
        repeat = len_core / max_seg_pad;
        remain = len_core % max_seg_pad;
        remain_pad = NMS_UP(remain, NMS_SIZE);
    }

    // init the data ptr
    if (src == NRAM && MODE != 0) {
        inter_x1 = buffer;
        inter_y1 = inter_x1 + input_box_num;
        inter_x2 = inter_y1 + input_box_num;
        inter_y2 = inter_x2 + input_box_num;
        max_box = inter_y2 + input_box_num;  // the max score, x1, y1, x2, y2
        nram_save = max_box + 64;
    } else {
        score = buffer;
        x1 = score + max_seg_pad;
        y1 = x1 + max_seg_pad;
        x2 = y1 + max_seg_pad;
        y2 = x2 + max_seg_pad;
        inter_x1 = y2 + max_seg_pad;
        inter_y1 = inter_x1 + max_seg_pad;
        inter_x2 = inter_y1 + max_seg_pad;
        inter_y2 = inter_x2 + max_seg_pad;
        max_box = inter_y2 + max_seg_pad;  // the max score, x1, y1, x2, y2
        nram_save = max_box + 64;
    }


    /*====== EXECUTION PHASE ======*/

    for (int keep = 0; keep < keepNum; keep++) {
        if (core_limit != 1) {
            __sync_cluster();  // sync before current loop
        }

        int max_index = 0;         // the max score index
        int global_max_index = 0;  // for U1
        NMS_DT max_area = 0;       // the max socre area
        max_box[0] = 0;            // init 0

        // Find the box with max confidence score in every core
        for (int i = 0; i <= repeat; i++) {
            if (i == repeat && remain == 0)
                break;
            int seg_len = 0;  // the length every nms compute (padded)
            int cpy_len = 0;  // the length every nms memcpy
            seg_len = (i == repeat ?  remain_pad : max_seg_pad);
            cpy_len = (i == repeat ?  remain : max_seg_pad);

            // Load NMS
            if (MODE != 0) {
                score = input_score_ptr;
            } else {
                mluMemcpyDirection_t load_dir = SRAM2NRAM;
                if (src == NRAM) {
                    load_dir = NRAM2NRAM;
                } else if (src == SRAM) {
                    load_dir = SRAM2NRAM;
                } else {
                    load_dir = GDRAM2NRAM;
                }
                __nramset(score, seg_len, 0);       // 补0对齐
                __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT), 
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
            }

            __bang_max(inter_x1, score, seg_len);   // inter_x1: 临时存储score最大值[0]及其索引[1]
            if (inter_x1[0] > max_box[0]) {
                max_box[0] = inter_x1[0];
                max_index = ((unsigned short *)inter_x1)[1] * (sizeof(NMS_DT) == 2) +
                            ((unsigned int *)inter_x1)[1] * (sizeof(NMS_DT) == 4) +     // index in the core
                            input_offset + i * max_seg_pad;  // offset start from head of input_data
            }
        }

        // Find Global Max
        if (core_limit == 1) {
            max_box[1] = input_x1_ptr[max_index];
            max_box[2] = input_y1_ptr[max_index];
            max_box[3] = input_x2_ptr[max_index];
            max_box[4] = input_y2_ptr[max_index];
            max_area = (max_box[3] - max_box[1]) * (max_box[4] - max_box[2]);
            input_score_ptr[max_index] = 0;
            global_max_index = max_index;
        } else if (core_limit == 4) {
            // get the max box's x1, y1, x2, y2 on every core
            max_box[1] = input_x1_ptr[max_index];
            max_box[2] = input_y1_ptr[max_index];
            max_box[3] = input_x2_ptr[max_index];
            max_box[4] = input_y2_ptr[max_index];
            ((int32_t *)(max_box + 5))[0] = max_index;
            // copy every core's box info to sram, score---x1---y1---x2---y2---
            for (int i = 0; i < 5; i++) {
                __memcpy(sram + i * core_limit + coreId, max_box + i, 1 * sizeof(NMS_DT), NRAM2SRAM);
            }
            // copy every core's max_index to sram, use 2 NMS_DT to store max_index
            __memcpy(sram + 5 * core_limit + coreId * 2, max_box + 5, 2 * sizeof(NMS_DT), NRAM2SRAM);
            __sync_cluster();

            
            // copy score from sram to nram and find the max
            __nramset(inter_x1, 64, 0);
            __memcpy(inter_x1, sram, core_limit * sizeof(NMS_DT), SRAM2NRAM);
            __bang_max(max_box, inter_x1, 64);

            int max_core = ((unsigned short *)max_box)[1] * (sizeof(NMS_DT) == 2) +
                           ((unsigned int *)max_box)[1] * (sizeof(NMS_DT) == 4);
            // copy the max box to max_box
            __memcpy(max_box + 1, sram + 1 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);     // x1
            __memcpy(max_box + 2, sram + 2 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);     // y1
            __memcpy(max_box + 3, sram + 3 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);     // x2
            __memcpy(max_box + 4, sram + 4 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);     // y2
            __memcpy(max_box + 5, sram + 5 * core_limit + 2 * max_core, 2 * sizeof(NMS_DT), SRAM2NRAM); // max_index
            max_area = (max_box[3] - max_box[1]) * (max_box[4] - max_box[2]);
            global_max_index = ((int32_t *)(max_box + 5))[0];

            // 将搜索出的score最大的候选框的score置为零（排除在之后的移除操作之外）
            if (src != NRAM) {
                input_score_ptr[global_max_index] = 0;
            } else {
                if (coreId == max_core) {
                    input_score_ptr[global_max_index] = 0;
                }
            }
        }

        /*----- STORE -----*/
        // store to sram/gdram
        if (dst != NRAM && output_box_num != 0) {
            // 设置存储拷贝方向
            mluMemcpyDirection_t store_dir = NRAM2GDRAM;
            if (dst == SRAM) {
                store_dir = NRAM2SRAM;
            } else {
                store_dir = NRAM2GDRAM;
            }

            if ((nram_save_count == nram_save_limit_count) || (max_box[0] <= thresh_score)) {
                if (core_limit == 1) {
                    if (save_method == 0) {  // [score, x1, y1, x2, y2]
                        __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
                        output_data += nram_save_count * 5;
                    } else {  // score---, x1---, y1---, x2---, y2---
                        __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                                 output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
                        output_data += nram_save_count;
                    }
                    nram_save_count = 0;
                } else {
                    if (coreId == coreDim - 1) {
                        if (save_method == 0) {  // score, x1, y1, x2, y2
                            __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
                            output_data += nram_save_count * 5;
                        } else {  // score---, x1---, y1---, x2---, y2---
                            __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                                    output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
                            output_data += nram_save_count;
                        }
                        nram_save_count = 0;
                    }
                }   // if core_limit
            }       // if move data nram->sram/gdram
        }           // if dst

        // if the max score <= thresh, end
        if (core_limit == 1) {
            if (max_box[0] <= thresh_score) {
                break;
        }
        } else {
            if (max_box[0] <= thresh_score) {
                if (coreId == coreDim - 1) {
                    loop_end_flag[0] = 1;
                }
            }
            __sync_cluster();  // wait for update loop_end_flag
            if (loop_end_flag[0] == 1) {
                break;
            }
        }
        
        // store to nram
        NMS_DT* save_ptr;
        int save_offset = 0;
        int save_str_num = 0;
        if (dst == NRAM) {
            save_ptr = output_data;
            save_offset = output_box_num;
            save_str_num = input_box_num;
        } else {
            save_ptr = nram_save;
            save_offset = nram_save_count;
            save_str_num = nram_save_limit_count;
        }
        if (core_limit == 1) {
            if (save_method == 0) {  // score, x1, y1, x2, y2
                __memcpy(save_ptr + save_offset * 5, max_box, 5 * sizeof(NMS_DT), NRAM2NRAM,
                        5 * sizeof(NMS_DT), 5 * sizeof(NMS_DT), 0);
            } else if (save_method == 1) {  // score---, x1---, y1---, x2---, y2---
                __memcpy(save_ptr + save_offset, max_box, 1 * sizeof(NMS_DT), NRAM2NRAM,
                        save_str_num * sizeof(NMS_DT), 1 * sizeof(NMS_DT), 4);
            } else if (save_method == 2) {  // for ssd
                save_ptr[max_index] = max_box[0];
            }
        } else {
            if (coreId == coreDim - 1) {
                if (save_method == 0) {  // score, x1, y1, x2, y2
                __memcpy(save_ptr + save_offset * 5, max_box, 5 * sizeof(NMS_DT), NRAM2NRAM,
                        5 * sizeof(NMS_DT), 5 * sizeof(NMS_DT), 0);
                } else {  // score---, x1---, y1---, x2---, y2---
                __memcpy(save_ptr + save_offset, max_box, 1 * sizeof(NMS_DT), NRAM2NRAM,
                        save_str_num * sizeof(NMS_DT), 1 * sizeof(NMS_DT), 4);
                }
            }
        }
        nram_save_count++;
        output_box_num++;

        // store to sram/gdram --if keep == keepNum
        if (dst != NRAM && output_box_num != 0) {
            mluMemcpyDirection_t store_dir = NRAM2GDRAM;
            if (dst == SRAM) {
                store_dir = NRAM2SRAM;
            } else {
                store_dir = NRAM2GDRAM;
            }

            if (keep == keepNum) {
                if (core_limit == 1) {
                    if (save_method == 0) {  // score, x1, y1, x2, y2
                        __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
                    } else {  // score---, x1---, y1---, x2---, y2---
                        __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                                output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
                    }
                } else {
                    if (coreId == coreDim - 1) {
                        if (save_method == 0) {  // score, x1, y1, x2, y2
                            __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
                        } else {  // score---, x1---, y1---, x2---, y2---
                            __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                                    output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
                        }
                    }
                }
            }
        }      // if dst
        


        /*----- IoU 筛选 -----*/
        for (int i = 0; i <= repeat; i++) {
            if (i == repeat && remain == 0)
                break;
            int seg_len = 0;  // the length every nms compute
            int cpy_len = 0;  // the length every nms memcpy
            seg_len = (i == repeat ?  remain_pad : max_seg_pad);
            cpy_len = (i == repeat ?  remain : max_seg_pad);

            // Load NMS
            if (MODE != 0) {
                score = input_score_ptr;
                x1 = input_x1_ptr;
                y1 = input_y1_ptr;
                x2 = input_x2_ptr;
                y2 = input_y2_ptr;
            } else {
                mluMemcpyDirection_t load_dir = SRAM2NRAM;
                if (src == NRAM) {
                    load_dir = NRAM2NRAM;
                } else if (src == SRAM) {
                    load_dir = SRAM2NRAM;
                } else {
                    load_dir = GDRAM2NRAM;
                }
                __nramset(score, seg_len, 0);
                __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
                //
                __memcpy(x1, input_x1_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
                __memcpy(y1, input_y1_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
                __memcpy(x2, input_x2_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
                __memcpy(y2, input_y2_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                        load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
            }

            // Set the tail to zero when MODE == 1
            if (MODE == 1 && input_stride > input_box_num) {
                __nramset(inter_x1, seg_len, 0);
                int tail_len = input_stride - input_box_num;
                __memcpy(input_score_ptr + input_box_num, inter_x1, tail_len * sizeof(NMS_DT), NRAM2NRAM,
                         tail_len * sizeof(NMS_DT), tail_len * sizeof(NMS_DT), 0);
            }

            /*---- Compute IoU ----*/
            // 计算相交部分的面积
            // area_I = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            __nramset(inter_y1, seg_len, max_box[1]);       // max_x1
            __svmax_relu(inter_x1, x1, inter_y1, seg_len);  // inter_x1 (相交部分左上角横坐标)
            __nramset(inter_y2, seg_len, max_box[3]);       // max_x2
            __svmin_relu(inter_x2, x2, inter_y2, seg_len);  // inter_x2 (相交部分右下角横坐标)
            __bang_sub(inter_x1, inter_x2, inter_x1, seg_len);  // inter_x2 - inter_x1
            __bang_active_relu(inter_x1, inter_x1, seg_len);    // 相交部分的宽度 inter_w | inter_x2 - inter_x1|
            
            __nramset(inter_x2, seg_len, max_box[2]);         // max_y1
            __svmax_relu(inter_y1, y1, inter_x2, seg_len);    // inter_y1
            __nramset(inter_x2, seg_len, max_box[4]);         // max_y2
            __svmin_relu(inter_y2, y2, inter_x2, seg_len);    // inter_y2
            __bang_sub(inter_y1, inter_y2, inter_y1, seg_len);  // inter_y2, inter_y1
            __bang_active_relu(inter_y1, inter_y1, seg_len);    // 相交部分的高度 inter_h
            __bang_mul(inter_x1, inter_x1, inter_y1, seg_len);  // area_I

            // 计算输入框的面积 
            // area = (x2 - x1) * (y2 - y1);
            __bang_sub(inter_y1, x2, x1, seg_len);
            __bang_sub(inter_y2, y2, y1, seg_len);
            __bang_mul(inter_x2, inter_y1, inter_y2, seg_len);  // area

            // 计算相并部分的面积
            // area_U = area + max_area - area_I
            __nramset(inter_y1, seg_len, max_area);
            __bang_add(inter_x2, inter_x2, inter_y1, seg_len);
            __bang_sub(inter_x2, inter_x2, inter_x1, seg_len);  // area_U

            /*---- Select the box ----*/
            // if IoU is greater than threshold, set the score to zero
            __bang_mul_const(inter_x2, inter_x2, thresh_iou, seg_len);
            __bang_gt(inter_x1, inter_x2, inter_x1, seg_len);   // 比较向量化：area_U * thresh > area_I ?
            __bang_mul(score, score, inter_x1, seg_len);        // 置零向量化

            /*---- Update the score ----*/
            if (MODE == 0) {  // do nothing when MODE = 1
                mluMemcpyDirection_t update_dir = NRAM2SRAM;
                if (src == NRAM) {
                    update_dir = NRAM2NRAM;
                } else if (src == SRAM) {
                    update_dir = NRAM2SRAM;
                } else {
                    update_dir = NRAM2GDRAM;
                }
                __memcpy(input_score_ptr + input_offset + i * max_seg_pad, score, cpy_len * sizeof(NMS_DT),
                         update_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
            }
        }   // for repeat
    }       // for keepNum
}



#endif  // _NMS_DETECTION_H_