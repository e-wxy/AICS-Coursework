#ifndef __SBC_KERNEL_H__
#define __SBC_KERNEL_H__

extern "C" {
    // TODO：完成SBCKernel接口定义
    // __mlu_entry__ void SBCKernel(half* input_data_, half* output_data_, int batch_num_);
    void SBCKernel(half* input_data_, half* output_data_, int batch_num_);

}

#endif  // __SBC_KERNEL_H__

