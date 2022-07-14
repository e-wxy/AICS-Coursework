// tensorflow/core/kernels/yolov3_detection_output_op_mlu.h

// Copyright [2018] <Cambricon>
#ifndef TENSORFLOW_CORE_KERNELS_YOLOV3_DETECTION_OUTPUT_OP_MLU_H_
#define TENSORFLOW_CORE_KERNELS_YOLOV3_DETECTION_OUTPUT_OP_MLU_H_
#ifdef CAMBRICON_MLU
#include <memory>
#include <vector>
#include "tensorflow/core/framework/mlu_op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/node_def.pb.h"

#include "tensorflow/stream_executor/mlu/mlu_stream.h"

namespace tensorflow {
template<typename T>
class MLUYolov3DetectionOutputOp: public MLUOpKernel{
    public:
        explicit MLUYolov3DetectionOutputOp(OpKernelConstruction* context):MLUOpKernel(context){
            OP_REQUIRES_OK(context,context->GetAttr("batchNum",&batchNum_));
            OP_REQUIRES_OK(context,context->GetAttr("inputNum",&inputNum_));
            OP_REQUIRES_OK(context,context->GetAttr("classNum",&classNum_));
            OP_REQUIRES_OK(context,context->GetAttr("maskGroupNum",&maskGroupNum_));
            OP_REQUIRES_OK(context,context->GetAttr("maxBoxNum",&maxBoxNum_));
            OP_REQUIRES_OK(context,context->GetAttr("netw",&netw_));
            OP_REQUIRES_OK(context,context->GetAttr("neth",&neth_));
            OP_REQUIRES_OK(context,context->GetAttr("confidence_thresh",&confidence_thresh_));
            OP_REQUIRES_OK(context,context->GetAttr("nms_thresh",&nms_thresh_));
            OP_REQUIRES_OK(context,context->GetAttr("inputWs",&inputWs_));
            OP_REQUIRES_OK(context,context->GetAttr("inputHs",&inputHs_));
            OP_REQUIRES_OK(context,context->GetAttr("biases",&biases_));
        }

        void ComputeOnMLU(OpKernelContext* context) override {
          // auto* stream = context->op_device_context()->mlu_stream();
          // auto* mlustream_exec =
          //    context->op_device_context()->mlu_stream()->parent();
          se::mlu::MLUStream* stream = static_cast<se::mlu::MLUStream*>(
              context->op_device_context()->stream()->implementation());

          Tensor* input0 = const_cast<Tensor*>(&context->input(0));     // YOLOv3的特征图输出
          Tensor* input1 = const_cast<Tensor*>(&context->input(1));
          Tensor* input2 = const_cast<Tensor*>(&context->input(2));
          string op_parameter = context->op_kernel().type_string();
          // MLU_OP_CHECK_UNSUPPORTED(mlustream_exec, op_parameter, context);
          // TODO:参数检查与处理
          int c_arr_data[3] = {255, 255, 255};
          std::vector<int> input0_shape(4, 1);
          std::vector<int> input1_shape(4, 1);
          std::vector<int> input2_shape(4, 1);
          input0_shape[0] = batchNum_;
          input0_shape[1] = c_arr_data[0];
          input0_shape[2] = inputHs_[0];
          input0_shape[3] = inputWs_[0];

          input1_shape[0] = batchNum_;
          input1_shape[1] = c_arr_data[1];
          input1_shape[2] = inputHs_[1];
          input1_shape[3] = inputWs_[1];
          
          input2_shape[0] = batchNum_;
          input2_shape[1] = c_arr_data[2];
          input2_shape[2] = inputHs_[2];
          input2_shape[3] = inputWs_[2];
          
          // TODO:输出形状推断及输出内存分配
          int buffer_size = 255 * (inputHs_[0] * inputWs_[0] +
                                   inputHs_[1] * inputWs_[1] +
                                   inputHs_[2] * inputWs_[2]);
          std::vector<int> buffer_shape = {batchNum_, buffer_size, 1, 1};
          std::vector<int> output_shape(4, 1);
          output_shape[0] = batchNum_;
          output_shape[1] = 7 * maxBoxNum_ + 64;
          Tensor *output;
          Tensor *buffer;
          TensorShape tf_output_shape {output_shape[0], output_shape[1], output_shape[2], output_shape[3]};
          TensorShape tf_buffer_shape {buffer_shape[0], buffer_shape[1], buffer_shape[2], buffer_shape[3]};
          OP_REQUIRES_OK(context, context->allocate_output(0, tf_output_shape, &output));
          OP_REQUIRES_OK(context, context->allocate_output(0, tf_buffer_shape, &buffer));
          
          // TODO:调用MLUStream层接口完成算子计算
          if(output->NumElements() > 0 && buffer->NumElements() > 0) {
            OP_REQUIRES_OK(
                context, 
                stream->Yolov3DetectionOutput(
                    context, input0, input1, input2, batchNum_, inputNum_, classNum_, 
                    maskGroupNum_, maxBoxNum_, netw_, neth_, confidence_thresh_, nms_thresh_, 
                    inputWs_.data(), inputHs_.data(), biases_.data(), output, buffer
                ));
          } else {
            // mlustream_exec->insert_unsupported_op(context, op_parameter);
          }
        }
    private:
    int batchNum_;
    int inputNum_;
    int classNum_;
    int maskGroupNum_;
    int maxBoxNum_;
    int netw_;
    int neth_;
    float confidence_thresh_;
    float nms_thresh_;
    std::vector<int> inputWs_;
    std::vector<int> inputHs_;
    std::vector<float> biases_;
};
}
#endif
#endif
