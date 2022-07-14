// tensorflow/stream_executor/mlu/mlu_api/ops/yolov3detectionoutput.cc

#if CAMBRICON_MLU
#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"

namespace stream_executor {
namespace mlu {
namespace ops {


Status MLUYolov3DetectionOutput::CreateMLUOp(std::vector<MLUTensor*> &inputs, \
    std::vector<MLUTensor*> &outputs, void *param) {
    // TODO:补齐create函数实现
    TF_PARAMS_CHECK(inputs.size() > 0, "Missing input");
    TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");
    MLUBaseOp *op_ptr = nullptr;
    MLUTensor *input0 = inputs.at(0);
    MLUTensor *input1 = inputs.at(1);
    MLUTensor *input2 = inputs.at(2);
    MLUTensor *output = outputs.at(0);
    MLUTensor *buffer = outputs.at(1);

    int batchNum = ((MLUYolov3DetectionOutputOpParam*)param)->batchNum_;
    int inputNum = ((MLUYolov3DetectionOutputOpParam*)param)->inputNum_;
    int classNum = ((MLUYolov3DetectionOutputOpParam*)param)->classNum_;
    int maskGroupNum = ((MLUYolov3DetectionOutputOpParam*)param)->maskGroupNum_;
    int maxBoxNum = ((MLUYolov3DetectionOutputOpParam*)param)->maxBoxNum_;
    int netw = ((MLUYolov3DetectionOutputOpParam*)param)->netw_;
    int neth = ((MLUYolov3DetectionOutputOpParam*)param)->neth_;
    float confidence_thresh = ((MLUYolov3DetectionOutputOpParam*)param)->confidence_thresh_;
    float nms_thresh = ((MLUYolov3DetectionOutputOpParam*)param)->nms_thresh_;
    int *inputWs = ((MLUYolov3DetectionOutputOpParam*)param)->inputWs_;
    int *inputHs = ((MLUYolov3DetectionOutputOpParam*)param)->inputHs_;
    float *biases = ((MLUYolov3DetectionOutputOpParam*)param)->biases_;

    cnmlPluginYolov3DetectionOutputOpParam_t mlu_param;
    // const int num_anchors = 3;
    cnmlCoreVersion_t core_version = CNML_MLU270;

    // 调用 cnmlCreatePluginYolov3DetectionOutputOpParam
    cnmlCreatePluginYolov3DetectionOutputOpParam(
        &mlu_param, 
        batchNum, inputNum, classNum, 
        maskGroupNum, maxBoxNum, 
        // num_anchors, maxBoxNum, 
        netw, neth, 
        confidence_thresh, nms_thresh, 
        core_version,
        inputWs, inputHs, biases
    );

    std::vector<MLUTensor*> input_tensors = {input0, input1, input2};
    std::vector<MLUTensor*> output_tensors = {output, buffer};
    // 调用 CreateYolov3DetectionOutputOp
    TF_STATUS_CHECK(lib::CreateYolov3DetectionOutputOp(
        &op_ptr, input_tensors.data(), output_tensors.data(), mlu_param));

    base_ops_.push_back(op_ptr);

    return Status::OK();
}

Status MLUYolov3DetectionOutput::Compute(const std::vector<void *> &inputs,
    const std::vector<void *> &outputs, cnrtQueue_t queue) { 
    //TODO: 补齐compute函数实现
    int num_input = inputs.size();
    int num_output = outputs.size();
    assert(num_input == 3);
    assert(num_output == 2);
    TF_STATUS_CHECK(lib::ComputeYolov3DetectionOutputOp(
        base_ops_.at(0), queue,
        const_cast<void**>(inputs.data()),
        num_input,
        const_cast<void**>(outputs.data()),
        num_output
    ));

    // delete sync queue after delay copy
    cnrtSyncQueue(queue);

    return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor
#endif  // CAMBRICON_MLU
