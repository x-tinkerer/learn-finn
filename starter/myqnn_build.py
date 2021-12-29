import os

from finn.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow import (
    build_dataflow_cfg
)

from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hls,
    step_create_dataflow_partition,
)

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)

# def build_dataflow_directory(path_to_cfg_dir: str):
    # the Entry Point
# def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
# def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    # run the steps
    # model = transform_step(model, cfg)
# def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig):

#path_to_cfg_dir ="/home/wenjun/xlnx/finnstart/slim_gray_qnn/models"
path_to_cfg_dir ="./models"
json_filename = path_to_cfg_dir + "/dataflow_build_config.json"

"""
step: step_tidy_up [1/18]
step: step_pre_streamline [2/18]
step: step_streamline [3/18]
step: step_convert_to_hls [4/18]
step: step_convert_final_layers [5/18]
step: step_create_dataflow_partition [6/18]
"""
current_runing_step = 0
onnx_filename = path_to_cfg_dir + "/starter_qnn-ready.onnx"

qonnx_to_finn_onnx = onnx_filename
tidy_up_onnx = path_to_cfg_dir + "/intermediate_models/" + "step_qonnx_to_finn.onnx"
streamline_onnx = path_to_cfg_dir + "/intermediate_models/" + "step_tidy_up.onnx"
convert_to_hls_onnx = path_to_cfg_dir + "/intermediate_models/" + "step_streamline.onnx"
create_dataflow_partition_onnx = path_to_cfg_dir + "/intermediate_models/" + "step_convert_to_hls.onnx"


os.environ["FINN_BUILD_DIR"]="./build"

with open(json_filename, "r") as f:
        json_str = f.read()
build_cfg = DataflowBuildConfig.from_json(json_str)

if(current_runing_step == 0):
    # --- this will run all step ---
    ret = build_dataflow_cfg(onnx_filename, build_cfg)

# step_tidy_up
if(current_runing_step == 1):
    model = ModelWrapper(qonnx_to_finn_onnx)
    ret = step_qonnx_to_finn(model, build_cfg)    

if(current_runing_step == 2):
    model = ModelWrapper(tidy_up_onnx)
    ret = step_tidy_up(model, build_cfg) 
    
if(current_runing_step == 3):
    model = ModelWrapper(streamline_onnx)
    ret = step_streamline(model, build_cfg)
    
if(current_runing_step == 4):
    # run step_convert_to_hls
    model = ModelWrapper(convert_to_hls_onnx)
    step_convert_to_hls(model, build_cfg)

if(current_runing_step == 5):
    # run step_create_dataflow_partition
    model = ModelWrapper(create_dataflow_partition_onnx)
    step_create_dataflow_partition(model, build_cfg)
