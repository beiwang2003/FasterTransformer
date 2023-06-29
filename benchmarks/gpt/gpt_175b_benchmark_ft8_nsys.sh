# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

export NVIDIA_TF32_OVERRIDE=0
pipeline_para_size=1
for tensor_para_size in 8 4 2;
do
    total_gpu_count=$((tensor_para_size * pipeline_para_size))

    vocab_size=51200
    
    logdir="profiler/fp8/gpt-TP${tensor_para_size}-PP${pipeline_para_size}-log"
    if [ ! -f ${logdir} ]; then
	mkdir ${logdir} -p
    fi
    
    all_log="${logdir}/all-log.log"
    
    echo -e "| model size | Batch Size | Input length | Output length | Precision | FT latency (ms) |" > $all_log
    echo -e "|:----------:|:----------:|:------------:|:-------------:|:---------:|:---------------:|" >> $all_log

    cat /proc/cpuinfo > ${logdir}/cpuinfo.txt
    nvidia-smi > ${logdir}/gpuinfo.txt

    for model_size in "175b";
    do
	head_num=96
	size_per_head=128
	inter_size=$((head_num * size_per_head * 4))
	num_layer=24

	beam_width=1
	topk=200
	topp=0.95
	temperature=0.5

	for request_batch_size in 1 4 16;
	do
	    for input_length in 128;
	    do
		for request_output_len in 128;
		do
    
		    tmp_log=${logdir}/bs-${request_batch_size}-input_len-${input_length}-output_len-${request_output_len}.log

		    python ../examples/pytorch/gpt/utils/generate_start_ids.py --max_batch_size ${request_batch_size} --max_input_length ${input_length}
		    ./bin/gpt_gemm ${request_batch_size} ${beam_width} ${input_length} ${head_num} ${size_per_head} ${inter_size} ${vocab_size} 4 ${tensor_para_size} 0
		    python ../examples/pytorch/gpt/utils/generate_gpt_config.py \
                           --max_batch_size ${request_batch_size} \
                           --max_seq_len 256 \
                           --beam_width ${beam_width} \
                           --head_num ${head_num} \
                           --size_per_head ${size_per_head} \
                           --inter_size ${inter_size} \
                           --num_layer ${num_layer} \
                           -v 51200 \
                           -topk ${topk} \
                           -topp ${topp} \
                           --tensor_para_size ${tensor_para_size} \
                           --pipeline_para_size ${pipeline_para_size} \
                           -request_batch_size ${request_batch_size} \
                           --request_output_len ${request_output_len}
		    
		    FT_NVTX=ON nsys profile -s none -t cuda,nvtx,osrt --force-overwrite=true -o ${logdir}/bs-${request_batch_size}-int8_mode-${int8_mode}-layers-${num_layer} --capture-range=cudaProfilerApi --capture-range-end=stop  mpirun -n ${total_gpu_count} --allow-run-as-root ./bin/gpt_fp8_example .tmp.config.ini

		    rm .tmp.config.ini
		    
		done # request_output_len
	    done # input_length
	done # request_batch_size
    done # model_size
done # tensor_para_size
