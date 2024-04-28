import os
import sys

from cuda import cuda
import torch
import tensorrt_llm as tllm
from tensorrt_llm import Tensor
from tensorrt_llm.functional import bert_attention
from tensorrt_llm.plugin.plugin import ContextFMHAType

from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] >= 8, "SDPA operation is only supported on SM80 architecture (Ampere) or above"
# TODO: not consider group mode
def get_fwd_tflops(us, batch, nhead, s_q, s_k, h_q, h_v, causal):
    flop = batch * nhead * (2 * s_q * s_k * h_q + 2 * s_q * s_k * h_v) // (2 if causal else 1)
    return flop / 1e6 / us

b = 1     # batch size
h = 16    # query number of heads
s = 16384 # maximum sequence length
d = 128   # embedding dimension per head

# inner_loop = 1000

def bench_trtllm_bert_attention(dtype, b, h, s, d, use_causal_mask = False):
    builder = tllm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    net.plugin_config.set_bert_attention_plugin(dtype)
    
    #context_fmha_type = ContextFMHAType.enabled_with_fp32_acc
    #context_fmha_type = ContextFMHAType.disabled
    context_fmha_type = ContextFMHAType.enabled
    
    net.plugin_config.set_context_fmha(context_fmha_type)
    
    qkv_tensor = torch.randn((b, s, h * d * 3), dtype=tllm._utils.str_dtype_to_torch(dtype), device='cuda')
    input_lengths_tensor = torch.ones((b,), dtype= torch.int32, device='cuda') * s

    with tllm.net_guard(net):
        network = tllm.default_trtnet()

        qkv = Tensor(name='qkv',
                    shape=tuple(qkv_tensor.shape),
                    dtype=tllm.str_dtype_to_trt(dtype))
    
        input_lengths = Tensor(name = 'input_lengths',
                               shape=tuple(input_lengths_tensor.shape),
                               dtype=tllm.str_dtype_to_trt('int32'))

        #for _ in range(inner_loop):
        outputs = bert_attention(tensor=qkv,
                                     input_lengths=input_lengths,
                                     num_heads = h,
                                     head_size = d,
                                     q_scaling = 1.0)

        outputs.trt_tensor.name = 'output'
        outputs.trt_tensor.dtype = tllm.str_dtype_to_trt(dtype)
        network.mark_output(outputs.trt_tensor)

    # build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
    build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(fp16=(dtype == 'float16')))
    
    output_tensor = torch.zeros((b, s, h * d * 1), dtype = tllm._utils.str_dtype_to_torch(dtype), device='cuda')

    stream = torch.cuda.current_stream()
    feed_dict = {'qkv': qkv_tensor, 'input_lengths': input_lengths_tensor}

    # session = tllm.runtime.Session.from_engine(build_engine())
    _, start = cuda.cuEventCreate(0)
    _, stop = cuda.cuEventCreate(0)
    runtimes = []
    #with peer_access(mapping):
    with TrtRunner(build_engine) as runner:
        for _ in range(10):
            cuda.cuEventRecord(start, stream.cuda_stream)
            for _ in range(5):
                output = runner.infer(feed_dict = feed_dict)
            cuda.cuEventRecord(stop, stream.cuda_stream)
            torch.cuda.synchronize()
            _, ms = cuda.cuEventElapsedTime(start, stop)
            runtimes.append(ms/5)

    time = sorted(runtimes)[0]
    #assert torch.allclose(output, (input * world_size)**inner_loop)
    tflops =  get_fwd_tflops(time*1e3, b, h, s, s, d, d, causal = False)
    print(f'ms:{time}, tflops:{tflops:.3f}')


bench_trtllm_bert_attention('float16', b, h, s, d, use_causal_mask = False)
