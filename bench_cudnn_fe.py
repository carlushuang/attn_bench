import cudnn
import torch
import math
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# TODO: not consider group mode
def get_fwd_tflops(us, batch, nhead, s_q, s_k, h_q, h_v, causal):
    flop = batch * nhead * (2 * s_q * s_k * h_q + 2 * s_q * s_k * h_v) // (2 if causal else 1)
    return flop / 1e6 / us

def get_bwd_tflops(us, batch, nhead, s_q, s_k, h_q, h_v, causal):
    flop = 2.5 * batch * nhead * (2 * s_q * s_k * h_q + 2 * s_q * s_k * h_v) // (2 if causal else 1)
    return flop / 1e6 / us

def get_fwd_bwd_tflops(us, batch, nhead, s_q, s_k, h_q, h_v, causal):
    # simply is fwd+bwd
    flop = 3.5 * batch * nhead * (2 * s_q * s_k * h_q + 2 * s_q * s_k * h_v) // (2 if causal else 1)
    return flop / 1e6 / us

torch.manual_seed(42)
handle = cudnn.create_handle()

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] >= 8, "SDPA operation is only supported on SM80 architecture (Ampere) or above"
assert cudnn.backend_version() >= 8903, "SDPA operation is only supported cuDNN version 8.9.3 or above"

b = 1     # batch size
h = 16    # query number of heads
s = 16384 # maximum sequence length
d = 128   # embedding dimension per head

attn_scale = 1.0 / math.sqrt(d)

def bench_cudnn_inference(handle, b, h, s, d, use_causal_mask = False):
    attn_scale = 1.0 / math.sqrt(d)

    dims = (b, h, s, d)
    strides = (s * h * d, d, h * d, 1)

    q_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    k_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    v_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    o_gpu = torch.empty(b * s * h * d).half().cuda().as_strided(dims, strides)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    # the second return for the stats tensor is used for training only.
    # causal mask is enabled
    o, _ = graph.sdpa(
        name="sdpa",
        q=q, k=k, v=v,
        is_inference=True,
        attn_scale=attn_scale,
        use_causal_mask=use_causal_mask,
    )

    o.set_output(True).set_dim(dims).set_stride(strides)

    # build the graph
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    time=benchmark_torch_function_in_microseconds(graph.execute, variant_pack, workspace)
    tflops = get_fwd_tflops(time, b, h, s, s, d, d, use_causal_mask)
    print(f"[cudnn inference] b:{b}, h:{h}, s:{s}, d:{d}, causal:{use_causal_mask}, {time:.3f} us, tflops:{tflops:.3f}")


def bench_cudnn_training(handle, b, h, s, d, use_causal_mask = False):
    # The tensors will have non-interleaved
    # BSHD (batch, sequence_length, num_head, dims_per_head) physical tensor layout
    # BHSD (batch, num_head, sequence_length, dims_per_head) logical tensor layout
    dims = (b, h, s, d)
    strides = (s * h * d, d, h * d, 1)

    q_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    k_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    v_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
    o_gpu = torch.empty(b * s * h * d).half().cuda().as_strided(dims, strides)
    stats_gpu = torch.empty(b, h, s, 1).float().cuda()
    
    # note: torch 'like' preserves the strided layout
    dQ_gpu = torch.empty_like(q_gpu)
    dK_gpu = torch.empty_like(k_gpu)
    dV_gpu = torch.empty_like(v_gpu)
    dO_gpu = torch.randn_like(o_gpu)

    # graph forward
    graph_forward = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_forward = graph_forward.tensor_like(q_gpu)
    k_forward = graph_forward.tensor_like(k_gpu)
    v_forward = graph_forward.tensor_like(v_gpu)

    # training mode in enabled with is_inference=False
    # causal mask is enabled
    o_forward, stats_forward = graph_forward.sdpa(
        name="sdpa",
        q=q_forward, k=k_forward, v=v_forward,
        is_inference=False,
        attn_scale=attn_scale,
        use_causal_mask=use_causal_mask,
    )

    o_forward.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())
    stats_forward.set_output(True).set_dim(stats_gpu.size()).set_stride(stats_gpu.stride())
    stats_forward.set_data_type(cudnn.data_type.FLOAT)

    graph_forward.validate()
    graph_forward.build_operation_graph()
    graph_forward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph_forward.check_support()
    graph_forward.build_plans()
    
    # graph backward
    graph_backward = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_backward = graph_backward.tensor_like(q_gpu)
    k_backward = graph_backward.tensor_like(k_gpu)
    v_backward = graph_backward.tensor_like(v_gpu)
    o_backward = graph_backward.tensor_like(o_gpu)
    dO_backward = graph_backward.tensor_like(dO_gpu)
    stats_backward = graph_backward.tensor_like(stats_gpu)

    dQ_backward, dK_backward, dV_backward = graph_backward.sdpa_backward(
        name="sdpa_backward",
        q=q_backward, k=k_backward, v=v_backward,
        o=o_backward, dO=dO_backward, stats=stats_backward,
        attn_scale=attn_scale,
        use_causal_mask=use_causal_mask,
    )

    dQ_backward.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
    dK_backward.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
    dV_backward.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())

    graph_backward.validate()
    graph_backward.build_operation_graph()
    graph_backward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph_backward.check_support()
    graph_backward.build_plans()
    
    workspace_size = max(
        graph_forward.get_workspace_size(),
        graph_backward.get_workspace_size(),
    )
    workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)
    
    variant_pack_forward = {
        q_forward: q_gpu,
        k_forward: k_gpu,
        v_forward: v_gpu,
        o_forward: o_gpu,
        stats_forward: stats_gpu,
    }

    time_fwd=benchmark_torch_function_in_microseconds(graph_forward.execute, variant_pack_forward, workspace)
    tflops_fwd = get_fwd_tflops(time_fwd, b, h, s, s, d, d, use_causal_mask)
    
    
    variant_pack_backward = {
        q_backward: q_gpu,
        k_backward: k_gpu,
        v_backward: v_gpu,
        o_backward: o_gpu,
        dO_backward: dO_gpu,
        stats_backward: stats_gpu,
        dQ_backward: dQ_gpu,
        dK_backward: dK_gpu,
        dV_backward: dV_gpu,
    }
    time_bwd=benchmark_torch_function_in_microseconds(graph_backward.execute, variant_pack_backward, workspace)
    tflops_bwd = get_bwd_tflops(time_bwd, b, h, s, s, d, d, use_causal_mask)

    print(f"[cudnn training] b:{b}, h:{h}, s:{s}, d:{d}, causal:{use_causal_mask}, fwd:{time_fwd:.3f} us, tflops:{tflops_fwd:.3f}, bwd:{time_bwd:.3f} us, tflops:{tflops_bwd:.3f}")



bench_cudnn_inference(handle, b, h, s, d, False)
bench_cudnn_training(handle, b, h, s, d, False)