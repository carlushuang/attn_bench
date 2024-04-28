# attn benchmark
## cudnn v9+
cudnn v9+ improves performance a lot. use cudnn frontend to benchmark the perf
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $PWD:/dockerx nvcr.io/nvidia/pytorch:24.03-py3  /bin/bash
python bench_cudnn_fe.py
```

## trt llm
TODO: the perf is too slow to believe true

follow this link to install environmenent: https://nvidia.github.io/TensorRT-LLM/installation/linux.html
```
python3 bench_trtllm.py
```
