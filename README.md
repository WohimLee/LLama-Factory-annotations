# LLaMA-Factory 调试版本

官方原版仓库: https://github.com/hiyouga/LLaMA-Factory

## 调试
### 1 调试 cli.py
>命令
```sh
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

>launch.json
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run-debug.py",
            "args": [
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "FORCE_TORCHRUN": "true",
                "PATH": "/opt/miniconda/envs/llama39/bin:${env:PATH}" // which torchrun 找到的路径, 否则 process = subprocess.run 找不到命令会报错
            },
            "console": "integratedTerminal",
            "justMyCode": false // 允许调试库代码
        }
    ]
}
```

>run-debug.py
```py
# 用来调试 llamafactory-cli 命令, 
# 也就是 src/llamafactory/cli.py 里面的 main 函数

import sys
from llamafactory.cli import main

sys.argv = ["llamafactory-cli", "train", "examples/train_full/llama3_full_sft_ds3.yaml"]

if __name__ == "__main__":
    main()
```

### 2 调试 launcher.py
>命令
- 从 1 获得
```sh
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
  --master_addr 127.0.0.1 --master_port 20570 \
  /root/datav/nlp/LLama-Factory-annotations/src/llamafactory/launcher.py \
  examples/train_full/llama3_full_sft_ds3.yaml
```

>launch.json
- 使用 `which torchrun` 找到 torchrun 路径
- 然后直接把断点打在 launcher.py 里面就可以了
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            // "module": "llamafactory.cli",
            "program": "/opt/miniconda/envs/llama39/bin/torchrun",  // torchrun 的完整路径
            "args": [
                "--nnodes", "1",
                "--node_rank", "0",
                "--nproc_per_node", "1",
                "--master_addr", "127.0.0.1",
                "--master_port", "20570",
                "/root/datav/nlp/LLama-Factory-annotations/src/llamafactory/launcher.py",  // 脚本路径
                "examples/train_full/llama3_full_sft_ds3.yaml" 
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                // "FORCE_TORCHRUN": "true",
                "PATH": "/opt/miniconda/envs/llama39/bin:${env:PATH}" 
            },
            "subProcess": true,
            "console": "integratedTerminal",
            "justMyCode": false // 允许调试库代码
        }
    ]
}
```


### 3 调试 