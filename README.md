# LLaMA-Factory 调试版本

官方原版仓库: https://github.com/hiyouga/LLaMA-Factory

## 调试
- 最终可用版本
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