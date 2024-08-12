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
            // "module": "llamafactory.cli",
            "program": "${workspaceFolder}/run.py",
            "args": [
                // "train", 
                // "examples/train_lora/llama3_lora_sft_ds3.yaml",
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "FORCE_TORCHRUN": "true",
                "PATH": "/opt/miniconda/envs/llama39/bin:${env:PATH}"
                // "PYTHONPATH": "${workspaceFolder}",
                // "CUDA_VISIBLE_DEVICES" : "0"
                // "PYTHONWARNINGS": "ignore"
            },
            "console": "integratedTerminal",
            "justMyCode": false // 允许调试库代码
        }
    ]
}
```