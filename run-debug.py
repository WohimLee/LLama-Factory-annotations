# 用来调试 llamafactory-cli 命令, 
# 也就是 src/llamafactory/cli.py 里面的 main 函数

import sys
from llamafactory.cli import main

# 模拟命令行参数，比如你想调试 'train' 命令
sys.argv = ["llamafactory-cli", "train", "examples/train_full/llama3_full_sft_ds3.yaml"]

if __name__ == "__main__":
    main()