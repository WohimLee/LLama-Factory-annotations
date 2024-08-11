# Copyright 2024 the LlamaFactory team.
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

import os
import random
import subprocess
import sys
from enum import Enum, unique

from . import launcher
from .api.app import run_api
from .chat.chat_model import run_chat
from .eval.evaluator import run_eval
from .extras.env import VERSION, print_env
from .extras.logging import get_logger
from .extras.misc import get_device_count
from .train.tuner import export_model, run_exp
from .webui.interface import run_web_demo, run_web_ui


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + "| Welcome to LLaMA Factory, version {}".format(VERSION)
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"

def main():
    # 从命令行参数中获取第一个参数作为命令，如果没有参数，则使用 HELP 作为默认命令
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP

    # 如果命令是 API，则调用 run_api 函数
    if command == Command.API:
        run_api()

    # 如果命令是 CHAT，则调用 run_chat 函数
    elif command == Command.CHAT:
        run_chat()

    # 如果命令是 ENV，则调用 print_env 函数，输出环境信息
    elif command == Command.ENV:
        print_env()

    # 如果命令是 EVAL，则调用 run_eval 函数，执行评估
    elif command == Command.EVAL:
        run_eval()

    # 如果命令是 EXPORT，则调用 export_model 函数，导出模型
    elif command == Command.EXPORT:
        export_model()

    # 如果命令是 TRAIN，则执行模型训练逻辑
    elif command == Command.TRAIN:
        # 检查环境变量 FORCE_TORCHRUN 是否设置为 true 或 1，决定是否强制使用 torchrun 进行分布式训练
        force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        
        # 如果强制使用 torchrun 或者设备数量大于 1，则初始化分布式训练
        if force_torchrun or get_device_count() > 1:
            # 获取 MASTER_ADDR 和 MASTER_PORT 环境变量，默认值分别为 127.0.0.1 和一个随机端口
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
            
            # 打印初始化分布式任务的信息
            logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
            
            # 使用 subprocess.run 执行 torchrun 命令，进行分布式训练
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                ).format(
                    nnodes=os.environ.get("NNODES", "1"),  # 获取节点数量，默认为1
                    node_rank=os.environ.get("RANK", "0"),  # 获取当前节点的 rank，默认为0
                    nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),  # 获取每个节点的进程数，默认为设备数量
                    master_addr=master_addr,  # 主节点的地址
                    master_port=master_port,  # 主节点的端口
                    file_name=launcher.__file__,  # 启动脚本的文件名
                    args=" ".join(sys.argv[1:]),  # 传递给启动脚本的其他参数
                ),
                shell=True,
            )
            # 退出并返回 torchrun 命令的返回码
            sys.exit(process.returncode)
        else:
            # 如果不需要分布式训练，则调用 run_exp 函数，执行单机实验
            run_exp()

    # 如果命令是 WEBDEMO，则调用 run_web_demo 函数，启动 Web demo
    elif command == Command.WEBDEMO:
        run_web_demo()

    # 如果命令是 WEBUI，则调用 run_web_ui 函数，启动 Web UI
    elif command == Command.WEBUI:
        run_web_ui()

    # 如果命令是 VER，则打印版本信息
    elif command == Command.VER:
        print(WELCOME)

    # 如果命令是 HELP，则打印帮助信息
    elif command == Command.HELP:
        print(USAGE)

    # 如果输入了未知命令，抛出 NotImplementedError 异常
    else:
        raise NotImplementedError("Unknown command: {}".format(command))

