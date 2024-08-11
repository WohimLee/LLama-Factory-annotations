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
import re

from setuptools import find_packages, setup


def get_version():
    with open(os.path.join("src", "llamafactory", "extras", "env.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


extra_require = {
    "torch": ["torch>=1.13.1"],
    "torch-npu": ["torch==2.1.0", "torch-npu==2.1.0.post3", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.17.0", "auto-gptq>=0.5.0"],
    "awq": ["autoawq"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3"],
    "galore": ["galore-torch"],
    "badam": ["badam>=1.2.1"],
    "qwen": ["transformers_stream_generator"],
    "modelscope": ["modelscope"],
    "dev": ["ruff", "pytest"],
}

def main():
    # 调用 setup 函数，配置并发布一个 Python 包
    setup(
        name="llamafactory",  # 包的名称
        version=get_version(),  # 包的版本号，从 get_version() 函数中获取
        author="hiyouga",  # 包的作者名称
        author_email="hiyouga" "@" "buaa.edu.cn",  # 作者的电子邮件，字符串连接的方式避免了误用符号"@"
        description="Easy-to-use LLM fine-tuning framework",  # 包的简短描述
        long_description=open("README.md", "r", encoding="utf-8").read(),  # 包的详细描述，从 README.md 文件中读取
        long_description_content_type="text/markdown",  # 长描述的内容类型，这里指定为 Markdown 格式
        keywords=["LLaMA", "BLOOM", "Falcon", "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],  # 包的关键词列表，方便在包的索引中查找
        license="Apache 2.0 License",  # 包使用的许可证，这里是 Apache 2.0 许可证
        url="https://github.com/hiyouga/LLaMA-Factory",  # 包的主页 URL，这里是指向 GitHub 仓库
        package_dir={"": "src"},  # 指定包所在的目录，这里是 src 目录下
        packages=find_packages("src"),  # 自动查找 src 目录下的所有 Python 包
        python_requires=">=3.8.0",  # 指定所需的最低 Python 版本
        install_requires=get_requires(),  # 指定包的依赖，从 get_requires() 函数中获取
        extras_require=extra_require,  # 定义额外的可选依赖
        # 执行入口函数：输入 llamafactory-cli 时，
        # Python 会根据包的 setup() 函数中定义的 entry_points 参数，
        # 找到 llamafactory.cli:main 这个入口点并执行。
        # 这表示程序会执行 llamafactory 包中 cli.py 文件中的 main() 函数。
        entry_points={"console_scripts": ["llamafactory-cli = llamafactory.cli:main"]},  
        classifiers=[
            "Development Status :: 4 - Beta",  # 开发状态，表明该包处于 Beta 测试阶段
            "Intended Audience :: Developers",  # 目标用户是开发者
            "Intended Audience :: Education",  # 目标用户包括教育领域的人士
            "Intended Audience :: Science/Research",  # 目标用户还包括科学研究人员
            "License :: OSI Approved :: Apache Software License",  # 表明该包的许可证是经过 OSI 批准的 Apache 软件许可证
            "Operating System :: OS Independent",  # 该包与操作系统无关，可以在任何操作系统上运行
            "Programming Language :: Python :: 3",  # 该包使用 Python 3 编写
            "Programming Language :: Python :: 3.8",  # 该包兼容 Python 3.8
            "Programming Language :: Python :: 3.9",  # 该包兼容 Python 3.9
            "Programming Language :: Python :: 3.10",  # 该包兼容 Python 3.10
            "Programming Language :: Python :: 3.11",  # 该包兼容 Python 3.11
            "Topic :: Scientific/Engineering :: Artificial Intelligence",  # 该包的主题是科学/工程领域的人工智能
        ],
    )


if __name__ == "__main__":
    main()
