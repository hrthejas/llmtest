# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from setuptools import find_packages, setup

extras = {}
extras["quality"] = ["black ~= 22.0", "ruff>=0.0.241", "urllib3<=2.0.0"]
extras["docs_specific"] = ["hf-doc-builder"]
extras["dev"] = extras["quality"] + extras["docs_specific"]
extras["test"] = extras["dev"] + ["pytest", "pytest-xdist", "parameterized", "datasets", "diffusers"]

setup(
    name="llmtest",
    version="0.0.1.dev0",
    description="LLM Loader with local embeddings",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Thejas",
    author_email="h.r.thejas@gmail.com",
    url="https://github.com/hrthejas/llmtest",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.7.0",
    install_requires=[
        "pydantic==1.10.9",
        "protobuf==3.20.3",
        "environs",
        "bitsandbytes==0.39.1",
        "peft",
        "accelerate",
        "einops",
        "safetensors",
        "transformers==4.30.2",
        "xformers",
        "torch",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "tiktoken",
        "unstructured",
        "instructorembedding",
        "faiss-cpu",
        "gradio",
        "IPython",
        "openai==0.28.1",
        "mysql-connector-python",
        "jq",
        "auto_gptq"
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
