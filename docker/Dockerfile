FROM vllm/vllm-openai

RUN pip install --upgrade pip
RUN pip install "sglang[all]"
RUN pip uninstall -y triton triton-nightly && pip install --no-deps --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
