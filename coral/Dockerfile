FROM lmsysorg/sglang:v0.5.22-cu130

ARG SGLANG_ROOT=/sgl-workspace/sglang

COPY q6_penalties.patch /opt/coral/q6_penalties.patch
RUN cd "$SGLANG_ROOT" \
    && (git apply --check /opt/coral/q6_penalties.patch && git apply /opt/coral/q6_penalties.patch \
        || patch -p1 --forward < /opt/coral/q6_penalties.patch) \
    && test -f python/sglang/srt/speculative/draft_penalties.py

COPY . /opt/coral

ENV PYTHONPATH=/opt/coral

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
