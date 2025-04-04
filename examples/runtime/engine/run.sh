#pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
export CUDA_VISIBLE_DEVICES=1,2
python3 send.py > run.log 2>&1
