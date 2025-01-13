.PHONY: tree ln install build clean test format

tree:
	@tree --prune -I "__pycache__|*.egg-info|*.so|build"

ln:
	@rm -rf build && cmake . -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_CUDA_COMPILER=nvcc -B build && rm -rf compile_commands.json && ln -s build/compile_commands.json compile_commands.json

install:
	@pip install -e .

build:
	@export MAX_JOBS=$(nproc) && python3 setup.py bdist_wheel

clean:
	@rm -rf build dist *.egg-info

test:
	@pytest tests/

format:
	@find src tests -name '*.cc' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' | xargs clang-format -i && find src tests -name '*.py' | xargs isort && find src tests -name '*.py' | xargs black
