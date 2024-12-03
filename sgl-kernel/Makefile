.PHONY: tree ln install build clean test

tree:
	@tree --prune -I "__pycache__|*.egg-info|*.so|build"

ln:
	@rm -rf build && cmake . -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_CUDA_COMPILER=nvcc -B build && rm -rf compile_commands.json && ln -s build/compile_commands.json compile_commands.json

install:
	@pip install -e .

build:
	@python3 setup.py bdist_wheel

clean:
	@rm -rf build dist *.egg-info

test:
	@pytest tests/
