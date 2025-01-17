.PHONY: tree ln submodule install build clean test format

tree:
	@tree --prune -I "__pycache__|*.egg-info|*.so|build"

submodule:
	@git submodule update --init --recursive

ln: submodule
	@rm -rf build && bear python3 setup.py build

install: submodule
	@pip install -e .

build: submodule
	@export MAX_JOBS=$(nproc) && python3 setup.py bdist_wheel

clean:
	@rm -rf build dist *.egg-info

test:
	@pytest tests/

format:
	@find src tests -name '*.cc' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' | xargs clang-format -i && find src tests -name '*.py' | xargs isort && find src tests -name '*.py' | xargs black
