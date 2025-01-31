.PHONY: tree ln submodule install build clean rebuild test format

tree:
	@tree --prune -I "__pycache__|*.egg-info|*.so|build|3rdparty|dist"

submodule:
	@git submodule update --init --recursive

ln: submodule
	@rm -rf build && bear python3 setup.py build

install: submodule
	@pip install -e .

build: submodule
	@rm -rf dist/* || true && export MAX_JOBS=$(nproc) && python3 setup.py bdist_wheel && pip3 install dist/*whl --force-reinstall --no-deps

clean:
	@rm -rf build dist *.egg-info

rebuild: clean submodule build
	@echo "Succeed to rebuild"

test:
	@find tests -name "test_*.py" | xargs -n 1 python3

format:
	@find src tests -name '*.cc' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' | xargs clang-format -i && find src tests -name '*.py' | xargs isort && find src tests -name '*.py' | xargs black
