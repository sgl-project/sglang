cp ../README.md ../LICENSE .
rm -rf dist
python3 -m build
python3 -m twine upload dist/*

rm -rf README.md LICENSE
