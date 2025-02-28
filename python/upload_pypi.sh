cp ../README.md ../LICENSE .
rm -rf dist
python3 -m build
python3 -m twine upload dist/*

# Extract SHA-256 hashes from PyPI
VERSION=$(grep -o '__version__ = "[^"]*"' sglang/version.py | cut -d'"' -f2)
echo "Waiting for PyPI to process the upload..."
sleep 60
python3 ../scripts/release/extract_pypi_hashes.py sglang --version $VERSION --output hash.txt
echo "SHA-256 hashes extracted to hash.txt"

rm -rf README.md LICENSE
