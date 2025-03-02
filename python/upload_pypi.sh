cp ../README.md ../LICENSE .
rm -rf dist
python3 -m build
python3 -m twine upload dist/*

# Calculate SHA-256 hashes locally
VERSION=$(grep -o '__version__ = "[^"]*"' sglang/version.py | cut -d'"' -f2)
python3 ../scripts/release/calculate_local_hashes.py sglang --version $VERSION --dist-dir dist --output hash.txt
echo "SHA-256 hashes calculated locally and written to hash.txt"

rm -rf README.md LICENSE
