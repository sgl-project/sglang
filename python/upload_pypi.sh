cp ../README.md ../LICENSE .
rm -rf dist
python3 -m build
python3 -m twine upload dist/*

# Calculate SHA-256 hashes locally
VERSION=$(grep -o '__version__ = "[^"]*"' sglang/version.py | cut -d'"' -f2)
python3 ../scripts/release/calculate_local_hashes.py sglang --version $VERSION --dist-dir dist --output hash.txt
echo "SHA-256 hashes calculated locally and written to hash.txt"

# Extract SHA-256 hashes from PyPI as fallback
if [ $? -ne 0 ]; then
  echo "Local hash calculation failed, falling back to PyPI extraction..."
  python3 ../scripts/release/extract_pypi_hashes.py sglang --version $VERSION --output hash.txt --wait --max-wait-time 300
  echo "SHA-256 hashes extracted from PyPI and written to hash.txt"
fi

rm -rf README.md LICENSE
