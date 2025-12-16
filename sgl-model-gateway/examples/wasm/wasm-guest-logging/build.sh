#!/bin/bash
# Build script for WASM guest logging example
# This script simplifies the build process for the WASM middleware component

set -e

echo "Building WASM guest logging example..."

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Please run this script from the wasm-guest-logging directory."
    exit 1
fi

# Check for required tools
command -v cargo >/dev/null 2>&1 || { echo "Error: cargo is required but not installed. Aborting." >&2; exit 1; }

# Check and install wasm32-wasip2 target
echo "Checking for wasm32-wasip2 target..."
if ! rustup target list --installed | grep -q "wasm32-wasip2"; then
    echo "wasm32-wasip2 target not found. Installing..."
    rustup target add wasm32-wasip2
    echo "✓ wasm32-wasip2 target installed"
else
    echo "✓ wasm32-wasip2 target already installed"
fi

# Check for wasm-tools
if ! command -v wasm-tools >/dev/null 2>&1; then
    echo "Error: wasm-tools is required but not installed."
    echo "Install it with: cargo install wasm-tools"
    exit 1
fi

# Build with cargo (wit-bindgen uses cargo, not wasm-pack)
echo "Running cargo build..."
cargo build --target wasm32-wasip2 --release

# Output locations
WASM_MODULE="target/wasm32-wasip2/release/wasm_guest_logging.wasm"
WASM_COMPONENT="target/wasm32-wasip2/release/wasm_guest_logging.component.wasm"

if [ ! -f "$WASM_MODULE" ]; then
    echo "Error: Build failed - WASM module not found"
    exit 1
fi

# Check if the file is already a component
echo "Checking WASM file format..."
if wasm-tools print "$WASM_MODULE" 2>/dev/null | grep -q "^(\s*component"; then
    echo "✓ WASM file is already in component format"
    # Copy to component path for consistency
    cp "$WASM_MODULE" "$WASM_COMPONENT"
else
    # Wrap the WASM module into a component format
    echo "Wrapping WASM module into component format..."
    wasm-tools component new "$WASM_MODULE" -o "$WASM_COMPONENT"
    if [ ! -f "$WASM_COMPONENT" ]; then
        echo "Error: Failed to create component file"
        exit 1
    fi
fi

if [ -f "$WASM_COMPONENT" ]; then
    echo ""
    echo "✓ Build successful!"
    echo "  WASM module: $WASM_MODULE"
    echo "  WASM component: $WASM_COMPONENT"
    echo ""
    echo "Next steps:"
    echo "1. Use the component file ($WASM_COMPONENT) when adding the module"
    echo "2. Prepare the module configuration (see README.md for JSON format)"
    echo "3. Use the API endpoint to add the module (see README.md for details)"
else
    echo "Error: Component file not found"
    exit 1
fi
