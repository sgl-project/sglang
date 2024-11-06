# these are required for actix
apt-get update
apt-get install -y libssl-dev pkg-config

# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y


# Follow the installation prompts, then reload your shell
. "$HOME/.cargo/env"
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
