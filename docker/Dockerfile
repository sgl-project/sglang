ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/ \
    NVSHMEM_DIR=/sgl-workspace/nvshmem/install

# Set timezone and install all packages
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
 && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
 && apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common netcat-openbsd kmod unzip openssh-server \
    curl wget lsof zsh ccache tmux htop git-lfs tree \
    python3 python3-pip python3-dev libpython3-dev \
    build-essential cmake \
    libopenmpi-dev libnuma1 libnuma-dev \
    libibverbs-dev libibverbs1 libibumad3 \
    librdmacm1 libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    ibverbs-providers infiniband-diags perftest \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    nvidia-dkms-550 \
    devscripts debhelper fakeroot dkms check libsubunit0 libsubunit-dev \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v2.4.4 \
 && cd gdrcopy/packages \
 && CUDA=/usr/local/cuda ./build-deb-packages.sh \
 && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
 && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Clone and install SGLang
WORKDIR /sgl-workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel html5lib six \
 && git clone --depth=1 https://github.com/sgl-project/sglang.git \
 && cd sglang \
 && case "$CUDA_VERSION" in \
      12.6.1) CUINDEX=126 ;; \
      12.8.1) CUINDEX=128 ;; \
      *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
 && python3 -m pip install --no-cache-dir -e "python[${BUILD_TYPE}]" --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
 && if [ "$CUDA_VERSION" = "12.8.1" ]; then \
      python3 -m pip install --no-cache-dir nvidia-nccl-cu12==2.27.3 --force-reinstall --no-deps ; \
      python3 -m pip install --no-cache-dir https://github.com/sgl-project/whl/releases/download/v0.2.0/sgl_kernel-0.2.0+cu128-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall --no-deps ; \
    fi

# Build and install NVSHMEM + DeepEP
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz \
 && git clone https://github.com/deepseek-ai/DeepEP.git \
 && tar -xf nvshmem_src_3.2.5-1.txz && mv nvshmem_src nvshmem \
 && cd nvshmem \
 && git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch \
 && sed -i '1i#include <unistd.h>' examples/moe_shuffle.cu \
 && rm -f /sgl-workspace/nvshmem_src_3.2.5-1.txz \
 && NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_DIR} -DCMAKE_CUDA_ARCHITECTURES=90 \
 && cmake --build build --target install -j \
 && cd /sgl-workspace/DeepEP \
 && NVSHMEM_DIR=${NVSHMEM_DIR} pip install .

# Python tools
RUN python3 -m pip install --no-cache-dir \
    datamodel_code_generator \
    mooncake_transfer_engine==0.3.4.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core

# Install development tools and utilities
RUN apt-get update && apt-get install -y \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    wget \
    curl \
    locales \
    lsof \
    git \
    git-lfs \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    unzip \
    pkg-config \
    libssl-dev \
    bear \
    ccache \
    less \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt update -y \
    && apt install nsight-systems-cli -y

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install minimal Python packages
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pytest \
    black \
    isort \
    icdiff \
    scikit_build_core \
    uv \
    pre-commit

# Install diff-so-fancy
RUN curl -LSso /usr/local/bin/diff-so-fancy https://github.com/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /usr/local/bin/diff-so-fancy

# Install clang-format
RUN curl -LSso /usr/local/bin/clang-format https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /usr/local/bin/clang-format

# Install clangd
RUN curl -L https://github.com/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip -o clangd.zip \
    && unzip clangd.zip \
    && cp -r clangd_18.1.3/bin/* /usr/local/bin/ \
    && cp -r clangd_18.1.3/lib/* /usr/local/lib/ \
    && rm -rf clangd_18.1.3 clangd.zip

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.1/cmake-3.31.1-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.31.1-linux-x86_64.tar.gz \
    && cp -r cmake-3.31.1-linux-x86_64/bin/* /usr/local/bin/ \
    && cp -r cmake-3.31.1-linux-x86_64/share/* /usr/local/share/ \
    && rm -rf cmake-3.31.1-linux-x86_64 cmake-3.31.1-linux-x86_64.tar.gz

# Add yank script
COPY --chown=root:root <<-"EOF" /usr/local/bin/yank
#!/bin/bash
put() {
  esc=$1
  test -n "$TMUX" -o -z "${TERM##screen*}" && esc="\033Ptmux;\033$esc\033\\"
  printf "$esc"
}
put "\033]52;c;!\a"
buf=$( cat "$@" )
len=$( printf %s "$buf" | wc -c ) max=74994
test $len -gt $max && echo "$0: input is $(( len - max )) bytes too long" >&2
put "\033]52;c;$( printf %s "$buf" | head -c $max | base64 | tr -d '\r\n' )\a"
test -n "$TMUX" && tmux set-buffer "$buf" ||:
EOF

RUN chmod +x /usr/local/bin/yank

# Install oh-my-zsh and plugins
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure Vim
COPY --chown=root:root <<-"EOF" /root/.vimrc
function! Yank(text) abort
  let escape = system('yank', a:text)
  if v:shell_error
    echoerr escape
  else
    call writefile([escape], '/dev/tty', 'b')
  endif
endfunction

noremap <silent> <Leader>y y:<C-U>call Yank(@0)<CR>

" automatically run yank(1) whenever yanking in Vim
function! CopyYank() abort
  call Yank(join(v:event.regcontents, "\n"))
endfunction

autocmd TextYankPost * call CopyYank()

" Basic settings
set number
syntax on
set mouse=a
filetype indent on

" Indentation
set autoindent nosmartindent
set smarttab
set expandtab
set shiftwidth=4
set softtabstop=4

" Visual guides
set colorcolumn=120
highlight ColorColumn ctermbg=5

" Status line
set laststatus=2
set statusline=%<%f\ %h%m%r%=%{\"[\".(&fenc==\"\"?&enc:&fenc).((exists(\"+bomb\")\ &&\ &bomb)?\",B\":\"\").\"]\ \"}%k\ %-14.(%l,%c%V%)\ %P

" Backspace behavior
set backspace=2

" Encoding
set encoding=utf-8
set fileencoding=utf-8
EOF

# Configure tmux
COPY --chown=root:root <<-"EOF" /root/.tmux.conf
# Pane border styling
set -g pane-border-style fg='#742727',bg=black
set -g pane-active-border-style fg=red,bg=black

# Status bar styling
set -g status-style bg='#0C8A92',fg=black

# Change prefix key to backtick
set-option -g prefix `
unbind C-b
bind-key ` send-prefix

# Split panes using - and = with current path
unbind '"'
bind - splitw -v -c '#{pane_current_path}'
unbind '%'
bind = splitw -h -c '#{pane_current_path}'

# Vi mode settings
bind-key -T copy-mode-vi Y send-keys -X copy-pipe 'yank > #{pane_tty}'
set-window-option -g mode-keys vi

# Other settings
set-option -g escape-time 0
set-option -g base-index 1
set-window-option -g mouse on
EOF

# Configure Git
RUN git config --global core.editor "vim" \
    && git config --global core.whitespace "fix,-indent-with-non-tab,trailing-space,cr-at-eol" \
    && git config --global core.pager "diff-so-fancy | less --tabs=4 -RFX" \
    && git config --global color.ui true \
    && git config --global color."diff-highlight".oldNormal "red bold" \
    && git config --global color."diff-highlight".oldHighlight "red bold 52" \
    && git config --global color."diff-highlight".newNormal "green bold" \
    && git config --global color."diff-highlight".newHighlight "green bold 22" \
    && git config --global color.diff.meta "11" \
    && git config --global color.diff.frag "magenta bold" \
    && git config --global color.diff.commit "yellow bold" \
    && git config --global color.diff.old "red bold" \
    && git config --global color.diff.new "green bold" \
    && git config --global color.diff.whitespace "red reverse" \
    && git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset - %s %Cgreen(%cr) %C(bold blue)<%an>%Creset%C(auto)%d%Creset' --abbrev-commit --" \
    && git config --global http.sslVerify false \
    && git config --global pull.rebase true

# Configure zsh
COPY --chown=root:root <<-"EOF" /root/.zshrc
export ZSH="/root/.oh-my-zsh"

# Theme
ZSH_THEME="robbyrussell"

# Plugins
plugins=(
    git
    z
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias vi='vim'

# Enhanced history
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS
setopt INC_APPEND_HISTORY
EOF

RUN set -euxo ; \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Set workspace directory
WORKDIR /sgl-workspace/sglang
