FROM lmsysorg/sglang:dev

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN set -eux; \
    if [ "${HOST_UID}" = "0" ] || [ "${HOST_GID}" = "0" ]; then \
        echo "HOST_UID and HOST_GID must be non-zero" >&2; \
        exit 1; \
    fi; \
    if getent group "${HOST_GID}" >/dev/null; then \
        target_group="$(getent group "${HOST_GID}" | cut -d: -f1)"; \
    elif getent group devuser >/dev/null; then \
        groupmod -g "${HOST_GID}" devuser; \
        target_group=devuser; \
    else \
        groupadd -g "${HOST_GID}" devuser; \
        target_group=devuser; \
    fi; \
    existing_user=""; \
    if getent passwd "${HOST_UID}" >/dev/null; then \
        existing_user="$(getent passwd "${HOST_UID}" | cut -d: -f1)"; \
    fi; \
    if getent passwd devuser >/dev/null; then \
        if [ -n "${existing_user}" ] && [ "${existing_user}" != "devuser" ]; then \
            userdel "${existing_user}"; \
        fi; \
        usermod -u "${HOST_UID}" -g "${HOST_GID}" -d /home/devuser -s /bin/zsh devuser; \
    elif [ -n "${existing_user}" ]; then \
        usermod -l devuser -u "${HOST_UID}" -g "${HOST_GID}" -d /home/devuser -s /bin/zsh "${existing_user}"; \
    else \
        useradd -u "${HOST_UID}" -g "${HOST_GID}" -d /home/devuser -s /bin/zsh devuser; \
    fi; \
    mkdir -p /home/devuser; \
    chown -R devuser:"$(id -gn devuser)" /home/devuser; \
    echo "devuser uses uid=$(id -u devuser), gid=$(id -g devuser), group=${target_group}"

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set up oh-my-zsh for devuser
RUN cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh && \
    cp /root/.zshrc /home/devuser/.zshrc && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc && \
    chown -R devuser:"$(id -gn devuser)" /home/devuser/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:"$(id -gn devuser)" /sgl-workspace

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
