#!/bin/bash

# Check if gh is installed before attempting to install it
if ! command -v gh &> /dev/null
then
echo "GitHub CLI not found. Installing now..."
(type -p wget >/dev/null || ( apt update &&  apt install wget -y)) \
&&  mkdir -p -m 755 /etc/apt/keyrings \
&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
&& cat $out |  tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&&  chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&&  mkdir -p -m 755 /etc/apt/sources.list.d \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" |  tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&&  apt update \
&&  apt install gh -y
else
echo "GitHub CLI is already installed. Skipping installation."
fi
