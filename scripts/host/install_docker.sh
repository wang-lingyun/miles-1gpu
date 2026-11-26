#!/usr/bin/env bash
# One-time host setup on the GPU box (Ubuntu 24.04): Docker Engine + NVIDIA Container Toolkit, so the pinned Miles
# image can see the GPU. Needs sudo; everything else in this repo runs as your user.
#
#   sudo bash scripts/host/install_docker.sh
#
# Then log out and back in (or `newgrp docker`) so your user can run docker without sudo.
# Note: membership in the docker group is root-equivalent on this machine. That's the normal setup for a single-user
# box; on a shared one, skip the usermod line and run docker through sudo instead.
#
# Sources: https://docs.docker.com/engine/install/ubuntu/
#          https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo: sudo bash $0" >&2
  exit 1
fi
TARGET_USER="${SUDO_USER:-}"

apt-get update
apt-get install -y ca-certificates curl gnupg

# Docker's apt repository
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# NVIDIA's apt repository for the container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

if [ -n "$TARGET_USER" ] && [ "$TARGET_USER" != root ]; then
  usermod -aG docker "$TARGET_USER"
  echo "added $TARGET_USER to the docker group (log out and back in for it to apply)"
fi

echo "--- GPU visible inside a container?"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L
