#!/usr/bin/env bash
# Optional one-time host setup (sudo), for a box shared with untrusted accounts: stop local non-root processes from opening TCP connections into Docker's bridge
# networks (172.16.0.0/12, Docker's default address pools).
#
# Why: when W&B is on, the training container needs a network, and then Ray (GCS :6379, client server :10001, worker
# gRPC ports) and SGLang listen on the container's bridge address without authentication. Docker doesn't route the
# LAN to that address, but every local account can reach it, and Ray's client port runs code as root in the
# container, which holds /data and WANDB_API_KEY. With this rule, only root (dockerd) can open connections there.
# Containers' own outbound traffic (W&B uploads) is forwarded, not host-originated, so it is unaffected, as are
# `docker exec` and `docker logs`.
#
#   sudo bash scripts/host/guard_docker_bridge.sh             # install a systemd unit that applies it at boot, and apply now
#   sudo bash scripts/host/guard_docker_bridge.sh --remove    # undo
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo: sudo bash $0" >&2
  exit 1
fi

UNIT=/etc/systemd/system/docker-bridge-guard.service
RULE=(OUTPUT -d 172.16.0.0/12 -p tcp -m owner ! --uid-owner 0 -m conntrack --ctstate NEW -j REJECT --reject-with tcp-reset)

if [ "${1:-}" = --remove ]; then
  systemctl disable --now docker-bridge-guard.service 2>/dev/null || true
  rm -f "$UNIT"
  while iptables -C "${RULE[@]}" 2>/dev/null; do iptables -D "${RULE[@]}"; done
  systemctl daemon-reload
  echo "removed"
  exit 0
fi

cat > "$UNIT" <<EOF
[Unit]
Description=Block non-root local TCP connections into Docker bridge networks (Ray/SGLang in training containers)
After=network-pre.target docker.service
Wants=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'iptables -C ${RULE[*]} 2>/dev/null || iptables -I ${RULE[*]}'
ExecStop=/bin/sh -c 'while iptables -C ${RULE[*]} 2>/dev/null; do iptables -D ${RULE[*]}; done'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now docker-bridge-guard.service
iptables -S OUTPUT | grep -- "-d 172.16.0.0/12" && echo "active"
