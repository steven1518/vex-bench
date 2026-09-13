#!/usr/bin/env bash
# Build language-base + per-(lang,agent) docker images.
#
# Layout:
#   base.<lang>.Dockerfile          → vex-bench-<lang>:latest
#   <lang>.<agent>.Dockerfile       → vex-bench-<lang>-<agent>:latest
#
# Usage:
#   ./build.sh                  # build everything
#   ./build.sh go               # only Go base + Go agents
#   ./build.sh go opencode      # only vex-bench-go + vex-bench-go-opencode

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

LANGS=(go python java)
AGENTS=(codex opencode claude)

want_lang="${1:-}"
want_agent="${2:-}"

build_base() {
    local lang="$1"
    echo ">>> building vex-bench-${lang}"
    docker build -t "vex-bench-${lang}:latest" -f "${HERE}/base.${lang}.Dockerfile" "${HERE}"
}

build_agent() {
    local lang="$1" agent="$2"
    echo ">>> building vex-bench-${lang}-${agent}"
    docker build -t "vex-bench-${lang}-${agent}:latest" \
        -f "${HERE}/${lang}.${agent}.Dockerfile" "${HERE}"
}

for lang in "${LANGS[@]}"; do
    [[ -n "$want_lang" && "$want_lang" != "$lang" ]] && continue
    build_base "$lang"
    for agent in "${AGENTS[@]}"; do
        [[ -n "$want_agent" && "$want_agent" != "$agent" ]] && continue
        build_agent "$lang" "$agent"
    done
done
