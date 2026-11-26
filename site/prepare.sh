#!/usr/bin/env bash
# Copy the course notes into site/docs so MkDocs can build them.
# The notes stay where they are; site/docs is generated and git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf site/docs
mkdir -p site/docs
cp -R Module\ * site/docs/
cp README.md site/docs/index.md
cp -R site/assets site/docs/assets
