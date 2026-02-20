#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build a versioned local release artifact for the FiftyOne Rerun plugin.

Usage:
  ./scripts/build-release-local.sh [--zip] <rerun_version>

Example:
  ./scripts/build-release-local.sh 0.29.2
  ./scripts/build-release-local.sh --zip 0.29.2

Environment overrides:
  FIFTYONE_REPO_URL     (default: https://github.com/voxel51/fiftyone.git)
  FIFTYONE_REPO_BRANCH  (default: main)
USAGE
}

zip_output=false
version=""
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --zip)
      zip_output=true
      ;;
    -*)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$version" ]]; then
        echo "Only one rerun version argument is allowed" >&2
        usage
        exit 1
      fi
      version="$arg"
      ;;
  esac
done

if [[ -z "$version" ]]; then
  usage
  exit 1
fi

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]]; then
  echo "Invalid version: $version" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fiftyone_repo_url="${FIFTYONE_REPO_URL:-https://github.com/voxel51/fiftyone.git}"
fiftyone_repo_branch="${FIFTYONE_REPO_BRANCH:-main}"
fiftyone_dir="$(cd "$repo_root/.." && pwd)/fiftyone"

if [[ ! -d "$fiftyone_dir/.git" ]]; then
  echo "Cloning FiftyOne into $fiftyone_dir"
  git clone --depth 1 --branch "$fiftyone_repo_branch" "$fiftyone_repo_url" "$fiftyone_dir"
else
  echo "Using existing FiftyOne checkout at $fiftyone_dir"
fi

if command -v corepack >/dev/null 2>&1; then
  corepack enable
fi

echo "Setting plugin and dependency versions to $version"
RERUN_VERSION="$version" node -e '
  const fs = require("fs");
  const version = process.env.RERUN_VERSION;
  const pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
  pkg.version = version;
  pkg.dependencies["@rerun-io/web-viewer"] = `^${version}`;
  fs.writeFileSync("package.json", JSON.stringify(pkg, null, 2) + "\n");
'

if sed --version >/dev/null 2>&1; then
  sed -i "s/^version: .*/version: ${version}/" fiftyone.yaml
else
  sed -i '' "s/^version: .*/version: ${version}/" fiftyone.yaml
fi

printf 'export FIFTYONE_DIR=%s\n' "$fiftyone_dir" > .env.dev

echo "Installing dependencies"
yarn install --no-immutable

echo "Building plugin"
yarn build

artifact_root_dir="dist_artifacts"
artifact_dir="${artifact_root_dir}/fiftyone-rerun-plugin-${version}"
rm -rf "$artifact_dir"
mkdir -p "$artifact_dir"
cp -r dist "$artifact_dir/"
cp fiftyone.yaml "$artifact_dir/"

echo
echo "Created artifact directory: $repo_root/$artifact_dir"

if [[ "$zip_output" == true ]]; then
  output_dir="dist"
  artifact_name="fiftyone-rerun-plugin-${version}.zip"
  artifact_path="${output_dir}/${artifact_name}"
  mkdir -p "$output_dir"

  (
    cd "$artifact_root_dir"
    rm -f "../${artifact_path}"
    zip -r "../${artifact_path}" "fiftyone-rerun-plugin-${version}"
  )

  echo "Created zip artifact: $repo_root/$artifact_path"
fi
