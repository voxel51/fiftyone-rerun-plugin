import { defineConfig } from "@voxel51/fiftyone-js-plugin-build";
import { dirname } from "path";
import { fileURLToPath } from "url";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const dir = __dirname;

const myPluginThirdPartyDependencies = [/^@rerun-io.*/, "lru-cache"];

export default defineConfig(dir, {
  forceBundleDependencies: myPluginThirdPartyDependencies,
  plugins: [wasm(), topLevelAwait()],
});
