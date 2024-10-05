import nodeResolve from "@rollup/plugin-node-resolve";
import react from "@vitejs/plugin-react";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import { externalizeDeps } from "vite-plugin-externalize-deps";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

import pkg from "./package.json" assert { type: "json" };

const { FIFTYONE_DIR } = process.env;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const dir = __dirname;

function fiftyonePlugin() {
  if (!FIFTYONE_DIR) {
    throw new Error(
      `FIFTYONE_DIR environment variable not set. This is required to resolve @fiftyone/* imports.`
    );
  }

  return {
    name: "fiftyone-bundle-private-packages",
    enforce: "pre",
    resolveId: (source) => {
      if (source.startsWith("@fiftyone")) {
        const pkg = source.split("/")[1];
        const modulePath = `${FIFTYONE_DIR}/app/packages/${pkg}`;
        return this.resolve(modulePath, source, { skipSelf: true });
      }
      return null;
    },
  };
}

export default defineConfig({
  mode: "development",
  plugins: [
    fiftyonePlugin(),
    wasm(),
    topLevelAwait(),
    nodeResolve(),
    react({ jsxRuntime: "classic" }),
    externalizeDeps({
      deps: true,
      devDeps: false,
      useFile: join(process.cwd(), "package.json"),
      // we want to bundle in the following dependencies and not rely on
      // them being available in the global scope
      except: [/^@rerun-io.*/],
    }),
  ],
  build: {
    minify: true,
    lib: {
      entry: join(dir, pkg.main),
      name: pkg.name,
      fileName: (format) => `index.${format}.js`,
      formats: ["umd"],
    },
    rollupOptions: {
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          "jsx-runtime": "jsx",
          "@fiftyone/state": "__fos__",
          "@fiftyone/plugins": "__fop__",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/components": "__foc__",
          "@fiftyone/utilities": "__fou__",
          "@fiftyone/spaces": "__fosp__",
          "@mui/material": "__mui__",
          "styled-components": "__styled__",
        },
      },
    },
  },
  define: {
    "process.env.NODE_ENV": '"development"',
  },
  optimizeDeps: {
    exclude: ["react", "react-dom"],
  },
});
