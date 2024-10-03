import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import nodeResolve from "@rollup/plugin-node-resolve";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { viteExternalsPlugin } from "vite-plugin-externals";
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
    name: "fiftyone-rollup",
    resolveId: {
      order: "pre",
      async handler(source) {
        if (source.startsWith("@fiftyone")) {
          const pkg = source.split("/")[1];
          const modulePath = `${FIFTYONE_DIR}/app/packages/${pkg}`;
          return this.resolve(modulePath, source, { skipSelf: true });
        }
        return null;
      },
    },
  };
}

export default defineConfig({
  mode: "development",
  plugins: [
    fiftyonePlugin(),
    nodeResolve(),
    react(),
    viteExternalsPlugin({
      react: "React",
      "react-dom": "ReactDOM",
      recoil: "recoil",
      "@fiftyone/state": "__fos__",
      "@fiftyone/operators": "__foo__",
      "@fiftyone/components": "__foc__",
      "@fiftyone/utilities": "__fou__",
      "@fiftyone/spaces": "__fosp__",
      "@mui/material": "__mui__",
      "styled-components": "__styled__",
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
  },
  define: {
    "process.env.NODE_ENV": '"development"',
  },
  optimizeDeps: {
    exclude: ["react", "react-dom"],
  },
});
