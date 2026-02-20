import { WebViewer } from "@rerun-io/web-viewer";

let viewer: WebViewer | null = null;
const viewerOptions = {
  width: "100%",
  height: "100%",
  hide_welcome_screen: true,
  allow_fullscreen: false,
  enable_history: false,
};

const rootElement = document.getElementById("root");
const errorElement = document.getElementById("error");
const rrdUrlRaw = rootElement?.getAttribute("data-rrd-b64");
const rrdUrlQuery = new URLSearchParams(window.location.search).get("rrd");

const fromBase64 = (encoded: string): string => {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return new TextDecoder().decode(bytes);
};

const decodeRepeatedly = (value: string): string => {
  let current = value;

  for (let index = 0; index < 3; index += 1) {
    try {
      const decoded = decodeURIComponent(current);
      if (decoded === current) {
        break;
      }
      current = decoded;
    } catch {
      break;
    }
  }

  return current;
};

const trimOuterQuotes = (value: string): string =>
  value.trim().replace(/^['"]+|['"]+$/g, "");

const normalizeMediaFilepathUrl = (url: URL): string => {
  if (!url.searchParams.has("filepath")) {
    return url.toString();
  }

  const rebuiltParams: string[] = [];
  for (const [key, value] of url.searchParams.entries()) {
    if (key === "filepath") {
      const normalizedFilepath = trimOuterQuotes(decodeRepeatedly(value));
      rebuiltParams.push(`filepath=${encodeURI(normalizedFilepath)}`);
      continue;
    }

    rebuiltParams.push(
      `${encodeURIComponent(key)}=${encodeURIComponent(value)}`
    );
  }

  const query = rebuiltParams.length > 0 ? `?${rebuiltParams.join("&")}` : "";
  return `${url.origin}${url.pathname}${query}${url.hash}`;
};

const shouldUseStreamingChannel = (url: URL): boolean => {
  return url.pathname.endsWith("/media") && url.searchParams.has("filepath");
};

const normalizeRrdUrl = ({
  rawBase64Url,
  rawQueryUrl,
}: {
  rawBase64Url: string | null;
  rawQueryUrl: string | null;
}): { rrdUrl: string; useStreamingChannel: boolean } | null => {
  let candidate: string | null = null;

  if (rawBase64Url) {
    try {
      candidate = decodeURIComponent(fromBase64(rawBase64Url));
    } catch {
      // fall through to query param
    }
  }

  if (!candidate && rawQueryUrl) {
    try {
      candidate = decodeURIComponent(rawQueryUrl);
    } catch {
      candidate = rawQueryUrl;
    }
  }

  console.log(">>>candidate is ", candidate);

  if (!candidate) {
    return null;
  }

  const trimmed = trimOuterQuotes(candidate);
  if (!trimmed) {
    return null;
  }

  try {
    const normalized = new URL(trimmed, window.location.href);
    if (normalized.protocol !== "http:" && normalized.protocol !== "https:") {
      return null;
    }

    const useStreamingChannel = shouldUseStreamingChannel(normalized);

    const rrdUrl = useStreamingChannel
      ? normalizeMediaFilepathUrl(normalized)
      : normalized.toString();

    return { rrdUrl, useStreamingChannel };
  } catch {
    return null;
  }
};

const normalizedRrd = normalizeRrdUrl({
  rawBase64Url: rrdUrlRaw,
  rawQueryUrl: rrdUrlQuery,
});

const showError = (message: string) => {
  if (rootElement) {
    rootElement.style.display = "none";
  }

  if (errorElement) {
    errorElement.textContent = message;
    errorElement.style.display = "flex";
  }
};

const hideError = () => {
  if (rootElement) {
    rootElement.style.display = "block";
  }

  if (errorElement) {
    errorElement.style.display = "none";
    errorElement.textContent = "";
  }
};

const streamRrdToViewer = async (rrdUrl: string) => {
  if (!viewer) {
    throw new Error("Viewer is not initialized");
  }

  const response = await fetch(rrdUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch RRD (${response.status} ${response.statusText})`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  const channel = viewer.open_channel("fiftyone-rrd");
  try {
    // send_rrd expects one complete RRD payload per call.
    channel.send_rrd(new Uint8Array(arrayBuffer));
  } finally {
    channel.close();
  }
};

const boot = async () => {
  if (!rootElement) {
    throw new Error("Missing iframe root element");
  }

  if (!normalizedRrd) {
    showError("Missing required RRD URL.");
    return;
  }

  const { rrdUrl, useStreamingChannel } = normalizedRrd;

  try {
    hideError();

    viewer?.stop();
    viewer = new WebViewer();

    if (useStreamingChannel) {
      await viewer.start(null, rootElement, viewerOptions);
      await streamRrdToViewer(rrdUrl);
      return;
    }

    await viewer.start(rrdUrl, rootElement, viewerOptions);
  } catch (error) {
    console.error("Failed to boot Rerun iframe viewer:", error);
    const message =
      error instanceof Error ? error.message : "Unknown viewer startup error.";
    showError(`Failed to load RRD from URL:\n${message}`);
  }
};

window.addEventListener("beforeunload", () => {
  viewer?.stop();
  viewer = null;
});

void boot();
