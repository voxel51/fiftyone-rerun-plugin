/**
 * Embedded-document marker used by FiftyOne's Rerun integration.
 */
export const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
} as const;

/**
 * Normalized viewer payload used by both panel and sample-renderer flows.
 */
export type RrdSource = {
  url: string;
  identityKey: string;
  version?: string;
};

const shouldUseStreamingChannel = (url: URL): boolean => {
  return url.pathname.endsWith("/media") && url.searchParams.has("filepath");
};

/**
 * Produces a stable viewer key for an RRD URL while preserving streaming URLs.
 */
export const getRrdIdentityKey = (rawUrl: string): string => {
  try {
    const parsed = new URL(rawUrl, window.location.href);
    const normalized = parsed.toString();

    if (parsed.protocol === "http:") {
      return normalized;
    }

    if (parsed.protocol === "https:" && !shouldUseStreamingChannel(parsed)) {
      return normalized.split("?")[0];
    }

    return normalized;
  } catch {
    return rawUrl.split("?")[0] || rawUrl;
  }
};
