import { WebViewer } from "@rerun-io/web-viewer";
import { LRUCache } from "lru-cache";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { MAX_RRDS_IN_CACHE } from "./constants";

export const Rrd_V_0_19 = React.memo(({ url }: { url: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<WebViewer>(null);
  // state to track if the viewer instance is initialized
  const [initialized, setInitialized] = useState(false);
  // state to track if .start() has been called on the viewer instance,
  // because after first .start() call, we call .open() instead of .start()
  const [started, setStarted] = useState(false);

  const firstRrdUrlRef = useRef<string | null>(null);
  const cacheRef = useRef<LRUCache<string, boolean>>();

  /**
   * This effect initializes the viewer instance. Same viewer instance is used to
   * open/close multiple rrds.
   */
  useEffect(() => {
    viewerRef.current = new WebViewer();

    cacheRef.current = new LRUCache<string, boolean>({
      max: MAX_RRDS_IN_CACHE,
      dispose: (_val, urlKey) => {
        try {
          viewerRef.current.close(urlKey);
        } catch (e) {
          console.error(`Couldn't close viewer for url ${url}`, e);
        }
      },
    });

    setInitialized(true);

    return () => {
      // todo: the following is throwing runtime error that looks like a bug
      // viewerRef.current?.stop();

      try {
        // close all open URLs
        cacheRef.current?.forEach((_value, urlKey) => {
          try {
            viewerRef.current.close(urlKey);
          } catch (e) {
            console.error(`Couldn't close viewer for url ${urlKey}`, e);
          }
        });
        viewerRef.current.stop();
      } catch (e) {
        console.error("Couldn't stop viewer", e);
      }
    };
  }, []);

  /**
   * This effect oepns the first rrd file with .start()
   * and sets the `started` state to true.
   */
  useLayoutEffect(() => {
    if (started || !initialized || !containerRef.current) {
      return;
    }

    firstRrdUrlRef.current = url;

    viewerRef.current.start(url, containerRef.current, {
      render_backend: "webgpu",
      allow_fullscreen: false,
      hide_welcome_screen: true,
      width: "100%",
      height: "100%",
    });

    cacheRef.current?.set(url, true);

    setStarted(true);
  }, [url, started, initialized]);

  /**
   * This effect opens rrd files using .open() depending on the URL.
   */
  useLayoutEffect(() => {
    if (!started || firstRrdUrlRef.current === url) {
      // yield to effect above which uses .start() API
      return;
    }

    viewerRef.current.open(url);

    cacheRef.current?.set(url, true);

    // no need to close the URL here, as the cache will handle it
  }, [url, started]);

  return <div ref={containerRef} style={{ height: "100%", width: "100%" }} />;
});
