import * as fos from "@fiftyone/state";
import { getFieldsWithEmbeddedDocType } from "@fiftyone/utilities";
import * as RerunWebViewer from "@rerun-io/web-viewer";
import React, { useEffect, useMemo, useRef } from "react";
import { useRecoilValue } from "recoil";
import { CustomErrorBoundary } from "./CustomErrorBoundary";

export const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
};

type RerunFieldDescriptor = {
  _cls: "RrdFile";
  filepath: string;
  version: string;
};

const Rrd_V_0_18_2 = React.memo(({ url }: { url: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<RerunWebViewer.WebViewer>();

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    (async () => {
      viewerRef.current = new RerunWebViewer.WebViewer();
      await viewerRef.current.start(url, containerRef.current, {
        render_backend: "webgpu",
        allow_fullscreen: false,
        hide_welcome_screen: true,
        width: "100%",
        height: "100%",
      });
    })();

    return () => {
      viewerRef.current?.stop();
    };
  }, [url]);

  return <div ref={containerRef} style={{ height: "100%", width: "100%" }} />;
});

const RrdRenderer = React.memo(
  ({ url, version }: { url: string; version: string }) => {
    switch (version) {
      case "0.18.2":
      default:
        return <Rrd_V_0_18_2 url={url} />;
    }
  }
);

export const RerunViewer = () => {
  const currentSample = useRecoilValue(fos.modalSample);

  const schema = useRecoilValue(
    fos.fieldSchema({ space: fos.State.SPACE.SAMPLE })
  );

  const rerunFieldPath = useMemo(
    () =>
      getFieldsWithEmbeddedDocType(
        schema,
        RerunFileDescriptor.EMBEDDED_DOC_TYPE
      ).at(0)?.path,
    [schema]
  );

  const rrdParams = useMemo(() => {
    if (!rerunFieldPath || !currentSample.urls) {
      return undefined;
    }

    const filePathAndVersion = currentSample?.sample?.[
      rerunFieldPath
    ] as unknown as RerunFieldDescriptor;

    const urlsStandardized = fos.getStandardizedUrls(currentSample.urls);

    const rrdFilePath = urlsStandardized[`${rerunFieldPath}.filepath`];

    const url = fos.getSampleSrc(rrdFilePath);
    return {
      url,
      version: filePathAndVersion.version,
    };
  }, [currentSample, rerunFieldPath]);

  if (!rrdParams) {
    return <div>Loading</div>;
  }

  return (
    <CustomErrorBoundary>
      <RrdRenderer url={rrdParams.url} version={rrdParams.version} />
    </CustomErrorBoundary>
  );
};
