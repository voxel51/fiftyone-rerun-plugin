import * as fos from "@fiftyone/state";
import { getFieldsWithEmbeddedDocType } from "@fiftyone/utilities";
import React, { useMemo } from "react";
import { useRecoilValue } from "recoil";
import { CustomErrorBoundary } from "./CustomErrorBoundary";
import { RrdIframeRenderer } from "./RrdIframeRenderer";

export const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
};

type RerunFieldDescriptor = {
  _cls: "RrdFile";
  filepath: string;
  version: string;
};

export const RerunViewer = React.memo(() => {
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
      version: filePathAndVersion?.version,
    };
  }, [currentSample, rerunFieldPath]);

  if (!rrdParams) {
    return <div>Loading...</div>;
  }

  return (
    <CustomErrorBoundary>
      {/* use iframe until versioned web viewer renderer is more stable, note: vite will not bundle any rerun deps */}
      {/* <RrdWebViewerRenderer url={rrdParams.url} version={rrdParams.version} /> */}
      <RrdIframeRenderer url={rrdParams.url} />
    </CustomErrorBoundary>
  );
});
