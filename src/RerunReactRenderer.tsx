import * as fos from "@fiftyone/state";
import { getFieldsWithEmbeddedDocType } from "@fiftyone/utilities";
import { useEffect, useMemo, useState } from "react";
import { useRecoilValue } from "recoil";
import { RrdIframeRenderer } from "./RrdIframeRenderer";

export const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
};

type RerunFieldDescriptor = {
  _cls: "RrdFile";
  filepath: string;
  version: string;
};

type RrdParams = {
  url: string;
  version?: string;
};

export const RerunViewerReact = () => {
  const currentSample = useRecoilValue(fos.modalSample);
  const [stableRrdParams, setStableRrdParams] = useState<RrdParams | null>(
    null
  );

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

  const rrdParams = useMemo<RrdParams | undefined>(() => {
    if (!rerunFieldPath || !currentSample?.urls) {
      return undefined;
    }

    try {
      const filePathAndVersion = currentSample?.sample?.[
        rerunFieldPath
      ] as unknown as RerunFieldDescriptor;

      const urlsStandardized = fos.getStandardizedUrls(currentSample.urls);

      const rrdFilePath = urlsStandardized[`${rerunFieldPath}.filepath`];

      if (!rrdFilePath) {
        return undefined;
      }

      const url = fos.getSampleSrc(rrdFilePath);
      return {
        url,
        version: filePathAndVersion?.version,
      };
    } catch (error) {
      console.error("Error processing Rerun parameters:", error);
      return undefined;
    }
  }, [currentSample, rerunFieldPath]);

  useEffect(() => {
    if (rrdParams) {
      setStableRrdParams((previous) => {
        if (
          previous?.url === rrdParams.url &&
          previous?.version === rrdParams.version
        ) {
          return previous;
        }

        return rrdParams;
      });
      return;
    }

    setStableRrdParams(null);
  }, [rrdParams]);

  if (!stableRrdParams) {
    return <div>Resolving URL...</div>;
  }

  return (
    <RrdIframeRenderer url={stableRrdParams.url} key={stableRrdParams.url} />
  );
};
