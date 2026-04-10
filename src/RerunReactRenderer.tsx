import type { SampleRendererProps } from "@fiftyone/plugins";
import * as fos from "@fiftyone/state";
import { getFieldsWithEmbeddedDocType } from "@fiftyone/utilities";
import { useEffect, useMemo, useState } from "react";
import { useRecoilValue } from "recoil";
import { RerunViewerContainer } from "./RerunViewerContainer";
import {
  getRrdIdentityKey,
  RerunFileDescriptor,
  type RrdSource,
} from "./rerunUtils";

type RerunFieldDescriptor = {
  _cls: "RrdFile";
  filepath: string;
  version: string;
};

/**
 * Resolves the embedded RRD reference for the current modal sample panel.
 */
export const RerunViewerReact = () => {
  const currentSample = useRecoilValue(fos.modalSample);
  const [stableRrdSource, setStableRrdSource] = useState<RrdSource | null>(
    null,
  );

  const schema = useRecoilValue(
    fos.fieldSchema({ space: fos.State.SPACE.SAMPLE }),
  );

  const rerunFieldPath = useMemo(
    () =>
      getFieldsWithEmbeddedDocType(
        schema,
        RerunFileDescriptor.EMBEDDED_DOC_TYPE,
      ).at(0)?.path,
    [schema],
  );

  const rrdSource = useMemo<RrdSource | undefined>(() => {
    if (!rerunFieldPath || !currentSample?.urls) {
      return undefined;
    }

    try {
      const filePathAndVersion = currentSample?.sample?.[
        rerunFieldPath
      ] as unknown as RerunFieldDescriptor;

      const urlsStandardized = fos.getNormalizedUrls(currentSample.urls);

      const rrdFilePath = urlsStandardized[`${rerunFieldPath}.filepath`];

      if (!rrdFilePath) {
        return undefined;
      }

      const url = fos.getSampleSrc(rrdFilePath);
      return {
        url,
        identityKey: getRrdIdentityKey(url),
        version: filePathAndVersion?.version,
      };
    } catch (error) {
      console.error("Error processing Rerun parameters:", error);
      return undefined;
    }
  }, [currentSample, rerunFieldPath]);

  useEffect(() => {
    if (rrdSource) {
      setStableRrdSource((previous) => {
        if (
          previous?.identityKey === rrdSource.identityKey &&
          previous?.version === rrdSource.version
        ) {
          return previous;
        }

        return rrdSource;
      });
      return;
    }

    setStableRrdSource(null);
  }, [rrdSource]);

  if (!stableRrdSource) {
    return <RerunViewerContainer rrdSource={null} />;
  }

  return <RerunViewerContainer rrdSource={stableRrdSource} />;
};

/**
 * Sample renderer entrypoint for direct `.rrd` media rendered in the modal.
 */
export const RerunSampleRenderer = ({ ctx }: SampleRendererProps) => {
  const rrdSource = useMemo<RrdSource | null>(() => {
    if (!ctx.media.url) {
      return null;
    }

    return {
      url: ctx.media.url,
      identityKey: getRrdIdentityKey(ctx.media.url),
    };
  }, [ctx.media.url]);

  return <RerunViewerContainer rrdSource={rrdSource} />;
};
