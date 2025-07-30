import React, { Component, ReactNode, Suspense, useRef } from "react";
import { RerunViewerReact } from "./RerunReactRenderer";

export const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
};

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class RerunErrorBoundary extends Component<
  { children: ReactNode; action?: () => void },
  ErrorBoundaryState
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Rerun viewer error:", error, errorInfo);
  }

  resetError = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "100vh",
            padding: "40px 20px",
          }}
        >
          <div
            style={{
              maxWidth: "500px",
              width: "100%",
              padding: "32px 24px",
              textAlign: "center",
              color: "#d32f2f",
              fontSize: "14px",
              border: "1px solid #d32f2f",
              borderRadius: "12px",
              backgroundColor: "#ffebee",
              backdropFilter: "blur(10px)",
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "4px",
                backgroundColor: "#d32f2f",
              }}
            />
            <div
              style={{
                fontWeight: "600",
                fontSize: "18px",
                marginBottom: "12px",
                color: "#b71c1c",
              }}
            >
              Rerun viewer error
            </div>
            <div
              style={{
                fontSize: "14px",
                wordBreak: "break-word",
                lineHeight: "1.5",
                marginBottom: "20px",
                color: "#c62828",
              }}
            >
              {this.state.error?.message || "An unknown error occurred"}
            </div>
            {this.props.action ? (
              <button
                onClick={() => {
                  this.setState({ hasError: false });
                  this.props.action();
                }}
                style={{
                  padding: "10px 24px",
                  backgroundColor: "#d32f2f",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                Retry
              </button>
            ) : (
              <button
                onClick={() => {
                  this.setState({ hasError: false });
                }}
                style={{
                  padding: "10px 24px",
                  backgroundColor: "#d32f2f",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                Attempt Reload
              </button>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export const RerunViewer = React.memo(() => {
  const errorBoundaryRef = useRef<RerunErrorBoundary>(null);

  return (
    <Suspense
      fallback={
        <div
          style={{
            padding: "20px",
            textAlign: "center",
            color: "#666",
            fontSize: "14px",
          }}
        >
          Loading Rerun viewer...
        </div>
      }
    >
      <RerunErrorBoundary ref={errorBoundaryRef}>
        <RerunViewerReact />
      </RerunErrorBoundary>
    </Suspense>
  );
});
