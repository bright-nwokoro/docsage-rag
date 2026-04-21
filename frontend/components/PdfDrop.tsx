"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Loader2 } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";
import type { IngestResponse } from "@/lib/types";

interface Props {
  onUploaded: (resp: IngestResponse) => void;
}

export function PdfDrop({ onUploaded }: Props) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (files: File[]) => {
      setError(null);
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        setUploading(true);
        try {
          const resp = await fetch("/api/ingest", { method: "POST", body: form });
          if (!resp.ok) {
            const t = await resp.text();
            throw new Error(`upload failed: ${resp.status} ${t}`);
          }
          const body: IngestResponse = await resp.json();
          onUploaded(body);
        } catch (e: unknown) {
          setError(e instanceof Error ? e.message : String(e));
        } finally {
          setUploading(false);
        }
      }
    },
    [onUploaded],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    disabled: uploading,
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition",
          isDragActive ? "border-brand-500 bg-brand-50" : "border-slate-300 hover:border-slate-400",
          uploading && "opacity-60 cursor-not-allowed",
        )}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-brand-500" />
            <p className="text-sm text-slate-500">Uploading & indexing…</p>
          </>
        ) : (
          <>
            <UploadCloud className="h-5 w-5 text-slate-400" />
            <p className="text-sm font-medium">Drop a PDF here or click to select</p>
            <p className="text-xs text-slate-400">Max {formatBytes(25 * 1024 * 1024)} per file</p>
          </>
        )}
      </div>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
