"use client";

import { useEffect, useState, useCallback } from "react";
import { Trash2, FileText } from "lucide-react";
import type { DocSummary } from "@/lib/types";

interface Props {
  refreshKey: number;
}

export function DocList({ refreshKey }: Props) {
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/docs", { cache: "no-store" });
      if (resp.ok) setDocs(await resp.json());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  const onDelete = async (id: string) => {
    await fetch(`/api/docs/${id}`, { method: "DELETE" });
    await load();
  };

  if (loading) return <p className="text-xs text-slate-400">Loading…</p>;
  if (docs.length === 0) {
    return <p className="text-xs text-slate-400">No documents yet.</p>;
  }

  return (
    <ul className="space-y-1">
      {docs.map((d) => (
        <li key={d.id} className="group flex items-center gap-2 rounded px-2 py-1.5 hover:bg-slate-50">
          <FileText className="h-4 w-4 shrink-0 text-slate-400" />
          <span className="flex-1 truncate text-sm" title={d.filename}>
            {d.filename}
          </span>
          <span className="text-xs text-slate-400">{d.page_count}p</span>
          <button
            onClick={() => onDelete(d.id)}
            className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600"
            aria-label={`Delete ${d.filename}`}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </li>
      ))}
    </ul>
  );
}
