"use client";

import { useState } from "react";
import { BookOpen } from "lucide-react";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

export function CitationChip({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false);
  const score = Math.round(citation.score * 100);
  return (
    <div className="inline-block">
      <button
        onClick={() => setExpanded((v) => !v)}
        className={cn(
          "inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-50",
        )}
        title={`Confidence ${score}%`}
      >
        <BookOpen className="h-3 w-3 text-brand-500" />
        <span className="font-medium">{citation.source}</span>
        <span className="text-slate-400">p.{citation.page}</span>
        <span className="text-slate-400">·</span>
        <span className="text-slate-500">{score}%</span>
      </button>
    </div>
  );
}
