import { BookOpen } from "lucide-react";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

export function CitationChip({ citation }: { citation: Citation }) {
  const score = Math.round(citation.score * 100);
  return (
    <div className="inline-block">
      <span
        className={cn(
          "inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700",
        )}
        title={`Relevance ${score}%`}
      >
        <BookOpen className="h-3 w-3 text-brand-500" />
        <span className="font-medium">{citation.source}</span>
        <span className="text-slate-400">p.{citation.page}</span>
        <span className="text-slate-400">·</span>
        <span className="text-slate-500">{score}%</span>
      </span>
    </div>
  );
}
