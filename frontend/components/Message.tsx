"use client";

import type { ChatMessage } from "@/lib/types";
import { CitationChip } from "./CitationChip";
import { cn } from "@/lib/utils";

export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex flex-col gap-2", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap",
          isUser ? "bg-brand-500 text-white" : "bg-slate-100 text-slate-900",
        )}
      >
        {message.content || (message.streaming && (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0s" }} />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0.2s" }} />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0.4s" }} />
          </span>
        ))}
      </div>
      {message.error && <p className="text-xs text-red-600">{message.error}</p>}
      {!isUser && message.citations && message.citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {message.citations.map((c, i) => (
            <CitationChip key={`${c.source}-${c.page}-${i}`} citation={c} />
          ))}
        </div>
      )}
    </div>
  );
}
