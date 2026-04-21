"use client";

import { useState } from "react";
import { PdfDrop } from "@/components/PdfDrop";
import { DocList } from "@/components/DocList";
import { Chat } from "@/components/Chat";

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <main className="h-dvh flex">
      <aside className="w-72 shrink-0 border-r bg-slate-50/50 p-4 overflow-y-auto">
        <div className="mb-6">
          <h1 className="font-semibold text-lg">DocSage</h1>
          <p className="text-xs text-slate-500">
            RAG chatbot over your PDFs. Answers cite their sources.
          </p>
        </div>

        <div className="mb-4">
          <PdfDrop onUploaded={() => setRefreshKey((k) => k + 1)} />
        </div>

        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
            Documents
          </h2>
          <DocList refreshKey={refreshKey} />
        </div>
      </aside>

      <section className="flex-1">
        <Chat />
      </section>
    </main>
  );
}
