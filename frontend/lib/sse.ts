import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { Citation } from "./types";

export interface StreamHandlers {
  onAnswerDelta: (text: string) => void;
  onCitation: (c: Citation) => void;
  onDone: (verified: Citation[]) => void;
  onError: (message: string) => void;
}

export async function streamQuery(
  body: { question: string; history: { role: string; content: string }[] },
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  await fetchEventSource("/api/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
    openWhenHidden: true,
    onmessage(ev) {
      try {
        const data = JSON.parse(ev.data);
        if (ev.event === "answer_delta") handlers.onAnswerDelta(data.text);
        else if (ev.event === "citation") handlers.onCitation(data);
        else if (ev.event === "done") handlers.onDone(data.verified_citations ?? []);
        else if (ev.event === "error") handlers.onError(data.message ?? "unknown error");
      } catch (e) {
        handlers.onError(`bad event: ${String(e)}`);
      }
    },
    onerror(err) {
      handlers.onError(String(err));
      throw err; // stop reconnection
    },
  });
}
