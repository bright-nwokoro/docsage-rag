"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { SendHorizontal } from "lucide-react";
import { streamQuery } from "@/lib/sse";
import type { ChatMessage, Citation } from "@/lib/types";
import { Message } from "./Message";

type Action =
  | { type: "APPEND"; msg: ChatMessage }
  | { type: "APPEND_DELTA"; id: string; text: string }
  | { type: "ADD_CITATION"; id: string; citation: Citation }
  | { type: "SET_CITATIONS"; id: string; citations: Citation[] }
  | { type: "FINISH"; id: string }
  | { type: "FAIL"; id: string; error: string }
  | { type: "HYDRATE"; messages: ChatMessage[] }
  | { type: "CLEAR" };

function reducer(state: ChatMessage[], action: Action): ChatMessage[] {
  switch (action.type) {
    case "APPEND":
      return [...state, action.msg];
    case "APPEND_DELTA":
      return state.map((m) =>
        m.id === action.id ? { ...m, content: m.content + action.text } : m,
      );
    case "ADD_CITATION":
      return state.map((m) =>
        m.id === action.id
          ? { ...m, citations: [...(m.citations ?? []), action.citation] }
          : m,
      );
    case "SET_CITATIONS":
      return state.map((m) => (m.id === action.id ? { ...m, citations: action.citations } : m));
    case "FINISH":
      return state.map((m) => (m.id === action.id ? { ...m, streaming: false } : m));
    case "FAIL":
      return state.map((m) =>
        m.id === action.id ? { ...m, streaming: false, error: action.error } : m,
      );
    case "HYDRATE":
      return action.messages;
    case "CLEAR":
      return [];
  }
}

const STORAGE_KEY = "docsage:chat:v1";

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function Chat() {
  const [messages, dispatch] = useReducer(reducer, []);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Hydrate from localStorage.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) dispatch({ type: "HYDRATE", messages: JSON.parse(raw) });
    } catch {
      // ignore
    }
  }, []);

  // Persist on every change.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // ignore (quota, disabled storage, etc.)
    }
  }, [messages]);

  // Auto-scroll on new content.
  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight });
  }, [messages]);

  const onSubmit = useCallback(async () => {
    const q = input.trim();
    if (!q || streaming) return;

    const userMsg: ChatMessage = { id: makeId(), role: "user", content: q };
    const assistantMsg: ChatMessage = {
      id: makeId(),
      role: "assistant",
      content: "",
      streaming: true,
      citations: [],
    };
    dispatch({ type: "APPEND", msg: userMsg });
    dispatch({ type: "APPEND", msg: assistantMsg });
    setInput("");
    setStreaming(true);

    const history = messages
      .filter((m) => !m.error && !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      await streamQuery(
        { question: q, history },
        {
          onAnswerDelta: (text) =>
            dispatch({ type: "APPEND_DELTA", id: assistantMsg.id, text }),
          onCitation: (c) => dispatch({ type: "ADD_CITATION", id: assistantMsg.id, citation: c }),
          onDone: (verified) =>
            dispatch({ type: "SET_CITATIONS", id: assistantMsg.id, citations: verified }),
          onError: (err) => dispatch({ type: "FAIL", id: assistantMsg.id, error: err }),
        },
        abort.signal,
      );
    } catch (e) {
      dispatch({ type: "FAIL", id: assistantMsg.id, error: String(e) });
    } finally {
      dispatch({ type: "FINISH", id: assistantMsg.id });
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, streaming, messages]);

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSubmit();
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollerRef} className="flex-1 overflow-y-auto space-y-6 p-6">
        {messages.length === 0 && (
          <p className="text-center text-sm text-slate-400 pt-16">
            Upload a PDF and ask a question.
          </p>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
      </div>

      <div className="border-t bg-white p-4">
        <div className="flex items-end gap-2 rounded-2xl border border-slate-300 p-2 focus-within:border-brand-500">
          <textarea
            className="flex-1 resize-none bg-transparent px-2 py-1 text-sm outline-none"
            rows={1}
            placeholder="Ask anything about your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={streaming}
          />
          <button
            onClick={onSubmit}
            disabled={streaming || !input.trim()}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-brand-500 text-white disabled:opacity-40"
            aria-label="Send"
          >
            <SendHorizontal className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-2 flex justify-between text-xs text-slate-400">
          <span>Enter to send · Shift+Enter for newline</span>
          {messages.length > 0 && (
            <button
              onClick={() => {
                dispatch({ type: "CLEAR" });
              }}
              className="hover:text-slate-700"
            >
              Clear chat
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
