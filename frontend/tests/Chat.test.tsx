import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the SSE client so the component can be driven by a test script.
vi.mock("@/lib/sse", () => ({
  streamQuery: vi.fn(async (_body, handlers) => {
    handlers.onAnswerDelta("Hello ");
    handlers.onAnswerDelta("world");
    handlers.onCitation({ source: "a.pdf", page: 1, score: 0.9 });
    handlers.onDone([{ source: "a.pdf", page: 1, score: 0.9 }]);
  }),
}));

import { Chat } from "@/components/Chat";

describe("Chat", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("streams an answer and renders a citation chip", async () => {
    render(<Chat />);
    const input = screen.getByPlaceholderText(/ask anything/i) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "What does X do?" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText("Hello world")).toBeInTheDocument();
    });
    expect(screen.getByText("a.pdf")).toBeInTheDocument();
    expect(screen.getByText(/p\.1/)).toBeInTheDocument();
  });
});
