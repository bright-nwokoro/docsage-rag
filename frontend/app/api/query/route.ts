import { NextRequest } from "next/server";

export const runtime = "nodejs";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const upstream = await fetch(`${BACKEND}/query`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
