import { NextRequest } from "next/server";

export const runtime = "nodejs";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const upstream = await fetch(`${BACKEND}/ingest`, {
    method: "POST",
    body: req.body,
    // @ts-expect-error: Node fetch requires duplex for streaming bodies
    duplex: "half",
    headers: {
      "content-type": req.headers.get("content-type") ?? "",
    },
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}
