const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET() {
  const upstream = await fetch(`${BACKEND}/docs`, { cache: "no-store" });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}
