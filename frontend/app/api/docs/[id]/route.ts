const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function DELETE(_: Request, { params }: { params: { id: string } }) {
  const upstream = await fetch(`${BACKEND}/docs/${encodeURIComponent(params.id)}`, {
    method: "DELETE",
  });
  return new Response(null, { status: upstream.status });
}
