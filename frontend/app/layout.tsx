import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DocSage",
  description: "RAG chatbot over your PDFs with inline citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
