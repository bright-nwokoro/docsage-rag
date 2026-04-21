export interface DocSummary {
  id: string;
  filename: string;
  page_count: number;
  uploaded_at: string;
}

export interface Citation {
  source: string;
  page: number;
  score: number;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  error?: string;
}

export interface IngestResponse {
  doc_id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
}
