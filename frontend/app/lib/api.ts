export interface SearchResult {
  rank: number;
  relevance_score: number;
  conversation_id: string;
  conversation_name: string;
  message_id: string;
  // timestamp: string;
  // message_index: number;
  text: string;
  // original_text: string;
  url: string;
}

export interface IndexingStatus {
  task_id: string;
  status: string;
  progress: number;
  total_messages: number;
  processed_messages: number;
  error?: string;
}

interface HasConversationsResponse {
  has_conversations: boolean;
  conversation_count: number;
  message_count: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const claudeSearchApi = {
  async searchConversations(query: string, topK: number = 5): Promise<SearchResult[]> {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to search conversations');
    }

    const data = await response.json();
    return data.results;
  },

  async hasConversations(): Promise<HasConversationsResponse> {
    const response = await fetch(`${API_BASE_URL}/has-conversations`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to check conversations');
    }

    return await response.json();
  },

  async deleteConversations(): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to delete conversations');
    }
  },

  async getStats(): Promise<{ total_indexed_messages: number }> {
    const response = await fetch(`${API_BASE_URL}/stats`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get stats');
    }

    return await response.json();
  },

  async uploadFile(file: File): Promise<{ task_id: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to upload file');
    }

    return await response.json();
  },

  async getIndexingStatus(taskId: string): Promise<IndexingStatus> {
    const response = await fetch(`${API_BASE_URL}/indexing-status/${taskId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get indexing status');
    }

    return await response.json();
  }
};