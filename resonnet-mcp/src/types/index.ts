/**
 * Type definitions for the MCP server
 */

/**
 * Search result item structure
 */
export type SearchResult = {}

/**
 * Search response structure
 */
export type SearchResponse = {}

/**
 * Import response structure
 */
export type ImportResponse = {
  filePath: string;
  format: string;
  status: 'success' | 'error';
  totalImported: number;
  successful: number;
  failed: number;
  errors?: string[];
} 