#!/usr/bin/env node

/**
 * This MCP server implements two tools:
 * - search_conversations: Search across chat history
 * - import_conversations: Process Claude.ai export files
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

/**
 * Create an MCP server with capabilities for tools only
 */
const server = new Server(
  {
    name: "resonnet-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * Handler that lists available tools.
 * Exposes two tools: search_conversations and import_conversations
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "search_conversations",
        description: "Search across chat history",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query to find in conversation history"
            },
            limit: {
              type: "number",
              description: "Maximum number of results to return"
            }
          },
          required: ["query"]
        }
      },
      {
        name: "import_conversations",
        description: "Process Claude.ai export files",
        inputSchema: {
          type: "object",
          properties: {
            filePath: {
              type: "string",
              description: "Path to the Claude.ai export file"
            },
            format: {
              type: "string",
              description: "Format of the export file (json, csv, etc)",
              enum: ["json", "csv"]
            }
          },
          required: ["filePath"]
        }
      }
    ]
  };
});

/**
 * Handler for tool execution.
 * Implements the search_conversations and import_conversations tools.
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "search_conversations": {
      const query = String(request.params.arguments?.query);
      const limit = Number(request.params.arguments?.limit) || 10;
      
      if (!query) {
        throw new Error("Search query is required");
      }

      return {
        content: [{
          type: "text",
          text: `Searched for "${query}" with limit ${limit}. (Implementation placeholder)`
        }]
      };
    }

    case "import_conversations": {
      const filePath = String(request.params.arguments?.filePath);
      const format = String(request.params.arguments?.format) || "json";
      
      if (!filePath) {
        throw new Error("File path is required");
      }

      return {
        content: [{
          type: "text",
          text: `Imported conversations from ${filePath} in ${format} format. (Implementation placeholder)`
        }]
      };
    }

    default:
      throw new Error("Unknown tool");
  }
});

/**
 * Start the server using stdio transport.
 * This allows the server to communicate via standard input/output streams.
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
