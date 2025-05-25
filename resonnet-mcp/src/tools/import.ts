/**
 * Implementation of the import_conversations tool
 */

import { ImportResponse } from '../types/index.js';
import { createReadStream } from 'fs';
import JSONStream from 'JSONStream';
import { pipeline } from 'stream/promises';

/**
 * Process Claude.ai export files
 * @param filePath Path to the Claude.ai export file
 * @param format Format of the export file (json, csv)
 * @returns Import results
 */
export async function importConversations(filePath: string, format: string = "json"): Promise<ImportResponse> {
  console.log(`Importing conversations from: ${filePath} in ${format} format`);
  
  const results = {
    totalImported: 0,
    successful: 0,
    failed: 0,
  };

  /*
  Experiment: Only embedding user messages
  - Avoid pruning assistant messages with extensive explanations and pleasantries
  - User's own words are better source for preferences, and will reduce context contamination.
  */

  try {
    const fileStream = createReadStream(filePath, { encoding: 'utf-8' });
    const jsonParser = JSONStream.parse('*');

    jsonParser.on('data', (conversation: any) => {
      try{
        console.log(`Processing conversation: ${conversation.name}`);
        //results.successful++;
      } catch (err) {
        console.error('Failed to read / embed conversation:', err);
        // results.failed++;
      } finally {
        //results.totalImported++;
      }
    });

    await pipeline(fileStream, jsonParser);

  } catch (err) {
    console.error('Import failed:', err);
    return {
      filePath,
      format,
      status: 'error' as const,
      ...results,
    };
  }

  return {
    filePath,
    format,
    status: 'success' as const,
    ...results,
  };
}
