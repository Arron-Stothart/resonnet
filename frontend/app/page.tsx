'use client';

import { useState, useEffect } from 'react';
import { claudeSearchApi, SearchResult } from '@/app/lib/api';
import SearchBar from '@/components/search/SearchBar';
import SearchResults from '@/components/search/SearchResults';
import IndexingProgress from '@/components/upload/IndexingProgress';
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

export default function Page() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [conversationCount, setConversationCount] = useState(0);
  const [isManageOpen, setIsManageOpen] = useState(false);
  const [conversationStats, setConversationStats] = useState<{ 
    has_conversations: boolean; 
    conversation_count: number; 
    message_count: number; 
  } | null>(null);
  const [isCheckingStats, setIsCheckingStats] = useState(true);

  const performSearch = async (query: string) => {
    try {
      setIsLoading(true);
      const searchResults = await claudeSearchApi.searchConversations(query);
      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const stats = await claudeSearchApi.hasConversations();
        setConversationCount(stats.conversation_count);
        setConversationStats(stats);
      } catch (error) {
        console.error('Failed to fetch conversation count:', error);
      } finally {
        setIsCheckingStats(false);
      }
    };
    fetchData();
    performSearch('');
  }, []);

  return (
    <div className="min-h-screen text-[#E5E5E2]">
      <div className="w-full mx-auto px-4 py-4">
        <div className="flex flex-row justify-start items-center gap-4 mb-6">
          <h1 className="text-xl font-display font-serif text-[#CECCC5]">
            ReSonnet
          </h1>
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 256 256">
              <path d="M232.07,186.76a80,80,0,0,0-62.5-114.17A80,80,0,1,0,23.93,138.76l-7.27,24.71a16,16,0,0,0,19.87,19.87l24.71-7.27a80.39,80.39,0,0,0,25.18,7.35,80,80,0,0,0,108.34,40.65l24.71,7.27a16,16,0,0,0,19.87-19.86ZM62,159.5a8.28,8.28,0,0,0-2.26.32L32,168l8.17-27.76a8,8,0,0,0-.63-6,64,64,0,1,1,26.26,26.26A8,8,0,0,0,62,159.5Zm153.79,28.73L224,216l-27.76-8.17a8,8,0,0,0-6,.63,64.05,64.05,0,0,1-85.87-24.88A79.93,79.93,0,0,0,174.7,89.71a64,64,0,0,1,41.75,92.48A8,8,0,0,0,215.82,188.23Z"></path>
            </svg>
            <h1 className="text-xl font-display font-serif">
              Your Chat History
            </h1>
          </div>
        </div>
        <div className="max-w-4xl mx-auto flex flex-col gap-2 mt-10">
          <SearchBar onSearch={performSearch} />
          <div className="flex flex-row gap-2 bg-[#262624] rounded-none p-4 mx-[-16px]">
            <p className="text-md text-[#E5E5E2]">
              You have {conversationCount} previous chats with Claude
            </p>
            <button 
              className="text-md text-[#207FDE] hover:underline"
              onClick={() => setIsManageOpen(true)}
            >
              Manage
            </button>
          </div>
          <SearchResults results={results} isLoading={isLoading} />
        </div>
        <Dialog open={isManageOpen} onOpenChange={setIsManageOpen}>
          <DialogContent className="mx-w-3xl">
            {!isCheckingStats && <IndexingProgress initialStats={conversationStats} />}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
